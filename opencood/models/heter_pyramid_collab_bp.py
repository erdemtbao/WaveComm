""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
from pytorch_msssim import ssim

from torch.autograd import Function
import pywt

from einops.layers.torch import Rearrange 

class HeterPyramidCollab(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollab, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)            

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }



            
        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # token_nums = 32
        # H,W,C = 128,256,64
        # setattr(self, f"token_oper", TokenLearner(S=token_nums))
        # # setattr(self, f"tokenfuse_oper", TokenFuser(H=H, W=W, C=C, S=token_nums))
        # setattr(self, f"token_decoder", TiTokDecoder())
        
        # self.mlp_heter = MLP(input_dim=64, hidden_dim=32, output_dim=64)
        # self.protype_feat = nn.Parameter(torch.randn(H, W, C))

        wave = "haar"  # "haar"   "rbio1.1"
        setattr(self, f"model_enc", DWT_2D(wave = wave).cuda())
        setattr(self, f"model_dec", IDWT_2D(wave = wave).cuda())

        '''
        in_channels = args['fusion_backbone']['num_filters'][0]
        setattr(
            self,
            f"conv_highfreq",
            nn.Sequential(
                nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            )
        )
        setattr(
            self,
            f"single_head_highfreq",
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        '''
        # setattr(
        #     self,
        #     f"fusion_highfreq",
        #     nn.Sequential(
        #         nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        #         nn.BatchNorm2d(in_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # )


        # setattr(self, f"vae_{modality_name}", VAE(input_channels=C)) #,latent_dim=token_nums*C
        # setattr(self, f"vae_{modality_name}", VAE(input_dim=H*W*C, latent_dim=token_nums*C,hidden_dim=H*W*C//8))
        # setattr(self, f"mhsa_{modality_name}", nn.MultiheadAttention(embed_dim=C, num_heads=1))


        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

        self.wave_generator = WaveletGenerator(out_channels=64)
        self.discriminator  = WaveletDiscriminator(in_channels=64)

        # 蒸馏参数
        self.distill_cfg = {
            'lambda_recon': 10.0,    # 重建损失权重 #10.0
            # 'lambda_distill': 5.0,   # 蒸馏损失权重
            'lambda_adv': 1.0,       # 对抗损失权重
            'enable_distill': True    # 是否启用蒸馏
        }

    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # token_nums = 32
        # tklr = TokenLearner(S=token_nums)
        # tkfr = TokenFuser(H=256, W=512, C=64, S=token_nums)

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name) # N,64,256,512  N:bsz*num_agent
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)

            modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list).contiguous()  # bsz*agent,64,128,256

        x_ll, x_lh, x_hl, x_hh = eval(f"self.model_enc")(heter_feature_2d.to(dtype=torch.float16))
        reconstructed_feature  = self.wave_generator(x_ll.to(dtype=torch.float32))
        
        # ssim_value = ssim(recon_xll,heter_feature_2d, data_range=1.0)
        # mse_value = F.mse_loss(recon_xll, heter_feature_2d)


        # # style 1: 小波变换后使用高频分量-融合方式1
        # # x_ll, x_lh, x_hl, x_hh = eval(f"self.model_enc")(heter_feature_2d.to(dtype=torch.float16))
        # # high_freq = torch.cat([x_lh, x_hl, x_hh], dim=1).to(dtype=torch.float32) #[4, 64*3, 64, 128]
        # # high_freq_conv = eval(f"self.conv_highfreq")(high_freq) #[4, 64, 64, 128]
        # # high_freq_forward = eval(f"self.single_head_highfreq")(high_freq_conv) #[4, 1, 64, 128]
        # # output_dict.update({'high_freq_list': [high_freq_forward,]})
        # # x_ll = x_ll + (high_freq_forward * high_freq_conv).to(dtype=torch.float16)
        # # heter_feature_2d = eval(f"self.model_dec")(x_ll.contiguous(), None, None, None).to(dtype=torch.float32).contiguous()  

        # # style 2: 小波变换后融合高频分量-融合方式2
        # # fused_features = torch.cat([x_ll, high_freq_forward * high_freq_conv], dim=1)
        # # x_ll = eval(f"self.fusion_highfreq")(fused_features).to(dtype=torch.float16)
        # # heter_feature_2d = eval(f"self.model_dec")(x_ll.contiguous(), None, None, None).to(dtype=torch.float32).contiguous()  

        # # style 3: 多级小波变换
        # # x_ll1, x_lh, x_hl, x_hh = eval(f"self.model_enc")(x_ll)
        # # tmp = eval(f"self.model_dec")(x_ll1, None, None, None)

        #  =======================================================================================
        # # @Deprecated
        # bth,c,h,w = heter_feature_2d.shape
        # heter_feature_2d = heter_feature_2d.permute(0,2,3,1).contiguous()  # bth,h,w,c
        # token_feature = eval(f"self.token_oper")(heter_feature_2d) # bth,num_tokens,c
        # rec_token = eval(f"self.token_decoder")(token_feature.permute(0,2,1).unsqueeze(2)) # bth,c,h,w
        # output_dict.update({'ori': heter_feature_2d.clone(), #bth,h,w,c
        #                     'new_heter': rec_token.permute(0,2,3,1).contiguous(), #bth,h,w,c
        #                     })
        # heter_feature_2d = rec_token.contiguous()
        
        # ref_token_fuse = eval(f"self.tokenfuse_oper")(token_feature,heter_feature_2d) #bth,h,w,c
        
        # from opencood.models.fuse_modules.fusion_in_one import regroup
        # split_x_token = regroup(token_feature, record_len) 
        # split_x = regroup(heter_feature_2d, record_len)

        # B, L = affine_matrix.shape[:2]
        # new_heter = []
        # for b in range(B):
        #     N = record_len[b]    
        #     new_heter.append(split_x[b][0].unsqueeze(0))
            
        #     if N!=1:
        #         ego_node_tmp = split_x[b][0].repeat(len(split_x_token[b][1:]),1,1,1)
        #         else_node_token_tmp = split_x_token[b][1:]
        #         else_token_fuse = eval(f"self.tokenfuse_oper")(self.mlp_heter(else_node_token_tmp),self.mlp_heter(ego_node_tmp))
        #         new_heter.append(else_token_fuse)
        # new_heter = torch.stack(new_heter).squeeze(1)
        # # new_heter = self.mlp_heter(new_heter)

        # output_dict.update({'ref_token_fuse': ref_token_fuse, #bth,h,w,c
        #                     'new_heter': new_heter, #bth,h,w,c
        #                     })
        
        # heter_feature_2d = new_heter.permute(0,3,1,2).contiguous() #bth,c,h,w
        #  ========================================

        # token_fuse = eval(f"self.tokenfuse_oper")(token_feature,heter_feature_2d) 
        # token_fuse = self.mlp_heter(token_fuse)
        # heter_feature_2d = token_fuse.permute(0,3,1,2).contiguous() #bth,c,h,w
        

        '''
        bth,c,h,w = heter_feature_2d.shape

        protype_feat = self.protype_feat.unsqueeze(0).repeat(bth, 1, 1, 1).to(heter_feature_2d.device)

        heter_feature_2d = heter_feature_2d.permute(0,2,3,1).contiguous()
        token_feature = eval(f"self.token_{modality_name}")(heter_feature_2d) # bth,c,num_tokens

        token_fuse = eval(f"self.tokenfuse_{modality_name}")(token_feature,protype_feat)  

        output_dict.update({'recon_x': token_fuse,
                            'ori_x': heter_feature_2d,
                            })

        heter_feature_2d = token_fuse.permute(0,3,1,2).contiguous()
        '''

        '''
        bth,c,h,w = heter_feature_2d.shape
        heter_feature_2d = heter_feature_2d.permute(0,2,3,1).contiguous()  #bth,h,w,c
        token_feature = eval(f"self.token_{modality_name}")(heter_feature_2d) # bth,c,num_tokens

        # token_feature = token_feature.reshape(-1,bth,c)
        # token_feature, _ = eval(f"self.mhsa_{modality_name}")(token_feature,token_feature,token_feature) 
        # token_feature = token_feature.reshape(bth,-1,c)
        
        token_fuse = eval(f"self.tokenfuse_{modality_name}")(token_feature,heter_feature_2d) 
        heter_feature_2d = token_fuse.permute(0,3,1,2).contiguous() #bth,c,h,w
        '''

        # bth,c,h,w = heter_feature_2d.shape
        # heter_feature_2d_flat = heter_feature_2d.view(bth, -1) 
        # reconstructed, mu, logvar = eval(f"self.vae_{modality_name}")(heter_feature_2d_flat)

        # reconstructed, mu, logvar = eval(f"self.vae_{modality_name}")(heter_feature_2d)
        # output_dict.update({'recon_x': reconstructed,
        #                     'ori_x': heter_feature_2d,
        #                     'mu': mu,
        #                     'logvar': logvar})
        # # reconstructed = reconstructed.reshape(bth,c,h,w)
        # heter_feature_2d = reconstructed

        if self.distill_cfg['enable_distill']:
            output_dict.update({
                'original_feature': heter_feature_2d,
                'reconstructed_feature': reconstructed_feature,
            })


        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                                                heter_feature_2d,
                                                record_len, 
                                                affine_matrix, 
                                                agent_modality_list, 
                                                self.cam_crop_info
                                            )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        
        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)


        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

        output_dict.update({'distillation_loss': 
                            self.get_distillation_loss(output_dict)})
        return output_dict

    def get_distillation_loss(self, output_dict):
        """计算蒸馏相关的损失"""
        if not (self.training and self.distill_cfg['enable_distill']):
            return {}
            
        original_feature = output_dict['original_feature']
        reconstructed_feature = output_dict['reconstructed_feature']
        
        # 1. 重建损失
        recon_loss = F.l1_loss(reconstructed_feature, original_feature)
        
        # 2. 感知损失（特征空间的相似度）
        perceptual_loss = F.mse_loss(
            F.normalize(reconstructed_feature.flatten(2), dim=-1),
            F.normalize(original_feature.flatten(2), dim=-1)
        )
        
        # 3. 对抗损失
        d_real = self.discriminator(original_feature)
        d_fake = self.discriminator(reconstructed_feature.detach())
        d_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real)) + \
                 F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        
        g_fake = self.discriminator(reconstructed_feature)
        g_loss = F.binary_cross_entropy(g_fake, torch.ones_like(g_fake))
        
        # # 4. 检测头蒸馏损失
        # with torch.no_grad():
        #     orig_cls = self.cls_head(original_feature)
        #     orig_reg = self.reg_head(original_feature)
        #     orig_dir = self.dir_head(original_feature)
            
        # recon_cls = self.cls_head(reconstructed_feature)
        # recon_reg = self.reg_head(reconstructed_feature)
        # recon_dir = self.dir_head(reconstructed_feature)
        
        # distill_cls_loss = F.kl_div(
        #     F.log_softmax(recon_cls, dim=1),
        #     F.softmax(orig_cls, dim=1),
        #     reduction='batchmean'
        # )
        # distill_reg_loss = F.mse_loss(recon_reg, orig_reg)
        # distill_dir_loss = F.mse_loss(recon_dir, orig_dir)
        
        # 5. 组合所有损失
        total_loss = {
            'recon_loss': self.distill_cfg['lambda_recon'] * (recon_loss + 0.1 * perceptual_loss),
            # 'distill_loss': self.distill_cfg['lambda_distill'] * (
            #     distill_cls_loss + distill_reg_loss + distill_dir_loss
            # ),
            'adv_loss': self.distill_cfg['lambda_adv'] * g_loss,
            'd_loss': d_loss  # 判别器损失单独优化
        }
        
        return total_loss


class WaveletGenerator(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        # Generator network to reconstruct from LL component
        self.decoder = nn.Sequential(
            # Input: [B, 64, 64, 128]
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # [B, 128, 64, 128]
            nn.Conv2d(128, out_channels*2, 3, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(),
        )

        self.upsample = nn.Sequential(
            # 使用转置卷积在空间维度上进行2倍上采样
            nn.ConvTranspose2d(out_channels*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # [B, out_channels, 128, 256]
        )

        # 最终输出层
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x_ll):
        x = self.decoder(x_ll)          # [B, 128, 64, 128]
        x = self.upsample(x)            # [B, 64, 128, 256]
        x = self.output(x)              # [B, 64, 128, 256]
        return x
    
class WaveletDiscriminator(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        # Discriminator network
        self.disc = nn.Sequential(
            # Input: [B, in_channels, 64, 128]
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # [B, 64, 32, 64]
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # [B, 128, 16, 32]
            nn.Conv2d(128, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.disc(x)    





# 非局部模块（Non-local）
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.intermediate_channels = in_channels // 2
        
        self.g = nn.Conv2d(in_channels, self.intermediate_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.intermediate_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.intermediate_channels, kernel_size=1)
        self.o = nn.Conv2d(self.intermediate_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        g_x = self.g(x).view(B, self.intermediate_channels, -1)  # [B, C//2, H*W]
        theta_x = self.theta(x).view(B, self.intermediate_channels, -1)  # [B, C//2, H*W]
        phi_x = self.phi(x).view(B, self.intermediate_channels, -1)  # [B, C//2, H*W]
        
        # 计算注意力
        attention = torch.matmul(theta_x.permute(0, 2, 1), phi_x)  # [B, H*W, H*W]
        attention = F.softmax(attention, dim=-1)  # [B, H*W, H*W]
        
        # 聚合特征
        out = torch.matmul(attention, g_x.permute(0, 2, 1))  # [B, H*W, C//2]
        out = out.permute(0, 2, 1).view(B, self.intermediate_channels, H, W)  # [B, C//2, H, W]
        
        out = self.o(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
        # self.sgap = nn.AvgPool2d(2)
        # self.mlp = MLP(input_dim=64, hidden_dim=32, output_dim=1)

        self.use_non_local = False
        if self.use_non_local:
            # 非局部模块：捕捉全局空间依赖
            self.non_local = NonLocalBlock() #NonLocalBlock

    def forward(self, x):
        B, H, W, C = x.shape

        x = x.view(B, C, H, W)
        mx = torch.max(x, 1)[0].unsqueeze(1).to(x.device)
        avg = torch.mean(x, 1).unsqueeze(1).to(x.device)
        combined = torch.cat([mx, avg], dim=1).to(x.device)
        fmap = self.conv(combined)

        # fmap = self.mlp(x)
        # fmap = fmap.view(B, 1, H, W)

        weight_map = torch.sigmoid(fmap)

        # 如果使用非局部模块，应用非局部操作
        if self.use_non_local:
            out = self.non_local(x * weight_map)
            return out.mean(dim=(-2, -1)), x * weight_map 
        
        # x = x.view(B, C, H, W)
        out = (x * weight_map).mean(dim=(-2, -1))

        return out, x * weight_map

class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])
        
    def forward(self, x):
        B, _, _, C = x.shape
        Z = torch.Tensor(B, self.S, C).to(x.device)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x) # [B, C]
            Z[:, i, :] = Ai
        return Z

class TokenFuser(nn.Module):
    def __init__(self, H, W, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S
        
    def forward(self, y, x):
        # if torch.all(y == 0):
        #     return x
        # else:
        B, S, C = y.shape
        B, H, W, C = x.shape
        
        Y = self.projection(y.reshape(B, C, S)).reshape(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).view(B, H*W, S) # [B, HW, S]
        BwY = torch.matmul(Bw, Y)
        
        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H*W, C)
        
        out = (BwY + xj).view(B, H, W, C)
            
        return out 
        

# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim=256):
#         super(VAE, self).__init__()
#         # 编码器部分
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值
#         self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 方差

#         # 解码器部分
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         print("z:",z.shape)
#         return self.decode(z), mu, logvar
    
# class VAE(nn.Module):
#     def __init__(self, input_channels=64, latent_dim=128):
#         super(VAE, self).__init__()
#         self.latent_dim = latent_dim

#         # 第一层卷积，输入64通道，输出64个通道，卷积后特征图大小会减半
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

#         # 假设输入大小是(64, 128, 256)，经过3层卷积后，特征图大小为(256, 16, 32)
#         self.fc_mu = nn.Linear(256 * 16 * 32, latent_dim)  # 计算后的特征图大小：256 * 16 * 32
#         self.fc_logvar = nn.Linear(256 * 16 * 32, latent_dim)  # 与fc_mu一样

#         # 解码器：先将潜在向量变回原来的维度
#         self.fc_dec = nn.Linear(latent_dim, 256 * 16 * 32)  # latent_dim 到 256 * 16 * 32
#         self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1)

#     def encode(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)  # Flatten to vector
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, torch.clamp(logvar, min=-10000, max=10000)

#     def decode(self, z):
#         x = F.relu(self.fc_dec(z))
#         x = x.view(x.size(0), 256, 16, 32)  # Reshape to (batch_size, 256, 16, 32)
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = torch.sigmoid(self.deconv3(x))  # Sigmoid to get values in the same range as original input
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decode(z)
#         return recon_x, mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        # x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # return x
        return x_ll, x_lh, x_hl, x_hh

    @staticmethod
    def backward(ctx, dx_ll, dx_lh, dx_hl, dx_hh):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape

            dx_cat = torch.cat([dx_ll, dx_lh, dx_hl, dx_hh], dim=1)

            dx = dx_cat.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x_ll, x_lh, x_hl, x_hh, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x_ll.shape

        # Create zero tensors for missing components
        zeros = torch.zeros_like(x_ll, device=x_ll.device)
        if x_lh is None:
            x_lh = zeros
        if x_hl is None:
            x_hl = zeros
        if x_hh is None:
            x_hh = zeros
        
        # Concatenate all components
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        filters, = ctx.saved_tensors  # Unpack the saved tensors properly
        B, C, H, W = ctx.shape
        dx = dx.contiguous()

        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        
        # Initialize all gradients
        grad_x_ll = grad_x_lh = grad_x_hl = grad_x_hh = grad_filters = None

        if ctx.needs_input_grad[0]:  # x_ll gradient
            grad_x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
        
        if ctx.needs_input_grad[1]:  # x_lh gradient
            grad_x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            
        if ctx.needs_input_grad[2]:  # x_hl gradient
            grad_x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            
        if ctx.needs_input_grad[3]:  # x_hh gradient
            grad_x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)

        # Return gradients for all inputs in the same order as forward
        return grad_x_ll, grad_x_lh, grad_x_hl, grad_x_hh, grad_filters
    
    # @staticmethod
    # def backward(ctx, dx):
    #     if ctx.needs_input_grad[0]:
    #         filters = ctx.saved_tensors
    #         filters = filters[0]
    #         B, C, H, W = ctx.shape
    #         dx = dx.contiguous()

    #         w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
    #         x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
    #         x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
    #         x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
    #         x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
    #         dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
    #     return x_ll, x_lh, x_hl, x_hh, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x_ll, x_lh, x_hl, x_hh):
        return IDWT_Function.apply(x_ll, x_lh, x_hl, x_hh, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float16)
        self.w_lh = self.w_lh.to(dtype=torch.float16)
        self.w_hl = self.w_hl.to(dtype=torch.float16)
        self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


# ### 下述组合的stride=4
# class DWT_Function(Function):
#     @staticmethod
#     def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
#         x = x.contiguous()
#         ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
#         ctx.shape = x.shape

#         dim = x.shape[1]
#         stride = 4
#         ctx.stride = stride

#         x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = stride, groups = dim)
#         x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = stride, groups = dim)
#         x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = stride, groups = dim)
#         x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = stride, groups = dim)
#         # x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
#         # return x

#         ctx.sub_shape = x_ll.shape
#         return x_ll, x_lh, x_hl, x_hh

#     @staticmethod
#     def backward(ctx, dx_ll, dx_lh, dx_hl, dx_hh):
#         if ctx.needs_input_grad[0]:
#             w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
#             B, C, H, W = ctx.shape
#             _, _, sub_h, sub_w = ctx.sub_shape
#             stride = ctx.stride

#             dx_cat = torch.cat([dx_ll, dx_lh, dx_hl, dx_hh], dim=1)
#             dx = dx_cat.view(B, 4, -1, sub_h, sub_w)
            
#             dx = dx.transpose(1,2).reshape(B, -1, sub_h, sub_w)
#             filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)

#             # 计算 padding 和 output_padding
#             kernel_size = filters.shape[-1]  # 滤波器的大小
#             padding = (kernel_size - 1) // 2  # 通常的 padding 计算方式
#             output_padding_H = H - ((sub_h - 1) * stride - 2 * padding + kernel_size)
#             output_padding_W = W - ((sub_w - 1) * stride - 2 * padding + kernel_size)


#             dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=stride, groups=C, padding=padding, output_padding=(output_padding_H, output_padding_W),)

#         return dx, None, None, None, None
# class IDWT_Function(Function):
#     @staticmethod
#     def forward(ctx, x_ll, x_lh, x_hl, x_hh, filters):
#         ctx.save_for_backward(filters)
#         ctx.shape = x_ll.shape

#         # Create zero tensors for missing components
#         zeros = torch.zeros_like(x_ll, device=x_ll.device)
#         if x_lh is None:
#             x_lh = zeros
#         if x_hl is None:
#             x_hl = zeros
#         if x_hh is None:
#             x_hh = zeros
        
#         # Concatenate all components
#         x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

#         B, _, H, W = x.shape
#         x = x.view(B, 4, -1, H, W).transpose(1, 2)
#         C = x.shape[1]
#         x = x.reshape(B, -1, H, W)
#         filters = filters.repeat(C, 1, 1, 1)

#         stride = 4
#         ctx.stride = stride
#         original_H, original_W =  H*stride, W*stride
#         # 计算所需的 padding 和 output_padding
#         kernel_size = filters.shape[-1]  # 滤波器的大小
#         padding = (kernel_size - 1) // 2  # 通常的 padding 计算方式
#         output_padding_H = original_H - ((H - 1) * stride - 2 * padding + kernel_size)
#         output_padding_W = original_W - ((W - 1) * stride - 2 * padding + kernel_size)

#         x = torch.nn.functional.conv_transpose2d(x, filters, stride=stride, groups=C, padding=padding, output_padding=(output_padding_H, output_padding_W))

#         return x
    
#     @staticmethod
#     def backward(ctx, dx):
#         filters, = ctx.saved_tensors  # Unpack the saved tensors properly
#         B, C, H, W = ctx.shape
#         dx = dx.contiguous()

#         w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
#         stride = ctx.stride  
#         # Initialize all gradients
#         grad_x_ll = grad_x_lh = grad_x_hl = grad_x_hh = grad_filters = None
#         # print(ctx.needs_input_grad[0],ctx.needs_input_grad[1],ctx.needs_input_grad[2],ctx.needs_input_grad[3])
#         if ctx.needs_input_grad[0]:  # x_ll gradient
#             grad_x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)
        
#         if ctx.needs_input_grad[1]:  # x_lh gradient
#             grad_x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)
            
#         if ctx.needs_input_grad[2]:  # x_hl gradient
#             grad_x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)
            
#         if ctx.needs_input_grad[3]:  # x_hh gradient
#             grad_x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)

#         # Return gradients for all inputs in the same order as forward
#         return grad_x_ll, grad_x_lh, grad_x_hl, grad_x_hh, grad_filters
# class IDWT_2D(nn.Module):
#     def __init__(self, wave):
#         super(IDWT_2D, self).__init__()
#         w = pywt.Wavelet(wave)
#         rec_hi = torch.Tensor(w.rec_hi)
#         rec_lo = torch.Tensor(w.rec_lo)
        
#         w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
#         w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
#         w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
#         w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

#         w_ll = w_ll.unsqueeze(0).unsqueeze(1)
#         w_lh = w_lh.unsqueeze(0).unsqueeze(1)
#         w_hl = w_hl.unsqueeze(0).unsqueeze(1)
#         w_hh = w_hh.unsqueeze(0).unsqueeze(1)
#         filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
#         self.register_buffer('filters', filters)
#         self.filters = self.filters.to(dtype=torch.float16)

#     def forward(self, x_ll, x_lh, x_hl, x_hh):
#         return IDWT_Function.apply(x_ll, x_lh, x_hl, x_hh, self.filters)
# class DWT_2D(nn.Module):
#     def __init__(self, wave):
#         super(DWT_2D, self).__init__()
#         w = pywt.Wavelet(wave)
#         dec_hi = torch.Tensor(w.dec_hi[::-1]) 
#         dec_lo = torch.Tensor(w.dec_lo[::-1])

#         w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
#         w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
#         w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
#         w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

#         self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
#         self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

#         self.w_ll = self.w_ll.to(dtype=torch.float16)
#         self.w_lh = self.w_lh.to(dtype=torch.float16)
#         self.w_hl = self.w_hl.to(dtype=torch.float16)
#         self.w_hh = self.w_hh.to(dtype=torch.float16)

#     def forward(self, x):
#         return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
 
def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x



class TiTokDecoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.image_height = 128
        self.image_width = 256
        self.patch_size = 16
        self.grid_height = self.image_height // self.patch_size  # 8
        self.grid_width = self.image_width // self.patch_size   # 16
        self.model_size = "small"
        self.num_latent_tokens = 32
        self.token_size = 64
        self.out_channels = 64  # 输出特征的通道数
        self.width = self.image_width
        self.num_layers = 8
        self.num_heads = 8

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_height * self.grid_width + 1, self.width))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        # 重新设计解码头，直接输出所需的特征维度
        self.decoder_head = nn.Sequential(
            # 首先通过1x1卷积调整通道数
            nn.Conv2d(self.width, self.width // 2, 1, bias=True),
            nn.BatchNorm2d(self.width // 2),
            nn.ReLU(inplace=True),
            # 通过转置卷积上采样到目标分辨率
            nn.ConvTranspose2d(
                self.width // 2,
                self.width // 4,
                kernel_size=self.patch_size ,
                stride=self.patch_size ,
                bias=True
            ),
            nn.BatchNorm2d(self.width // 4),
            nn.ReLU(inplace=True),
            # 最后的1x1卷积调整到目标通道数
            nn.Conv2d(self.width // 4, self.out_channels, 1, bias=True),
        )
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_height * self.grid_width, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1 + self.grid_height * self.grid_width]  # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_height, self.grid_width)
        x = self.decoder_head(x)
        return x
    

class TiTokDecoder1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.image_height = 128
        self.image_width = 256
        self.patch_size = 8
        self.grid_height = self.image_height // self.patch_size  # 8
        self.grid_width = self.image_width // self.patch_size   # 16
        self.model_size = "small"
        self.num_latent_tokens = 32
        self.token_size = 64
        self.is_legacy = False
        self.width = self.image_width
        self.num_layers = 8
        self.num_heads = 8

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        # Adjust positional embedding size for rectangular grid
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_height * self.grid_width + 1, self.width))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels, adjusted for rectangular patches
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 64, 3, padding=1, bias=True)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        # Adjust mask tokens for rectangular grid
        mask_tokens = self.mask_token.repeat(batchsize, self.grid_height * self.grid_width, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1 + self.grid_height * self.grid_width]  # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W (now with rectangular grid)
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_height, self.grid_width)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x
 