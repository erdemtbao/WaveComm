""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic  # 导入 icecream 库，用于调试和打印变量（类似增强版 print）。
from collections import OrderedDict, Counter  # 从 collections 导入 OrderedDict（有序字典）和 Counter（计数器），用于管理模态和统计。
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone  # 鸟瞰图（BEV）主干网络，基于 ResNet。
from opencood.models.sub_modules.feature_alignnet import AlignNet # 特征对齐网络
from opencood.models.sub_modules.downsample_conv import DownsampleConv # 降采样卷积模块
from opencood.models.sub_modules.naive_compress import NaiveCompressor # 简单压缩模块
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion    # 多尺度特征融合模块
from opencood.utils.transformation_utils import normalize_pairwise_tfm # 归一化智能体间的空间变换矩阵
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn # 检查可训练模块、冻结/解冻批归一化层
import importlib # 用于动态加载模块
import torchvision


class HeterPyramidCollab(nn.Module):
    def __init__(self, args):  # 构造函数，接收参数 args（通常是一个字典，包含模型配置）
        super(HeterPyramidCollab, self).__init__()  # 调用父类 nn.Module 的构造函数，初始化模型
        self.args = args # 保存传入的配置参数
        modality_name_list = list(args.keys()) # 获取 args 字典的所有键，包含模态配置
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] # 过滤出以 "m" 开头且后面跟数字的键（如 m1, m2），这些键表示不同模态（如激光雷达、相机）
        self.modality_name_list = modality_name_list # 保存过滤后的模态名称列表

        self.cav_range = args['lidar_range'] # 从 args 获取激光雷达的感知范围（如 [x_min, y_min, z_min, x_max, y_max, z_max]），用于空间对齐
        self.sensor_type_dict = OrderedDict() # 初始化有序字典，存储模态名称到传感器类型的映射（如 m1: lidar）

        self.cam_crop_info = {}  # 初始化字典，存储相机模态的裁剪信息

        # setup each modality model  设置每个模态模型
        for modality_name in self.modality_name_list: # 遍历所有模态（如 m1, m2）
            model_setting = args[modality_name] # 获取当前模态的配置（如编码器、主干网络参数）
            sensor_name = model_setting['sensor_type'] # 获取传感器类型（如 lidar 或 camera）
            self.sensor_type_dict[modality_name] = sensor_name # 将模态名称和传感器类型存入字典

            # import model
            encoder_filename = "opencood.models.heter_encoders"  # 指定编码器模块的路径
            encoder_lib = importlib.import_module(encoder_filename) # 动态导入编码器模块
            encoder_class = None   # 初始化编码器类为 None
            target_model_name = model_setting['core_method'].replace('_', '')  # 获取编码器的核心方法名（如 voxel_net 转为 voxelnet）

            for name, cls in encoder_lib.__dict__.items():  # 遍历编码器模块中的所有属性，查找与 target_model_name 匹配的类
                if name.lower() == target_model_name.lower():  # 忽略大小写，匹配类名
                    encoder_class = cls  # 找到匹配的编码器类

            """
            Encoder building  构建Encoder
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))  # 为当前模态创建编码器实例（如 self.encoder_m1），并传入编码器参数
            if model_setting['encoder_args'].get("depth_supervision", False):  # 检查编码器参数是否支持深度监督（depth supervision）
                setattr(self, f"depth_supervision_{modality_name}", True)      # 记录当前模态是否启用深度监督
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building  构建Backbone
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            # 为当前模态创建 ResNetBEV 主干网络实例（如 self.backbone_m1），传入主干网络参数。
            """
            Aligner building  构建Aligner对齐
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))  # 为当前模态创建特征对齐网络实例（如 self.aligner_m1），传入对齐参数
            if sensor_name == "camera":  # 如果传感器是相机，执行裁剪相关配置
                camera_mask_args = model_setting['camera_mask_args']  # 获取相机掩码参数
                setattr(self, f"crop_ratio_W_{modality_name}",
                        (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))  # 计算宽度裁剪比例（激光雷达 x 范围除以相机 x 范围）
                setattr(self, f"crop_ratio_H_{modality_name}",
                        (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))  # 计算高度裁剪比例（激光雷达 y 范围除以相机 y 范围）
                setattr(self, f"xdist_{modality_name}",
                        (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))  # 计算相机 x 轴范围（最大值 - 最小值）
                setattr(self, f"ydist_{modality_name}",
                        (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))  # 计算相机 y 轴范围
                self.cam_crop_info[modality_name] = {   #将裁剪比例保存到 cam_crop_info 字典
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """For feature transformation 用于特征变换"""
        self.H = (self.cav_range[4] - self.cav_range[1])  # 计算感知范围的高度（y_max - y_min）
        self.W = (self.cav_range[3] - self.cav_range[0])  # 计算感知范围的宽度（x_max - x_min）
        self.fake_voxel_size = 1  # 设置虚拟体素大小为 1，用于空间变换

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        融合，默认为多尺度融合：
        注意：PyramidFusion 的输入已下采样 2 倍。（需要 SECOND）
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])  # 创建多尺度融合模块 PyramidFusion，传入融合参数。输入特征已降采样 2 倍

        """
        Shrink header 收缩标题
        """
        self.shrink_flag = False    # 初始化缩减标志为 False
        if 'shrink_header' in args: # 如果 args 包含缩减头参数
            self.shrink_flag = True # 启用缩减
            self.shrink_conv = DownsampleConv(args['shrink_header'])  # 创建降采样卷积模块

        """
        Shared Heads 共享Heads头
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)  # 创建分类头，1x1 卷积，输入通道为 in_head，输出通道为锚框数量（anchor_number）
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)  # 创建回归头，输出 7 个回归参数（位置、尺寸等）乘以锚框数量
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1)  # BIN_NUM = 2 # 创建方向头，输出方向分类的 bin 数乘以锚框数量

        # compressor will be only trainable 压缩机只能训练
        self.compress = False    # 初始化压缩标志为 False
        if 'compressor' in args: # 如果 args 包含压缩参数
            self.compress = True # 启用压缩
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])  # 创建压缩模块，传入输入维度和压缩比率

        self.model_train_init()  # 调用初始化训练设置函数
        # check again which module is not fixed.
        check_trainable_module(self)  # 检查模型中哪些模块是可训练的

    def model_train_init(self):  # 定义训练初始化函数
        # if compress, only make compressor trainable
        if self.compress:  # 如果启用压缩
            # freeze all
            self.eval()    # 将模型设置为评估模式（冻结批归一化等）
            for p in self.parameters(): # 冻结所有参数
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()    # 将压缩器设置为训练模式
            for p in self.compressor.parameters(): # 解冻压缩器的参数
                p.requires_grad_(True)

    def forward(self, data_dict):  # 定义前向传播函数，接收输入数据字典
        output_dict = {'pyramid': 'collab'}  # 初始化输出字典，记录融合类型为协作
        agent_modality_list = data_dict['agent_modality_list']  # 获取智能体模态列表（如 [m1, m2, m1]）
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)  # 归一化智能体间的空间变换矩阵，基于感知范围和体素大小
        record_len = data_dict['record_len'] # 获取记录长度（可能表示批次中智能体数量）
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)  # 统计每个模态出现的次数（如 {m1: 2, m2: 1}）
        modality_feature_dict = {}  # 初始化字典，存储每个模态的特征

        for modality_name in self.modality_name_list:  # 循环 for modality_name in self.modality_name_list:：遍历所有模态
            if modality_name not in modality_count_dict:
                continue   # 如果当前模态未出现在输入数据中，跳过
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)  # 调用模态的编码器提取特征
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']  # 通过主干网络处理特征，获取 2D 空间特征
            feature = eval(f"self.aligner_{modality_name}")(feature)  # 通过对齐网络对特征进行空间对齐
            modality_feature_dict[modality_name] = feature  # 保存特征到字典

        """
        Crop/Padd camera feature map.  裁剪/填充相机特征图。
        """
        for modality_name in self.modality_name_list:  # 循环 for modality_name in self.modality_name_list:：遍历模态，处理相机特征
            if modality_name in modality_count_dict:   # 确保模态存在于输入中
                if self.sensor_type_dict[modality_name] == "camera":  # 如果模态是相机
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]    # 获取特征
                    _, _, H, W = feature.shape   # 获取特征图的高和宽
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))  # 计算裁剪后的目标高度
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))  # 计算裁剪后的目标宽度

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))  # 创建中心裁剪函数
                    modality_feature_dict[modality_name] = crop_func(feature)       # 裁剪特征图
                    if eval(f"self.depth_supervision_{modality_name}"):  # 如果支持深度监督
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })  # 将编码器的深度项添加到输出字典

        """
        Assemble heter features  汇集heter features
        """
        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}   # 初始化计数字典，跟踪每个模态的特征索引
        heter_feature_2d_list = []   # 初始化 2D 特征列表
        for modality_name in agent_modality_list:  # 循环 for modality_name in agent_modality_list:：按智能体模态顺序组装特征
            feat_idx = counting_dict[modality_name]  # 获取当前模态的特征索引
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])  # 添加对应特征
            counting_dict[modality_name] += 1  # 索引加 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)  # 将特征列表堆叠成张量

        if self.compress:  # 如果启用压缩
            heter_feature_2d = self.compressor(heter_feature_2d)  # 通过压缩器处理特征

        # heter_feature_2d is downsampled 2x   heter_feature_2d 已下采样 2 倍
        # add croping information to collaboration module    将裁剪信息添加到协作模块

        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(  # 调用融合模块的协作前向函数，输入
            heter_feature_2d,     # 异构特征
            record_len,           # 记录长度
            affine_matrix,        # 空间变换矩阵
            agent_modality_list,  # 模态列表
            self.cam_crop_info    # 相机裁剪信息
        )  # 返回融合特征（fused_feature）和占用预测（occ_outputs）

        if self.shrink_flag:  # 如果启用缩减
            fused_feature = self.shrink_conv(fused_feature)  # 通过降采样卷积处理融合特征

        cls_preds = self.cls_head(fused_feature)  # 通过分类头生成分类预测
        reg_preds = self.reg_head(fused_feature)  # 通过回归头生成回归预测
        dir_preds = self.dir_head(fused_feature)  # 通过方向头生成方向预测

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds}) # 将分类、回归、方向预测和占用预测添加到输出字典

        output_dict.update({'occ_single_list':
                                occ_outputs})

        return output_dict  # 返回包含所有预测的字典