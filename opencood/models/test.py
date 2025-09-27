import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

import pywt
import numpy as np


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        stride = 4 
        ctx.stride = stride

        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = stride, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = stride, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = stride, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = stride, groups = dim)
        # x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        # return x

        ctx.sub_shape = x_ll.shape
        return x_ll, x_lh, x_hl, x_hh

    @staticmethod
    def backward(ctx, dx_ll, dx_lh, dx_hl, dx_hh):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            _, _, sub_h, sub_w = ctx.sub_shape
            stride = ctx.stride

            dx_cat = torch.cat([dx_ll, dx_lh, dx_hl, dx_hh], dim=1)
            dx = dx_cat.view(B, 4, -1, sub_h, sub_w)

            dx = dx.transpose(1,2).reshape(B, -1, sub_h, sub_w)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=stride, groups=C)

        return dx, None, None, None, None

class IDWT_Function2(Function):
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

        stride = 4
        ctx.stride = stride
        original_H, original_W =  H*stride, W*stride
        # 计算所需的 padding 和 output_padding
        kernel_size = filters.shape[-1]  # 滤波器的大小
        padding = (kernel_size - 1) // 2  # 通常的 padding 计算方式
        output_padding_H = original_H - ((H - 1) * stride - 2 * padding + kernel_size)
        output_padding_W = original_W - ((W - 1) * stride - 2 * padding + kernel_size)

        x = torch.nn.functional.conv_transpose2d(x, filters, stride=stride, groups=C, padding=padding, output_padding=(output_padding_H, output_padding_W))

        return x
    
    @staticmethod
    def backward(ctx, dx):
        filters, = ctx.saved_tensors  # Unpack the saved tensors properly
        B, C, H, W = ctx.shape
        dx = dx.contiguous()

        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        stride = ctx.stride  
        # Initialize all gradients
        grad_x_ll = grad_x_lh = grad_x_hl = grad_x_hh = grad_filters = None
        # print(ctx.needs_input_grad[0],ctx.needs_input_grad[1],ctx.needs_input_grad[2],ctx.needs_input_grad[3])
        if ctx.needs_input_grad[0]:  # x_ll gradient
            grad_x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)
        
        if ctx.needs_input_grad[1]:  # x_lh gradient
            grad_x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)
            
        if ctx.needs_input_grad[2]:  # x_hl gradient
            grad_x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)
            
        if ctx.needs_input_grad[3]:  # x_hh gradient
            grad_x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=stride, groups=C)

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
        return IDWT_Function2.apply(x_ll, x_lh, x_hl, x_hh, self.filters)

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
  
class LLCoeffCompression(nn.Module):
    def __init__(self, strategy='quantize', **kwargs):
        """
        压缩策略：
        - 'quantize': 量化压缩
        - 'prune': 稀疏化压缩
        - 'predictor': 预测编码压缩
        """
        super(LLCoeffCompression, self).__init__()
        self.strategy = strategy
        self.setup_params(**kwargs)
        
    def setup_params(self, **kwargs):
        if self.strategy == 'quantize':
            self.bits = kwargs.get('bits', 8)  # 量化位数
            self.scale = 2 ** (self.bits - 1) - 1
        elif self.strategy == 'prune':
            self.threshold = kwargs.get('threshold', 0.1)  # 裁剪阈值
        elif self.strategy == 'predictor':
            self.block_size = kwargs.get('block_size', 4)  # 预测块大小
            
    def quantize_compress(self, x_ll):
        """量化压缩方法"""
        # 找到数据范围
        x_min = torch.min(x_ll)
        x_max = torch.max(x_ll)
        
        # 归一化到[-1, 1]
        x_normalized = (x_ll - x_min) / (x_max - x_min) * 2 - 1
        
        # 量化
        x_quantized = torch.round(x_normalized * self.scale) / self.scale
        
        return {
            'data': x_quantized,
            'min': x_min,
            'max': x_max,
            'shape': x_ll.shape
        }
    
    def quantize_decompress(self, compressed):
        """量化解压方法"""
        x_min = compressed['min']
        x_max = compressed['max']
        x_quantized = compressed['data']
        
        # 反归一化
        x_reconstructed = (x_quantized + 1) / 2 * (x_max - x_min) + x_min
        return x_reconstructed
    
    def prune_compress(self, x_ll):
        """稀疏化压缩方法"""
        # 计算自适应阈值
        abs_mean = torch.mean(torch.abs(x_ll))
        threshold = self.threshold * abs_mean
        
        # 保留大系数
        mask = torch.abs(x_ll) > threshold
        values = x_ll[mask]
        indices = torch.nonzero(mask)
        
        return {
            'values': values,
            'indices': indices,
            'shape': x_ll.shape,
            'threshold': threshold
        }
    
    def prune_decompress(self, compressed):
        """稀疏化解压方法"""
        shape = compressed['shape']
        values = compressed['values']
        indices = compressed['indices']
        
        x_reconstructed = torch.zeros(shape, device=values.device, dtype=values.dtype)
        x_reconstructed[tuple(indices.T)] = values
        return x_reconstructed
    
    def predictor_compress(self, x_ll):
        """预测编码压缩方法"""
        B, C, H, W = x_ll.shape
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        
        # 补齐到block_size的整数倍
        if pad_h > 0 or pad_w > 0:
            x_ll = F.pad(x_ll, (0, pad_w, 0, pad_h), mode='reflect')
            
        # 分块处理
        blocks = x_ll.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        
        # 使用块的平均值作为预测值
        block_means = torch.mean(blocks, dim=(4,5), keepdim=True)
        residuals = blocks - block_means
        
        return {
            'means': block_means,
            'residuals': residuals,
            'shape': x_ll.shape,
            'pad': (pad_h, pad_w)
        }
    
    def predictor_decompress(self, compressed):
        """预测编码解压方法"""
        means = compressed['means']
        residuals = compressed['residuals']
        original_shape = compressed['shape']
        pad_h, pad_w = compressed['pad']
        
        # 重构块
        blocks = means + residuals
        
        # 重构原始tensor
        B, C, H, W = original_shape
        x_reconstructed = blocks.permute(0,1,2,3,4,5).contiguous()
        x_reconstructed = x_reconstructed.view(B, C, -1, W)
        
        # 移除padding
        if pad_h > 0:
            x_reconstructed = x_reconstructed[:, :, :-pad_h, :]
        if pad_w > 0:
            x_reconstructed = x_reconstructed[:, :, :, :-pad_w]
            
        return x_reconstructed
    
    def forward(self, x_ll, compress=True):
        if compress:
            if self.strategy == 'quantize':
                return self.quantize_compress(x_ll)
            elif self.strategy == 'prune':
                return self.prune_compress(x_ll)
            elif self.strategy == 'predictor':
                return self.predictor_compress(x_ll)
        else:
            if self.strategy == 'quantize':
                return self.quantize_decompress(x_ll)
            elif self.strategy == 'prune':
                return self.prune_decompress(x_ll)
            elif self.strategy == 'predictor':
                return self.predictor_decompress(x_ll)

class DataSizeCalculator:
    @staticmethod
    def get_tensor_size_bytes(tensor):
        """计算单个tensor的字节大小"""
        return tensor.element_size() * tensor.nelement()
    
    @staticmethod
    def get_compressed_size_bytes(compressed_data, strategy):
        """计算压缩后数据的字节大小"""
        total_bytes = 0
        
        if strategy == 'quantize':
            # 量化数据的大小
            total_bytes += DataSizeCalculator.get_tensor_size_bytes(compressed_data['data'])
            # 添加min和max值的大小(每个都是一个浮点数)
            total_bytes += 8  # float32 min和max各占4字节
            # shape信息(4个整数)
            total_bytes += 16  # 4个int32
            
        elif strategy == 'prune':
            # 非零值的大小
            total_bytes += DataSizeCalculator.get_tensor_size_bytes(compressed_data['values'])
            # 索引的大小
            total_bytes += DataSizeCalculator.get_tensor_size_bytes(compressed_data['indices'])
            # shape信息
            total_bytes += 16  # 4个int32
            # 阈值
            total_bytes += 4  # float32
            
        elif strategy == 'predictor':
            # 平均值的大小
            total_bytes += DataSizeCalculator.get_tensor_size_bytes(compressed_data['means'])
            # 残差的大小
            total_bytes += DataSizeCalculator.get_tensor_size_bytes(compressed_data['residuals'])
            # shape和padding信息
            total_bytes += 24  # 4个int32用于shape，2个int32用于padding
            
        return total_bytes


def analyze_tensor_size(x_ll):
    # 基本信息
    element_size = x_ll.element_size()  # 每个元素的字节数
    num_elements = x_ll.nelement()      # 元素总数
    total_bytes = element_size * num_elements
    
    print(f"Shape: {x_ll.shape}")
    print(f"Data type: {x_ll.dtype}")
    print(f"Element size: {element_size} bytes")
    print(f"Number of elements: {num_elements}")
    print(f"Total size: {total_bytes / (1024*1024):.2f} MB")
    return total_bytes

class DetailedSizeCalculator:
    @staticmethod
    def analyze_compressed_data(compressed_data, strategy):
        """详细分析压缩数据的各个组成部分的大小"""
        size_details = {}
        
        if strategy == 'quantize':
            # 量化数据
            data_size = compressed_data['data'].element_size() * compressed_data['data'].nelement()
            size_details['quantized_data'] = data_size
            # 元数据（min, max, shape）
            size_details['metadata'] = 8 + 16  # min/max (8 bytes) + shape (16 bytes)
            
        elif strategy == 'prune':
            # 非零值
            values_size = compressed_data['values'].element_size() * compressed_data['values'].nelement()
            size_details['values'] = values_size
            # 索引
            indices_size = compressed_data['indices'].element_size() * compressed_data['indices'].nelement()
            size_details['indices'] = indices_size
            # 元数据
            size_details['metadata'] = 20  # shape (16 bytes) + threshold (4 bytes)
            
        elif strategy == 'predictor':
            # 平均值
            means_size = compressed_data['means'].element_size() * compressed_data['means'].nelement()
            size_details['means'] = means_size
            # 残差
            residuals_size = compressed_data['residuals'].element_size() * compressed_data['residuals'].nelement()
            size_details['residuals'] = residuals_size
            # 元数据
            size_details['metadata'] = 24  # shape (16 bytes) + padding (8 bytes)
            
        size_details['total'] = sum(size_details.values())
        return size_details
    
def test_compression():
    # 创建测试数据
    x = torch.randn(4, 64, 128, 256).cuda().to(dtype=torch.float16)
    dwt = DWT_2D(wave="haar").cuda()
    idwt = IDWT_2D(wave="haar").cuda()
    
    # 获取LL系数
    x_ll, _, _, _ = dwt(x)
    print(analyze_tensor_size(x_ll))
    # 测试不同压缩方法
    methods = {
        'quantize': {'bits': 8},
        'prune': {'threshold': 0.1},
        'predictor': {'block_size': 4}
    }
    
    results = {}
    for method, params in methods.items():
        # 创建压缩器
        compressor = LLCoeffCompression(strategy=method, **params).cuda()
        print("x_ll:",x_ll.numel())
        print("x_ll size:",DataSizeCalculator.get_tensor_size_bytes(x_ll))
        # 压缩
        compressed = compressor(x_ll, compress=True)
        print("compressed:",compressed['shape'])
        # 计算压缩后的大小

        size_details = DetailedSizeCalculator.analyze_compressed_data(compressed, method)
        # 打印详细信息
        print("\nSize breakdown:")
        for key, size in size_details.items():
            if key != 'total':
                print(f"{key}: {size / (1024*1024):.4f} MB")
        print(f"Total compressed size: {size_details['total'] / (1024*1024):.4f} MB")


        # 解压缩
        reconstructed_ll = compressor(compressed, compress=False)
        print("\nreconstructed_ll:",reconstructed_ll.shape)
        # 重构
        reconstructed = idwt(reconstructed_ll, None, None, None)
        
        # 计算指标
        if method == 'prune':
            compression_ratio = x_ll.numel() / compressed['values'].numel()
        elif method == 'quantize':
            compression_ratio = 32 / compressor.bits  # 假设原始数据是32位
        else:
            compression_ratio = x_ll.numel() / (compressed['means'].numel() + compressed['residuals'].numel())
        
        mse = torch.mean((x - reconstructed) ** 2)
        
        results[method] = {
            'compression_ratio': compression_ratio,
            'mse': mse.item()
        }
        print(results)
    return results

test_compression()