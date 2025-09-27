import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck

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
        
'''
def test_time(x, dwt1, dwt2):
    loop = 1000
    total_time1 = 0
    total_time2 = 0

    for i in range(loop):
        start = time.time()
        y1 = dwt1(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time1 += end - start
    
    for i in range(loop):
        start = time.time()
        y2_ll, YH = dwt2(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time2 += end - start

    print(total_time1)
    print(total_time2)

def test_diff(x, dwt1, dwt2):
    y1 = dwt1(x)
    B, C, H, W = y1.shape
    y1 = y1.view(B, 4, -1, H, W)
    y1_ll = y1[:, 0] 
    y1_lh = y1[:, 1]
    y1_hl = y1[:, 2]
    y1_hh = y1[:, 3]
    y2_ll, YH = dwt2(x)
    y2_lh = YH[0][:,:,0]
    y2_hl = YH[0][:,:,1]
    y2_hh = YH[0][:,:,2]
    diff1 = (y1_ll - y2_ll).max()
    diff2 = (y1_lh - y2_lh).max()
    diff3 = (y1_hl - y2_hl).max()
    diff4 = (y1_hh - y2_hh).max()
    print(diff1)
    print(diff2)
    print(diff3)
    print(diff4)

def test_idfiff(x, idwt1, idwt2):
    y1 = idwt1(x)

    x = x.view(x.size(0), 4, -1, x.size(-2), x.size(-1))
    y2 = idwt2((x[:, 0], [x[:,1:].transpose(1, 2)]))
    diff = (y1-y2).max()
    print(diff)

def test_itime(x, idwt1, idwt2):
    loop = 1000
    total_time1 = 0
    total_time2 = 0

    for i in range(loop):
        start = time.time()
        y1 = idwt1(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time1 += end - start
    
    for i in range(loop):
        start = time.time()
        x = x.view(x.size(0), 4, -1, x.size(-2), x.size(-1))
        y2 = idwt2((x[:, 0], [x[:,1:].transpose(1, 2)]))
        torch.cuda.synchronize()
        end = time.time()
        total_time2 += end - start

    print(total_time1)
    print(total_time2)

if __name__ == '__main__':
    #size = (96, 32, 56, 56)
    #size = (96, 64, 28, 28)
    size = (96, 160, 14, 14)
    x = torch.randn(size).cuda().to(dtype=torch.float16)
    dwt1 = DWT_2D('haar').cuda()
    dwt2 = DWTForward(wave='haar').cuda()
    test_diff(x, dwt1, dwt2)
    test_time(x, dwt1, dwt2)

    #size = (96, 32*4, 28, 28)
    #size = (96, 64*4, 14, 14)
    #size = (96, 160*4, 7, 7)
    #x = torch.randn(size).cuda().to(dtype=torch.float16)
    #idwt1 = IDWT_2D('haar').cuda()
    #idwt2 = DWTInverse(wave='haar').cuda()
    #test_idfiff(x, idwt1, idwt2)
    #test_itime(x, idwt1, idwt2)
'''
def test_dwt_grad():
    # size = (4, 8, 14, 14)
    size = (4, 64, 128, 256)
    x = torch.randn(size).double()

    w = pywt.Wavelet('haar')
    dec_hi = torch.Tensor(w.dec_hi[::-1]) 
    dec_lo = torch.Tensor(w.dec_lo[::-1])

    w_ll = (dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_lh = (dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hl = (dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hh = (dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()

    input = (
        Variable(x, requires_grad=True),
        Variable(w_ll, requires_grad=False),
        Variable(w_lh, requires_grad=False),
        Variable(w_hl, requires_grad=False),
        Variable(w_hh, requires_grad=False),
    )
    test = gradcheck(DWT_Function.apply, input)
    print("test:", test)

def test_idwt_grad():
    # size = (4, 4*8, 7, 7)
    size = (4, 4*64, 32, 64)
    x = torch.randn(size).double()

    w = pywt.Wavelet('haar')
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
    filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).double()

    # components_dict = {
    #     'll': torch.randn((4,8,7,7)).double(),
    #     # 'lh': x_lh,
    #     # 'hl': x_hl,
    #     # 'hh': x_hh
    # }
    sub_size = (4, 64, 32, 64) #(4,8,4,4) #(4, 64, 32, 64)
    x_ll=torch.randn(sub_size).double()
    x_lh=torch.randn(sub_size).double()
    x_hl=torch.randn(sub_size).double()
    x_hh=torch.randn(sub_size).double()
    input = (
        Variable(x_ll, requires_grad=True),
        Variable(x_lh, requires_grad=True),
        Variable(x_hl, requires_grad=True),
        Variable(x_hh, requires_grad=True),
        Variable(filters, requires_grad=False),
    )
    test = gradcheck(IDWT_Function2.apply, input)
    print("test:", test)

if __name__ == "__main__":
    # test_dwt_grad().cuda()
    x = torch.randn(4,64,128,256).cuda().to(dtype=torch.float16)
    model_enc = DWT_2D(wave = 'haar').cuda()
    model_dec = IDWT_2D(wave = 'haar').cuda()
    x_ll, x_lh, x_hl, x_hh = model_enc(x)
    print("x_ll:",x_ll.shape)

    # test_idwt_grad()
    rec_y = model_dec(x_ll, None, x_hl, x_hh)
    print("rec_y:",rec_y.shape)
    mse = torch.mean((x - rec_y) ** 2)
    print(f"MSE: {mse.item():.4f}")

