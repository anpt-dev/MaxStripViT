# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:10:44 2025

@author: An Pham
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

#-----------------------------------------------
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # 1. Convolution chính (Weight)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        
        # 2. Convolution để học Offset và Mask
        # Output channel = 3 * kernel_size * kernel_size
        # (2 cho offset x,y + 1 cho mask modulation)
        self.offset_mask_conv = nn.Conv2d(in_channels, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=padding, 
                                          bias=True)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
        # Quan trọng: Init offset & mask về 0 để ban đầu nó hoạt động như Conv thường
        # Giúp training ổn định hơn ở các epoch đầu.
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, x):
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        # Gọi hàm từ torchvision.ops
        return ops.deform_conv2d(input=x, 
                                 offset=offset, 
                                 weight=self.weight, 
                                 bias=self.bias, 
                                 stride=self.stride, 
                                 padding=self.padding, 
                                 mask=mask)
#----------------------------------------------------
class SEBlock(nn.Module):
    """ Squeeze-and-Excitation: Nhẹ hơn CA, tập trung vào Channel """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
#------------------------------------------------------------
class StemDCN(nn.Module):
    def __init__(self, in_chans=3, out_chans=64):
        super().__init__()
        mid_chans = out_chans // 2
        
        # Layer 1: Standard Conv (Khởi động nhẹ nhàng)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_chans),
            nn.GELU()
        )
        
        # Layer 2: Deformable Conv v2 (Bước quan trọng)
        # Thay thế Conv thường bằng DCNv2
        self.dcn = DeformableConv2d(mid_chans, out_chans, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.act2 = nn.GELU()
        
        # Layer 3: Coordinate Attention (Thay vì CBAM)
        # Đã thống nhất ở bước trước là CA tốt hơn cho AreaNet
        self.se = SEBlock(out_chans, reduction=4) # reduction nhỏ ở stem để giữ thông tin

    def forward(self, x):
        x = self.conv1(x)
        
        # DCN Forward
        x = self.dcn(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Apply Attention
        x = self.se(x)
        return x
# =============================================================================
# 1. CÁC MODULE CƠ BẢN (LAYERS)
# =============================================================================

class DropPath(nn.Module):
    """Stochastic Depth / Drop Path regularization."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.Module):
    """LayerNorm hỗ trợ format [B, C, H, W] (như ConvNeXt)."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# --- CBAM (Channel & Spatial Attention) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# =============================================================================
# 2. KHỐI TÍNH TOÁN (MBConv, ConvFFN)
# =============================================================================

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # Reduction ratio thường là 32 để tiết kiệm tham số
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Pool W về 1 -> Giữ lại H: [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Pool H về 1 -> Giữ lại W: [B, C, 1, W]

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() # CA paper dùng HardSwish, rất nhẹ và hiệu quả
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # 1. Pooling theo 2 hướng
        x_h = self.pool_h(x) # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [B, C, 1, W] -> [B, C, W, 1]

        # 2. Ghép lại để xử lý chung (tận dụng mối tương quan giữa H và W)
        y = torch.cat([x_h, x_w], dim=2) # [B, C, H+W, 1]
        
        # 3. Conv 1x1 giảm chiều & Activation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # 4. Tách ra lại
        x_h, x_w = torch.split(y, [h, w], dim=2)
        
        # 5. Conv 1x1 khôi phục chiều channel & Sigmoid
        a_h = self.conv_h(x_h).sigmoid() # Attention map cho chiều H
        a_w = self.conv_w(x_w).permute(0, 1, 3, 2).sigmoid() # Attention map cho chiều W

        # 6. Apply Attention
        out = identity * a_h * a_w
        return out

class MBConvCA(nn.Module):
    """
    Mobile Inverted Residual Block sử dụng Coordinate Attention
    thay vì CBAM.
    """
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.conv = nn.Sequential(
            # Expansion (1x1 conv)
            nn.Conv2d(dim, hidden_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            # Depthwise Conv (3x3)
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            # --- Thay CBAM bằng Coordinate Attention ---
            CoordAtt(hidden_dim, reduction=32), 
            # -------------------------------------------
            
            # Projection (1x1 conv)
            nn.Conv2d(hidden_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv(x)


#-------------------
class MBConvCBAM(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            CBAM(hidden_dim), # Tích hợp CBAM
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x):
        return x + self.conv(x)

class ConvFFN(nn.Module):
    """Feed Forward Network sử dụng Conv để tăng tính cục bộ."""
    def __init__(self, dim, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or dim * 4
        self.fc1 = nn.Conv2d(dim, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# =============================================================================
# 3. AREA ATTENTION (DYNAMIC RECTANGULAR SPLIT)
# =============================================================================

class AreaAttention(nn.Module):
    def __init__(self, dim, num_segments=1, axis='h', num_heads=8, attn_drop=0., proj_drop=0.):
        """
        num_segments (n): Số lượng vùng cần chia.
        axis: 'h' (chia chiều cao) hoặc 'w' (chia chiều rộng).
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_segments = num_segments
        self.axis = axis

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        n = self.num_segments
        
        # --- Logic Chia Vùng & Reshape ---
        if self.axis == 'h':
            # Chia H thành n phần -> Kích thước vùng: (H/n) x W
            if H % n != 0:
                # Padding nếu không chia hết (Optional implementation)
                pad_h = (n - H % n) % n
                x = F.pad(x, (0, 0, 0, pad_h))
                H = H + pad_h
            
            h_seg = H // n
            
            # [B, C, H, W] -> [B, C, n, h_seg, W]
            # -> permute -> [B, n, h_seg, W, C]
            # -> flatten spatial -> [B*n, (h_seg * W), C]
            x_reshaped = x.view(B, C, n, h_seg, W).permute(0, 2, 3, 4, 1).contiguous()
            x_in = x_reshaped.view(B * n, h_seg * W, C)
            
        else: # axis == 'w'
            # Chia W thành n phần -> Kích thước vùng: H x (W/n)
            if W % n != 0:
                pad_w = (n - W % n) % n
                x = F.pad(x, (0, pad_w, 0, 0))
                W = W + pad_w
            
            w_seg = W // n
            
            # [B, C, H, W] -> [B, C, H, n, w_seg]
            # -> permute -> [B, n, H, w_seg, C]
            # -> flatten spatial -> [B*n, (H * w_seg), C]
            x_reshaped = x.view(B, C, H, n, w_seg).permute(0, 3, 2, 4, 1).contiguous()
            x_in = x_reshaped.view(B * n, H * w_seg, C)

        # --- Tính Attention ---
        x_norm = self.norm(x_in)
        N_tokens = x_norm.shape[1] # Số tokens trong 1 vùng chữ nhật
        
        qkv = self.qkv(x_norm).reshape(x_norm.shape[0], N_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Ưu tiên Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).flatten(2)
        x_out = self.proj(x_attn)
        x_out = self.proj_drop(x_out)
        
        # Residual
        x_out = x_out + x_in

        # --- Khôi phục Shape cũ ---
        if self.axis == 'h':
            # [B*n, (h*w), C] -> [B, n, h, w, C]
            x_out = x_out.view(B, n, h_seg, W, C)
            # -> [B, C, n, h, w] -> [B, C, H, W]
            x_out = x_out.permute(0, 4, 1, 2, 3).contiguous().view(B, C, H, W)
            
            # Remove padding nếu có
            if x_out.shape[2] != x.shape[2]: # H check
                 x_out = x_out[:, :, :x.shape[2], :]
                 
        else: # axis == 'w'
            x_out = x_out.view(B, n, H, w_seg, C)
            x_out = x_out.permute(0, 4, 2, 1, 3).contiguous().view(B, C, H, W)
            
            if x_out.shape[3] != x.shape[3]: # W check
                 x_out = x_out[:, :, :, :x.shape[3]]

        return x_out

# =============================================================================
# 4. AREA ATTENTION BLOCK & STAGE
# =============================================================================

class AreaAttentionBlock(nn.Module):
    def __init__(self, dim, num_segments, num_heads=4, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        
        # 1. Local Branch (MBConv)
        self.mbconv = MBConvCA(dim)
        
        # 2. H-Area Branch (Attention trên các vùng dọc)
        self.h_attn = AreaAttention(dim, num_segments=num_segments, axis='h', 
                                    num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.h_ffn = ConvFFN(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        
        # 3. W-Area Branch (Attention trên các vùng ngang)
        self.w_attn = AreaAttention(dim, num_segments=num_segments, axis='w', 
                                    num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.w_ffn = ConvFFN(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # MBConv
        x = self.mbconv(x)
        
        # H-Attention
        x = self.h_attn(x)
        x = x + self.drop_path(self.h_ffn(x))
        
        # W-Attention
        x = self.w_attn(x)
        x = x + self.drop_path(self.w_ffn(x))
        
        return x
#-------------------
class DownsampleLayer(nn.Module):
    """Giảm kích thước không gian đi 2 lần, tăng channel."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim)
        )
    def forward(self, x):
        return self.down(x)

# =============================================================================
# 5. FULL MODEL: AREANET
# =============================================================================

class AreaNet(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3],       # Số lượng block mỗi stage
                 dims=[64, 128, 256, 512],  # Số channel mỗi stage
                 num_heads=[2, 4, 8, 16],   # Số head attention
                 segment_splits=[4, 2, 2, 1], # Quan Trọng: Giá trị n cho từng stage
                 drop_path_rate=0.1):
        super().__init__()
        
        # S0: Stem (Giảm kích thước 4 lần: 224 -> 56)
        
        self.stem = StemDCN()
        # self.stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0]//2, 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(dims[0]//2),
        #     nn.GELU(),
        #     nn.Conv2d(dims[0]//2, dims[0], 3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(dims[0]),
        #     nn.GELU()
        # )
        
        # Tạo chuỗi stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        self.stages = nn.ModuleList()
        curr_idx = 0
        
        for i in range(4):
            layers = []
            
            # Downsampling (trừ stage đầu tiên sau stem)
            if i > 0:
                layers.append(DownsampleLayer(dims[i-1], dims[i]))
                
            # Tạo các Block cho stage i
            # Truyền tham số n (segment_splits[i]) vào block
            for j in range(depths[i]):
                layers.append(
                    AreaAttentionBlock(
                        dim=dims[i], 
                        num_segments=segment_splits[i], # n thay đổi theo stage
                        num_heads=num_heads[i], 
                        drop_path=dpr[curr_idx]
                    )
                )
                curr_idx += 1
            
            self.stages.append(nn.Sequential(*layers))
            
        # Classification Head
        self.norm = nn.BatchNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
            
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

# =============================================================================
# 6. CONFIGURATIONS
# =============================================================================

def area_net_tiny(num_classes=1000):
    # Cấu hình Tiny
    # segment_splits=[4, 2, 2, 1]:
    # Stage 1 (56x56): Chia 4 vùng -> Mỗi vùng 14x56
    # Stage 2 (28x28): Chia 2 vùng -> Mỗi vùng 14x28
    # Stage 3 (14x14): Chia 2 vùng -> Mỗi vùng 7x14
    # Stage 4 (7x7):   Chia 1 vùng -> Global Attention 7x7
    return AreaNet(depths=[2, 2, 6, 2], 
                   dims=[64, 128, 256, 512], 
                   num_heads=[4, 8, 16, 32], 
                   segment_splits=[8, 4, 2, 1], 
                   drop_path_rate=0.1, 
                   num_classes=num_classes)

def area_net_small(num_classes=1000):
    return AreaNet(depths=[3, 3, 27, 3], 
                   dims=[96, 192, 384, 768], 
                   num_heads=[3, 6, 12, 24], 
                   segment_splits=[8, 4, 2, 1], 
                   drop_path_rate=0.2, 
                   num_classes=num_classes)

# =============================================================================
# TEST RUN
# =============================================================================
# if __name__ == "__main__":
    # model = area_net_tiny()
    # # model = model.cuda() 
    
    # # Input chuẩn ImageNet
    # dummy = torch.randn(1, 3, 224, 224)
    # out = model(dummy)
    
    # print(f"AreaNet Tiny created.")
    # print(f"Input: {dummy.shape}")
    # print(f"Output: {out.shape}")
    
    # # Check parameters
    # params = sum(p.numel() for p in model.parameters())
    # print(f"Total Params: {params/1e6:.2f}M")