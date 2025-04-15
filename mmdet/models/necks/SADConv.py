import torch
import torch.nn as nn
import mmcv.ops as ops

class SemanticAwareDeformConv2D(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1, dilation=2, max_offset=4.0, num_classes=3):
        super(SemanticAwareDeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.max_offset = max_offset
        self.padding = ((kernel_size - 1) * dilation) // 2
        self.num_classes = num_classes  # 语义类别数（如背景、边缘、前景）

        # 偏移计算层
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                      padding=self.padding, dilation=dilation, bias=True),
        )

        # 语义分支（按像素预测类别）
        self.semantic_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, num_classes, kernel_size=1),  # 每个像素都有 num_classes 维的概率
            nn.Softmax(dim=1)  # 归一化类别概率，保证 sum=1
        )

        # 使用 1x1 卷积代替 einsum 计算偏移修正量
        self.semantic_to_offset = nn.Conv2d(num_classes, 2 * kernel_size * kernel_size, kernel_size=1, bias=False)

        # 每个类别对应的最大偏移约束 (可学习参数)
        self.max_offset_scale = nn.Parameter(torch.full((num_classes, 1, 1, 1), max_offset))

        # 可变形卷积层
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=self.padding, dilation=dilation, bias=False)

    def constrain_offset(self, offset, semantic_map):
        """ 约束偏移量的最大范围 (维度修正版) """
        B, _, H, W = offset.shape
        offset_x = offset[:, 0::2, :, :]
        offset_y = offset[:, 1::2, :, :]

        # 计算偏移模长 (保持四维)
        offset_norm = torch.sqrt(torch.clamp(offset_x ** 2 + offset_y ** 2, min=1e-6))

        # 计算每个像素的 max_offset
        max_offset_per_pixel = torch.einsum('bchw,cxyz->bxyz', semantic_map, self.max_offset_scale)  # (B, 1, H, W)

        # 约束偏移
        mask = offset_norm > max_offset_per_pixel
        scale_factor = torch.ones_like(offset_norm)
        scale_factor[mask] = (max_offset_per_pixel / offset_norm)[mask]

        offset_x = offset_x * scale_factor
        offset_y = offset_y * scale_factor

        constrained_offset = torch.zeros_like(offset)
        constrained_offset[:, 0::2, :, :] = offset_x
        constrained_offset[:, 1::2, :, :] = offset_y
        return constrained_offset

    def forward(self, x):
        base_offset = self.offset_conv(x)  # 计算基本偏移，[B,2*K*K,H,W]
        semantic_map = self.semantic_branch(x)  # (B, num_classes, H, W) 的语义概率图

        # 通过 1x1 卷积计算偏移修正量
        offset_adjustment = self.semantic_to_offset(semantic_map)  # (B, 2*K*K, H, W)

        offset = base_offset + offset_adjustment  # 根据类别调整偏移
        offset = self.constrain_offset(offset, semantic_map)  # 约束最大偏移
        out = self.deform_conv(x, offset)  # 进行可变形卷积
        return out
