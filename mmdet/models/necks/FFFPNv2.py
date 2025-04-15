import torch
import torch.nn as nn
import torch.nn.functional as F


class FreeFlowFPN(nn.Module):
    def __init__(self, in_channels, out_channels, tau=0.1):
        """
        标准 Feature Pyramid Network (FPN)
        :param in_channels: 输入的特征层通道数
        :param out_channels: 输出特征层通道数
        """
        super().__init__()

        # 细化特征层 (3x3 卷积)，金字塔只有三层
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) for _ in range(3)
        ])

    def forward(self, features):
        """
        :param features: List of feature maps from backbone (从高到低)
        :return: FPN 输出特征层列表
        """
        num_layers = len(features)

        # 确保输入只有3层特征
        assert num_layers == 3, "输入的特征层必须是三层"

        # 2️⃣ 自顶向下融合特征
        for i in range(num_layers - 1, 0, -1):  # 从高层到低层
            upsampled = F.interpolate(features[i], size=features[i - 1].shape[2:], mode='bilinear',
                                      align_corners=False)
            features[i - 1] += upsampled  # 融合高层信息

        # 3️⃣ 细化特征 (3x3 卷积)
        fpn_outputs = [smooth_conv(f) for smooth_conv, f in zip(self.smooth_convs, features)]

        return fpn_outputs
