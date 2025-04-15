from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType, OptMultiConfig
from mmdet.models.necks.SADConv import SemanticAwareDeformConv2D
from mmdet.models.necks.FreeFlowFPN import FreeFlowFPN
# from mmdet.models.necks.FFFPNv2 import FreeFlowFPN


class ModifiedChannelMapper(BaseModule):
    def __init__(self, in_channels: List[int], out_channels: int = 256, kernel_size: int = 3,
                 conv_cfg: OptConfigType = None, norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = dict(type='ReLU'), bias: Union[bool, str] = 'auto',
                 num_outs: int = 4,
                 init_cfg: OptMultiConfig = dict(type='Xavier', layer='Conv2d', distribution='uniform')) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=bias
                )
            )

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        assert len(inputs) == len(self.in_channels)

        # 处理输入特征图
        processed_inputs = [self.convs[i](inputs[i]) for i in range(len(inputs))]

        # return tuple(processed_inputs)
        return processed_inputs


# 复杂特征融合模块：卷积 + MHSA + SKNet
class ComplexFeatureFusionModule(BaseModule):
    # reduction=16
    def __init__(self, in_channels, norm_cfg=None, num_heads=8, num_layers=2, reduction=4):
        super(ComplexFeatureFusionModule, self).__init__()
        # 五个分支卷积
        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.branch2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)
        # self.branch3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, groups=in_channels)
        # self.branch4_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # self.branch4_2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        # self.branch5_1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        # self.branch5_2 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=9, dilation=3, groups=in_channels)

        # self.branch6_1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        # self.branch6_2 = nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=16, dilation=4, groups=in_channels)

        self.down4=nn.Conv2d(in_channels,in_channels//4,kernel_size=1)

        # Transformer编码器层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels//4, nhead=num_heads, dim_feedforward=in_channels//4 * 4),
            num_layers=num_layers
        )

        self.up4=nn.Conv2d(in_channels//4,in_channels,kernel_size=1)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # # 全连接层，用于生成通道注意力
        # # in_channels // reduction
        # self.fc1 = nn.Linear(in_channels, in_channels * reduction, bias=False)
        # # 6
        # self.fc2 = nn.Linear(in_channels * reduction, in_channels * 2, bias=False)
        #
        # # softmax层
        # self.softmax = nn.Softmax(dim=1)

        # 归一化层
        self.norm = nn.BatchNorm2d(in_channels) if norm_cfg is None else build_norm_layer(norm_cfg, in_channels)[1]

        # 调用初始化权重的方法
        self.init_weights()

        # ————————————LSK部分(空间注意力)————————————
        # self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)

        self.conv_squeeze_s = nn.Conv2d(1, 1, 3, padding=1)  # 对avg_attn使用3x3卷积
        self.conv_squeeze_l = nn.Conv2d(1, 1, 7, padding=3)  # 对max_attn使用7x7卷积

        self.norm_s = nn.BatchNorm2d(in_channels) if norm_cfg is None else build_norm_layer(norm_cfg, in_channels)[1]

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        # 五个分支卷积
        x1 = self.branch1(x)
        # x2 = self.branch2(x)
        # x3 = self.branch3(x)
        # x4 = self.branch4_1(x)
        # x4 = self.branch4_2(x4)
        # x5 = self.branch5_1(x)
        # x5 = self.branch5_2(x5)

        # x6 = self.branch6_1(x)
        # x6 = self.branch6_2(x6)

        # 将输入特征图转换为序列
        down4=self.down4(x)
        down4=F.relu(down4)
        b, c, h, w = x.size()
        x_seq = rearrange(down4, 'b c h w -> (h w) b c')

        # Transformer编码
        x_seq = self.transformer(x_seq)

        # 将序列转换回特征图
        x_seq = rearrange(x_seq, '(h w) b c -> b c h w', h=h, w=w)
        up4=self.up4(x_seq)
        up4=F.relu(up4)

        # # ————————————SK部分(通道注意力)————————————
        #
        # # 特征融合
        # # x1, x2, x3, x4, x5
        # feats = torch.stack([x1, x2], dim=1)  # [B, 6, C, H, W]
        #
        # # 全局平均池化
        # # x1 + x2 + x3 + x4 + x5
        # U = x1 + x2
        # s = self.global_pool(U)  # [B, C, 1, 1]
        # s = s.view(s.size(0), -1)  # [B, C]
        #
        # # 通过全连接层生成权重
        # z = self.fc1(s)
        # z = nn.ReLU(inplace=False)(z)
        # a_b = self.fc2(z)  # [B, 6*C]
        # # 6
        # a_b = a_b.view(a_b.size(0), 2, -1)  # [B, 6, C]
        #
        # # 使用softmax生成权重
        # attention = self.softmax(a_b)  # [B, 6, C]
        #
        # # 将生成的值与原先的两个特征图相乘
        # attention = attention.unsqueeze(-1).unsqueeze(-1)  # [B, 6, C, 1, 1]
        # out = (feats * attention).sum(dim=1)  # [B, C, H, W]
        #
        # out = self.norm(out)
        # # 将原始特征图与输出特征图作点乘
        # out = out * x

        # ————————————新CA部分(通道注意力)————————————

        # 计算注意力权重
        # x_seq
        attn_combined_c = torch.cat([x1, up4], dim=1)  # [B, 2*C, H, W]
        global_avg_attn = self.global_pool(attn_combined_c)  # [B, 2*C, 1, 1]

        # 使用 sigmoid 激活函数生成注意力权重
        global_avg_attn = global_avg_attn.view(b, 2, c)  # [B, 2, C]
        sig_c = global_avg_attn.sigmoid()  # [B, 2, C]

        # 生成适合广播的注意力权重
        sig_c = sig_c.view(b, 2, c, 1, 1)  # [B, 2, C, 1, 1]

        # 应用生成的注意力权重
        attn1_c = x1 * sig_c[:, 0, :, :]  # [B, C, H, W]
        # x_seq
        attn2_c = up4 * sig_c[:, 1, :, :]  # [B, C, H, W]

        # 融合注意力特征图
        attn_fused_c = attn1_c + attn2_c  # [B, C, H, W]

        # 与原始特征图进行逐元素乘法
        out = x * attn_fused_c  # [B, C, H, W]

        # 归一化
        out = self.norm(out)

        # ————————————LSK部分(空间注意力)————————————

        # 结合注意力特征图
        attn1 = x1
        # x_seq
        attn2 = up4

        attn_combined = torch.cat([attn1, attn2], dim=1)  # [B, 2*C, H, W]

        # 生成注意力权重
        avg_attn = torch.mean(attn_combined, dim=1, keepdim=True)  # 平均池化 [B, 1, H, W]
        max_attn, _ = torch.max(attn_combined, dim=1, keepdim=True)  # 最大池化 [B, 1, H, W]
        # agg = torch.cat([avg_attn, max_attn], dim=1)  # 结合平均池化和最大池化 [B, 2, H, W]
        # sig = self.conv_squeeze(agg).sigmoid()  # 使用 sigmoid 激活函数生成注意力权重 [B, 2, H, W]

        avg_attn = self.conv_squeeze_s(avg_attn)  # 对avg_attn使用3x3卷积
        max_attn = self.conv_squeeze_l(max_attn)  # 对max_attn使用7x7卷积
        agg = torch.cat([avg_attn, max_attn], dim=1)  # 结合卷积后的avg_attn和max_attn [B, 2, H, W]
        sig = agg.sigmoid()  # 使用 sigmoid 激活函数生成注意力权重 [B, 2, H, W]

        # 应用生成的注意力权重
        attn1 = x1 * sig[:, 0, :, :].unsqueeze(1)  # [B, C, H, W]
        # x_seq
        attn2 = up4 * sig[:, 1, :, :].unsqueeze(1)  # [B, C, H, W]

        # 融合注意力特征图
        attn_fused = attn1 + attn2  # [B, C, H, W]

        # 与原始特征图进行逐元素乘法
        out_sa = x * attn_fused  # [B, C, H, W]

        # 对新旧输出进行归一化
        out_sa = self.norm_s(out_sa)

        # 将原始特征图与新旧输出特征图做点乘
        # *
        out = out + out_sa

        out = x + out

        return out


# 自定义FPN
@MODELS.register_module()
class L2GFPNSFM(BaseModule):
    def __init__(self, in_channels, out_channels=256, norm_cfg=None, init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')):
        super(L2GFPNSFM, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 使用 ModifiedChannelMapper 处理输入特征图
        self.channel_mapper = ModifiedChannelMapper(
            in_channels=in_channels,  # 输入 [512, 1024, 2048]
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            # num_outs=4
        )

        # 每个特征图的复杂特征融合模块（使用修改后的SKNet）
        self.fusion_modules = nn.ModuleList([
            ComplexFeatureFusionModule(out_channels, norm_cfg) for _ in range(3)  # 原4改为3
        ])

        # ----------------- 第四个输出生成模块 -----------------
        self.extra_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg
        )

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        # # 打印每个输入特征图的维度
        # for i, x in enumerate(inputs):
        #     print(f"Input {i} shape: {x.shape}")

        assert len(inputs) == len(self.in_channels)

        # 使用 ModifiedChannelMapper 处理输入特征图
        inputs = self.channel_mapper(inputs)

        # 复杂特征融合（使用修改后的SKNet）  # td_features[i]
        fusion_features = [fusion(inputs[i]) for i, fusion in enumerate(self.fusion_modules)]

        # ================= 第六阶段：生成第四个输出（FPN/PAFPN标准方式）=================
        ###   bu_features[-1]   ###   bu_features.append(p6)   ###   tuple(bu_features)
        p5 = fusion_features[-1]  # 获取最高层特征P5 [B,256,H/32,W/32]
        p6 = self.extra_conv(p5)  # 通过步长2卷积生成P6 [B,256,H/64,W/64]
        fusion_features.append(p6)

        return tuple(fusion_features)  # 输出 (P3, P4, P5, P6)
