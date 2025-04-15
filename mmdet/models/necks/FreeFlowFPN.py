import torch
import torch.nn as nn
import torch.nn.functional as F

class FreeFlowFPN(nn.Module):
    def __init__(self, in_channels, out_channels, tau=0.1):
        """
        自由流动 FPN (Free-Flow Feature Pyramid Network)
        :param in_channels: 每个特征层的通道数
        :param out_channels: 每层输出的通道数
        :param tau: 剪枝阈值（低于该值的路径直接剪掉）
        """
        super().__init__()
        self.tau = tau  # 剪枝阈值

        # 计算启动顺序的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出重要性分数
        )

        # 计算 QK，用于路径权重
        self.qk_proj = nn.Linear(in_channels, in_channels * 2)

        # FPN 细化卷积层 (3x3 卷积)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(4)
        ])

    def forward(self, features):
        batch_size, channels, _, _ = features[0].shape
        num_layers = len(features)

        # 计算每层的重要性分数
        feature_vecs = [F.adaptive_avg_pool2d(f, (1, 1)).view(batch_size, -1) for f in features]
        feature_tensor = torch.stack(feature_vecs, dim=1)
        scores = self.mlp(feature_tensor).squeeze(-1)
        order = torch.argsort(scores, dim=1, descending=True)  # 按重要性降序排序

        # 计算 QK 注意力
        qk = self.qk_proj(feature_tensor)
        q, k = torch.chunk(qk, 2, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (channels ** 0.5)

        # 🚨 **屏蔽自身权重（在 softmax 之前！）**
        attn[:, torch.arange(num_layers), torch.arange(num_layers)] = float('-inf')

        attn = F.softmax(attn, dim=-1)  # 归一化

        # 3️⃣ 计算新的特征图（信息融合）
        new_features = [features[i].clone() for i in range(num_layers)]  # 先保留原始特征

        # **所有层都可以向其他层传播**
        for i in range(num_layers):  # **按照 order 顺序依次传播**
            idx = order[:, i]  # 取当前层索引

            for j in range(num_layers):  # **它可以向所有层传播**
                target_idx = order[:, j]  # 目标层索引
                if idx == target_idx:  # 🚨 **跳过自身**
                    continue

                weight = attn[:, idx, target_idx].unsqueeze(-1).unsqueeze(-1)
                if weight.mean() < self.tau:  # 剪枝
                    continue

                # 上/下采样至目标层的尺度
                scaled_feature = F.interpolate(new_features[idx], size=features[target_idx].shape[2:], mode='bilinear',
                                               align_corners=False)
                new_features[target_idx] += weight * scaled_feature  # 传播给目标层

        # 2️⃣ 对每个层的融合特征应用 3x3 卷积进行细化
        refined_features = [F.relu(self.fpn_convs[i](new_features[i])) for i in range(num_layers)]

        return refined_features
