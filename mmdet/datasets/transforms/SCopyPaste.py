"""实现目标的复制粘贴，支持："""
import math
import cv2
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class SCopyPaste(BaseTransform):
    def __init__(self, paste_ratio=0.5, horizontal_flip_prob=0.5, vertical_flip_prob=0.5,
                 resize_ratio=0.5, scale_limit=1.2, max_attempts=50):
        self.paste_ratio = paste_ratio  # 复制粘贴的概率
        self.horizontal_flip_prob = horizontal_flip_prob  # 水平翻转的概率
        self.vertical_flip_prob = vertical_flip_prob  # 垂直翻转的概率
        self.resize_ratio = resize_ratio  # 放大或缩小的概率
        self.scale_limit = scale_limit  # 放大和缩小的倍数上限
        self.max_attempts = max_attempts  # 最大尝试次数以找到合适的无重叠位置

    @cache_randomness
    def _should_paste(self):
        """随机决定是否进行复制粘贴操作。"""
        return np.random.rand() < self.paste_ratio

    @cache_randomness
    def _should_flip_horizontal(self):
        """随机决定是否进行水平翻转。"""
        return np.random.rand() < self.horizontal_flip_prob

    @cache_randomness
    def _should_flip_vertical(self):
        """随机决定是否进行垂直翻转。"""
        return np.random.rand() < self.vertical_flip_prob

    @cache_randomness
    def _get_scale(self):
        """根据 resize_ratio 确定放大或缩小倍数。"""
        if np.random.rand() < self.resize_ratio:
            return np.random.uniform(1, self.scale_limit)  # 放大
        else:
            return np.random.uniform(1 / self.scale_limit, 1)  # 缩小

    def _bbox_overlap(self, bbox1, bbox2):
        """计算两个边界框的交并比（IoU）。"""
        x1, y1, x2, y2 = np.maximum(bbox1[0], bbox2[0]), np.maximum(bbox1[1], bbox2[1]), \
                         np.minimum(bbox1[2], bbox2[2]), np.minimum(bbox1[3], bbox2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def transform(self, results):
        """实现复制粘贴增强。"""
        img = results['img']  # 获取图像数据
        height, width = img.shape[:2]
        original_bboxes = results['gt_bboxes'].tensor.numpy().copy()  # 获取原始边界框列表
        original_labels = results['gt_bboxes_labels'].copy()  # 获取原始边界框对应的标签

        if len(original_bboxes) == 0:
            return results  # 如果没有边界框，直接返回原结果

        for bbox, label in zip(original_bboxes, original_labels):
            x1, y1, x2, y2 = bbox  # 提取边界框的坐标，并转换为列表

            if self._should_paste():
                w, h = x2 - x1, y2 - y1  # 计算宽和高

                # 提取目标区域
                copied_region = img[int(y1):int(y2), int(x1):int(x2)].copy()

                # 按照概率进行水平和垂直翻转
                if self._should_flip_horizontal():
                    copied_region = cv2.flip(copied_region, 1)  # 水平翻转
                    x1, x2 = width - x2, width - x1  # 更新水平翻转后的边界框

                if self._should_flip_vertical():
                    copied_region = cv2.flip(copied_region, 0)  # 垂直翻转
                    y1, y2 = height - y2, height - y1  # 更新垂直翻转后的边界框

                scale = self._get_scale()  # 获取放大或缩小倍数
                new_w = int(w * scale)
                new_h = int(h * scale)

                # 检查新尺寸是否有效
                if new_w <= 0 or new_h <= 0:
                    continue

                # 更新边界框的尺寸
                new_x1, new_y1 = x1, y1
                new_x2 = new_x1 + new_w
                new_y2 = new_y1 + new_h

                # 确保边界框不越界
                new_x2 = min(new_x2, img.shape[1])
                new_y2 = min(new_y2, img.shape[0])
                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1

                new_bbox = [new_x1, new_y1, new_x2, new_y2]  # 缩放后的边界框

                # 使用合适的插值方法
                if scale >= 1:
                    interpolation = cv2.INTER_CUBIC  # 放大时使用三次插值
                else:
                    interpolation = cv2.INTER_AREA  # 缩小时使用区域插值

                # 转换为整数类型
                new_w = int(new_w)
                new_h = int(new_h)

                copied_region = cv2.resize(copied_region, (new_w, new_h), interpolation=interpolation)

                # 创建一个全白掩膜
                mask = np.ones((new_h, new_w), dtype=np.uint8) * 255

                # 多范围角度随机选择旋转
                angle_ranges = [(-10, 10), (80, 100), (170, 190), (260, 280)]
                angles = np.concatenate(angle_ranges)
                angle = np.random.choice(angles)
                # angle = np.random.uniform(0, 360)  # 全角度随机选择旋转

                M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)

                # 计算旋转后区域的外接框
                new_rotated_bbox = self._rotated_rect_bbox(new_bbox, angle, new_w, new_h)

                # 确保旋转后的图像块大小与其外接矩形匹配
                final_rotated_bbox_size = (new_rotated_bbox[2] - new_rotated_bbox[0],
                                           new_rotated_bbox[3] - new_rotated_bbox[1])

                # 确保边界不超过图像尺寸
                final_rotated_bbox_size = (min(final_rotated_bbox_size[0], img.shape[1]),
                                           min(final_rotated_bbox_size[1], img.shape[0]))

                M[0, 2] += (final_rotated_bbox_size[0] - new_w) // 2
                M[1, 2] += (final_rotated_bbox_size[1] - new_h) // 2

                # 进行仿射变换，生成旋转后的图像和掩膜
                rotated_copied_region = cv2.warpAffine(copied_region, M, (final_rotated_bbox_size[0], final_rotated_bbox_size[1]), flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                rotated_mask = cv2.warpAffine(mask, M, (final_rotated_bbox_size[0], final_rotated_bbox_size[1]),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                for _ in range(self.max_attempts):
                    # 检查范围是否有效
                    if img.shape[1] - final_rotated_bbox_size[0] <= 0 or img.shape[0] - final_rotated_bbox_size[1] <= 0:
                        break

                    # 尝试找到无重叠的位置
                    offset_x = np.random.randint(0, img.shape[1] - final_rotated_bbox_size[0])
                    offset_y = np.random.randint(0, img.shape[0] - final_rotated_bbox_size[1])

                    new_x1 = offset_x
                    new_y1 = offset_y
                    new_x2 = new_x1 + final_rotated_bbox_size[0]
                    new_y2 = new_y1 + final_rotated_bbox_size[1]

                    final_bbox = [new_x1, new_y1, new_x2, new_y2]  # 新的边界框

                    # 检查新bbox是否与现有bbox重叠
                    overlaps = [self._bbox_overlap(final_bbox, existing_bbox) for existing_bbox in results['gt_bboxes'].tensor.numpy()]
                    if max(overlaps) == 0:  # 设置不允许任何重叠
                        # 仅在掩膜为白色的区域粘贴新图像区域
                        mask_inv = cv2.bitwise_not(rotated_mask)
                        img_region = img[new_y1:new_y2, new_x1:new_x2]

                        img_region_bg = cv2.bitwise_and(img_region, img_region, mask=mask_inv)  # 保留原图像中的非掩膜部分
                        rotated_copied_region_fg = cv2.bitwise_and(rotated_copied_region, rotated_copied_region,
                                                                   mask=rotated_mask)  # 提取掩膜部分

                        img[new_y1:new_y2, new_x1:new_x2] = cv2.add(img_region_bg, rotated_copied_region_fg)  # 组合背景和前景
                        results['gt_bboxes'] = HorizontalBoxes(np.vstack((results['gt_bboxes'].tensor.numpy().astype(np.float32), np.array(final_bbox).astype(np.float32))))  # 添加新边界框
                        results['gt_bboxes_labels'] = np.append(results['gt_bboxes_labels'], label)  # 添加新标签
                        break  # 找到合适的位置后退出

        results['img'] = img  # 更新图像数据
        # results['gt_bboxes'] = results['gt_bboxes'].astype(np.float32)
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'].astype(np.int64)
        return results  # 返回更新后的结果

    def _rotated_rect_bbox(self, bbox, angle, width, height):
        """计算旋转后的矩形的外接框。"""
        x1, y1, x2, y2 = bbox

        # 中心点
        cx, cy = width // 2, height // 2

        # 四个角点的相对坐标
        corners = np.array([
            [x1 - cx, y1 - cy],
            [x2 - cx, y1 - cy],
            [x2 - cx, y2 - cy],
            [x1 - cx, y2 - cy]
        ])

        # 旋转角度（弧度）
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # 旋转所有角点
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # 获取旋转后的外接框
        new_x1 = int(np.min(rotated_corners[:, 0]) + cx)
        new_y1 = int(np.min(rotated_corners[:, 1]) + cy)
        new_x2 = int(np.max(rotated_corners[:, 0]) + cx)
        new_y2 = int(np.max(rotated_corners[:, 1]) + cy)

        return [new_x1, new_y1, new_x2, new_y2]

    def __repr__(self):
        return (f"{self.__class__.__name__}(paste_ratio={self.paste_ratio}, "
                f"horizontal_flip_prob={self.horizontal_flip_prob}, "
                f"vertical_flip_prob={self.vertical_flip_prob}, "
                f"resize_ratio={self.resize_ratio}, "
                f"scale_limit={self.scale_limit}, "
                f"max_attempts={self.max_attempts})")
