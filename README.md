Semantics-Guided Selective Fusion DETR in combination with SAM for Ship detection and segmentation in multi-mode Remote Sensing Images
=
Jiajie Ma, Lijun Zhao, Lianzhi Huo and Zhiqing Zhang

Abstract:    
In remote sensing, the complex backgrounds and noise present in various remote sensing images make ship localization challenging. Achieving high-precision detection and segmentation remains a difficult task. The DEtection TRansformer (DETR) model combines Convolutional Neural Networks (CNNs) and Transformers, but its detection performance is limited due to suboptimal feature map matching. To address this limitation, this paper proposes a high-precision object detection network, termed Semantics-Guided Selective Fusion DETR (SGSF-DETR), which leverages image semantics to guide multidimensional feature fusion. This network consists of three core modules. The first module, the Semantic-Aware Deformable Convolution Group (SADCG), utilizes a semantic-guided convolution kernel movement to extract target-relevant features in a multi-scale and adaptive manner, enhancing the localization capability while simultaneously suppressing interference. The second module, the Selective Fusion Mechanism (SFM), selectively fuses key information across both the channel and spatial dimensions of the CNN-Transformer joint feature map. The third module, FreeFlowFPN (FF-FPN), enables flexible information transfer across different levels of feature maps, thereby enhancing the representational capability. By using the optical remote sensing image dataset HRSC2016 and the Synthetic Aperture Radar (SAR) image dataset SSDD, the performance of SGSF-DETR is compared with that of other popular models. The results demonstrate that SGSF-DETR outperforms other models and exhibits excellent generalization. Additionally, we propose a segmentation mode that integrates the detection model with the Segment Anything Model (SAM). The experiments confirm that this approach achieves high accuracy and is highly transferable when applied to visible light remote sensing images. 

Official Pytorch implementation of our model.

Train
-
```python tools/train.py --config```

Test
-
```python tools/test.py --config --ckpt```
