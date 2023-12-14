# Bring Up Doc 
源码链接： https://github.com/weiyithu/SurroundOcc


## 环境安装
使用NV 1080Ti的机器进行导出，mmdet3d需要nv环境
[环境安装指南](https://github.com/weiyithu/SurroundOcc/blob/main/docs/install.md)

## 算子替换
### 1. resnet中的DCN算子替换
#### 1.1 ResNet算子替换
仿照projects/mmdet3d_plugin/surroundocc/dense_heads 形式，将resnet迁移到当前仓库，方便源码的修改。代码结构如下
```
|--projects/mmdet3d_plugin/surroundocc/
|  |--dense_heads
|  |--backbone
|  |  |--resnet_dl.py
|  |  |--__init__.py
```
将resnet_dl.py 中的372行替换成文件中自定义的DeformableConvPack即可。

类DeformableConvPack、DeformableConvPack参考自[mmcv_deform](https://github.com/open-mmlab/mmcv/blob/2e44eaeba36b3f4c304e830053fc2660d8407afb/mmcv/ops/modulated_deform_conv.py#L22)

#### 1.2 算子注册
在 projects/mmdet3d_plugin/surroundocc/__init__.py 加入下述行
```
from .backbone import *
```
### 2. spatial_cross_attention 中的MultiScaleDeformableAttnFunction算子替换
#### 2.1 算子注册
函数MultiScaleDeformableAttnFunction_fp32已经提供了清楚的输入输出大小，根据大小传递空数据，并将算子进行注册，定义函数MultiScaleDeformableAttnFunction_fp32_DL进行调用。
```
class ModulatedDeformConv2dFunction(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, offset, weight, stride, padding,
                 dilation, groups, deform_groups, bias=False, im2col_step=32):
        
        return g.op(
            'DL::DeformConv2d',
            input,
            offset,
            weight,
            stride_i=stride,
            padding_i=_quadruple(padding[0]),
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups,
            bias_i=bias,
            im2col_step_i=im2col_step)

    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deform_groups=1,
                bias=None,
                im2col_step=32):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        ctx.im2col_step = im2col_step
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        # ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        ext_module.deform_conv_forward(
            input,
            weight,
            offset,
            output,
            ctx._bufs[0],
            ctx._bufs[1],
            kW=weight.size(3),
            kH=weight.size(2),
            dW=ctx.stride[1],
            dH=ctx.stride[0],
            padW=ctx.padding[1],
            padH=ctx.padding[0],
            dilationW=ctx.dilation[1],
            dilationH=ctx.dilation[0],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            im2col_step=cur_im2col_step)
        return output
```
#### 2.2 MSDeformableAttention3D类注册
为了避免覆盖源码，在projects/mmdet3d_plugin/surrondocc/modules/spatial_cross_attention.py 定义了类MSDeformableAttention3D_DL， 该类与源码的区别就是调用了MultiScaleDeformableAttnFunction_fp32_DL


## onnx export 导出适配

1. projects/mmdet3d_plugin/surroundocc/detectors/surroundocc.py：forward函数修改
    1. forward函数直接调用`forward_test`
    ```
    def forward(self, img, img_meta):
        return self.forward_test(img_meta, img)
    ```
    2. forward_test 函数修改为
    ```
    def forward_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        output = self.simple_test(
            img_metas, img, **kwargs)
        
        pred_occ = output['occ_preds']
        if self.use_semantic:
            for i in range(4):
                _, pred_occ[i] = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
            return pred_occ
    ``` 
2. projects/mmdet3d_plugin/surroundocc/detectors/surroundocc.py:L76行，导出时onnx提示不支持in-place的squeeze，因此修改如下:
    ```
    # origin code
    img.squeeze_(0)
    # new code 
    img = img.squeeze(0)
    ```
3. lidar2img在export中是torch.tensor需要进行适配，在projects/mmdet3d_plugin/surroundocc/modules/encoder.py:L84行附近修改为
    ```
    # lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)        # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    lidar2img = torch.stack(lidar2img, dim=0)
    ```
4. torch.maximum算子在onnx中不支持，替换为torch.max
    ```
    # projects/mmdet3d_plugin/surroundocc/modules/encoder.py:L113
    reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    ```
5. 导出时报错RuntimeError: axes_vector.size() <= input_shape.size()，通过debug发现修改如下代码解决
    ```
    #projects/mmdet3d_plugin/surroundocc/modules/spatial_cross_attention.py:L157
    # old code 
    queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
    reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j,
        index_query_per_img]
    # new code 
    queries_rebatch[j, i, :index_query_per_img.shape[0]] = query[j, index_query_per_img]
    reference_points_rebatch[j, i, :index_query_per_img.shape[0]] = 
        reference_points_per_img[j, index_query_per_img]
    ```
6. tensorrt 8.5已经支持`nonzero`算子，DL卡尚未支持，因此将`projects/mmdet3d_plugin/surroundocc/modules/spatial_cross_attention.py`内的下述内容进行修改，但这样会增加计算量，并增加显存的消耗

    ```
    for i, mask_per_img in enumerate(bev_mask):
        # Use nonzero will come out dynamic shape. so use all index.
        # index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        index_query_per_img = torch.arange(mask_per_img[0].size(0))
        indexes.append(index_query_per_img)
    ```
## Config 适配
Config 参考文件:  projects/configs/surroundocc/surroundocc.py
1. img_backbone 中的type=ResNet 修改为 type=ResNet_DL
2. OccLayer中的type=MSDeformableAttention3D 替换为 type=MSDeformableAttention3D_DL

## ONNX 导出
使用如下指令即可导出：
python ./tools/export.py --config ./projects/configs/surroundocc/surroundocc.py 
