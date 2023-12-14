# SurroundOcc ONNX with Custom Ops Exporting 

## Getting Started 
- [Installation](docs/install.md) 
- [Export Detail](docs/export.md)

## Export Command
```
python ./tools/export.py --config ./projects/configs/surroundocc/surroundocc.py --checkpoint ${checkpoint_path}
```

## FEATURE
- [x] transformer中的MSDeformableAttention3D，使用cpu导出的话，可以不使用该算子，如果使用gpu的model导出，则需要支持该算子的plugin。
- [x] resnet 中DCN的算子替换。
- [x] cpu模式支持：若使用cpu模式，MSDeformableAttention3D走的是`multi_scale_deformable_attn_pytorch`函数，不需要额外的plugin。
- [ ] cuda模式支持：cuda模式下MSDeformableAttention3D将走plugin。本人使用的是NV1080Ti，走cuda模式的导出会爆显存,目前未成功导出过
- [x] tag v0.1.1 的deformableConv op算子和内网的对齐，目前mmcv v1.4.0 与当前的实现有所不同


