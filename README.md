# SurroundOcc ONNX with Custom Ops Exporting 

## Getting Started 
- [Installation](docs/install.md) 
- [Export Detail](docs/export.md)

## Export Command
```
python ./tools/export.py --config ./projects/configs/surroundocc/surroundocc.py --checkpoint ${checkpoint_path}
```

## 要替换的算子
- [x] MSDeformableAttention3D 在transformer中
- [x] resnet 中DCN的算子替换


