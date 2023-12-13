# EXPORT
```
python ./tools/export.py --config ./projects/configs/surroundocc/surroundocc.py --checkpoint ./ckpts/r101_dcn_fcos3d_pretrain.pth
```

# 要替换的算子
- [x] MSDeformableAttention3D 在transformer中
- [x] resnet 中DCN的算子替换