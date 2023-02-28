# DeiT
DeiT: Training data-efficient image transformers & distillation through attention

paper: [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)

offical code: [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)

#Usage:
1. Install timm=0.3.2
```Shell
pip install timm==0.3.2
```
2. Train or eval

# Train teacher 
```python
python teacher --gpu 0 --teacher_model "resnet34"
```

# Train student or Train teacher in deit/vit series
```python
# train without cache 
# --arch deit_tiny(default)/deit_small/deit_base/deit_base_384
python train.py --gpu 0
python train.py --gpu 0 --arch deit_small
python train.py --gpu 0 --arch deit_base
python train.py --gpu 0 --arch deit_base_384 input_size 384

# train with pretrained model
python train.py --gpu 0 --pretrained --model-path checkpoints/model_RegNetx_200mf_best.pt

# train with resume
python train.py --arch deit_tiny --resume checkpoints/model_deit_tiny_seed561_best.pt

python train.py --gpu 0 --teacher_model "resnet34" --distillation-type "soft" --distillation-alpha 0.5 --distillation-tau 1.0

```

# Validate
```python
# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_deit_tiny_seed561_best.pt
```
