# MobileViT
MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer 

paper: [https://arxiv.org/abs/2110.02178](https://arxiv.org/abs/2110.02178)

offical code: [https://github.com/apple/ml-cvnets](https://github.com/apple/ml-cvnets)

# Usage:
```python
# train without cache 
# --arch mobilevit_xxs/moblevit_xs/mobilevit_s
python train.py --gpu 1
# train with pretrained model
python train.py --pretrained --model-path checkpoints/model_RegNetx_200mf_best.pt
# train with resume
python train.py --resume checkpoints/model_RegNetx_200mf_best.pt

# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_RegNetx_200mf_best.pt
```
