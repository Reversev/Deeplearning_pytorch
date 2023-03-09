
# EfficientNetv2
EfficientNetV2: Smaller Models and Faster Training [https://arxiv.org/abs/2104.00298]

#Usage:
```python
# train without cache
python train.py --gpu 1
# train with pretrained model
python train.py --pretrained --model-path checkpoints/model_effnetv2_s_best.pt
# train with resume
python train.py --resume checkpoints/model_effnetv2_s_best.pt

# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_effnetv2_s_best.pt
```
Top-1: 90.581 (seed:32)
