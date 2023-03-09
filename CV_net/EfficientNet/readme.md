
# EfficientNetv1
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks [https://arxiv.org/abs/1905.11946]

#Usage:
```python
# train without cache
python train.py --gpu 1
# train with pretrained model
python train.py --pretrained --model-path checkpoints/model_effnetv1_b1_best.pt
# train with resume
python train.py --resume checkpoints/model_effnetv1_b1_best.pt

# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_effnetv1_b1_best.pt
```

