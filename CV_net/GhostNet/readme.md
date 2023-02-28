# GhostNet
GhostNet: More Features from Cheap Operations [https://arxiv.org/abs/1911.11907]

#Usage:
```pythons
# train without cache
python train.py --gpu 1
# train with pretrained model
python train.py --pretrained --model-path checkpoints/model_ghostnet_seed578_best.pt.pt
# train with resume
python train.py --resume checkpoints/model_ghostnet_seed578_best.pt.pt

# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_ghostnet_seed578_best.pt.pt
```

Top-1: 84.884 (seed:578)