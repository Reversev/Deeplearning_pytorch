
# RepVGG
RepVGG: Making VGG-style ConvNets Great Again [https://arxiv.org/abs/2101.03697]

#Usage:
```python
# train without cache
python train.py --gpu 1
# train with pretrained model
python train.py --pretrained --model-path checkpoints/model_RegNetx_200mf_best.pt
# train with resume
python train.py --resume checkpoints/model_RegNetx_200mf_best.pt

# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_RegNetx_200mf_best.pt
```
