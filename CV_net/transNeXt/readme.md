# TransNeXt
TransNeXt: Robust Foveal Visual Perception for Vision Transformers

paper: [https://arxiv.org/pdf/2311.17132.pdf](https://arxiv.org/pdf/2311.17132.pdf)

offical code: [https://github.com/DaiShiResearch/TransNeXt](https://github.com/DaiShiResearch/TransNeXt)

#Usage:
1. Install timm=0.3.2
```Shell
pip install timm==0.3.2
```
2. Train or eval
# Train model in transNeXt series
```python
# train without cache 
python train.py --gpu 0
python train.py --gpu 0 --arch 'transnext_micro' --batch_size 16     # 'transnext_tiny', 'transnext_small', 'transnext_base'

# train with resume
python train.py --arch deit_tiny --resume checkpoints/model_transnext_micro_seed561_best.pt --gpu 0

```

# Validate
```python
# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_transnext_micro_seed561_best.pt
```

# Predict
Modify ```model_name```, ```dataset_name```, ```MODEL_PATH``` and ```CLASS_NUM``` in ```predict.py``` script. Put pictures into ```results``` directory.
```python
python predict.py
```
