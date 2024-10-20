# MobileNetV4
MobileNetV4 - Universal Models for the Mobile Ecosystem

paper: [http://arxiv.org/abs/2404.10518](http://arxiv.org/abs/2404.10518)

offical code: [https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py(tensorflow)](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py(tensorflow))

# Usage
1. Install timm=0.3.2
```Shell
pip install timm==0.3.2
```
2. Train or eval
# Train model in MobileNetv4 series (MNV4ConvSmall, MNV4ConvMedium, MNV4ConvLarge, MNV4HybridMedium, MNV4HybridLarge)
```python
# train without cache 
python train.py --gpu 0
python train.py --gpu 0 --arch 'MNV4ConvSmall' --batch_size 16     # 'MNV4ConvSmall, MNV4ConvMedium', 'MNV4ConvLarge', 'MNV4HybridMedium', 'MNV4HybridLarge'

# train with resume
python train.py --arch MNV4ConvSmall --resume checkpoints/model_MNV4ConvSmall_seed561_best.pt --gpu 0

```

# Validate
```python
# validate
python train.py --evaluate --gpu 1 --resume checkpoints/model_MNV4ConvSmall_seed561_best.pt
```

# Predict
Modify ```model_name```, ```dataset_name```, ```MODEL_PATH``` and ```CLASS_NUM``` in ```predict.py``` script. Put pictures into ```results``` directory.
```python
python predict.py   # you need to modify model path at 18th line in the script. Default: './checkpoints/model_MNV4HybridMedium_seed772_best.pt'  
```
