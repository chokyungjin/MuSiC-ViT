# MuSiC-ViT: Multi-task Siamese Convolutional Vision Transformer for Differentiating Change or No-change of Follow-Up Chest X-Rays

---

![Figure_final_main](./image/Figure_final_main.png)

## Directory Architecture

Root

|---------- train.json (You have to create yourself.)

|---------- test.json (You have to create yourself.)

|---------- config.py

|---------- datasets.py

|---------- README.md

|---------- test.py

|---------- train.py

|---------- utils.py

|---------- runs (if you run the train code, it will be made automatically)

|---------- checkpoints (if you run the train code, it will be made automatically)

## Train

```
CUDA_VISIBLE_DEVICES=0 python train.py --msg=train--aug True --lr 6e-5 --batch_size=20 --print_freq=300 --backbone CMT_Ti
```

## Test

```
CUDA_VISIBLE_DEVICES=0 python test.py --msg=test --batch_size=6 --pretrained checkpoints/2022-08-25_163910_MuSiC_ViT/174048.pth --backbone CMT_Ti
```

MuSiC-ViT weight link: https://drive.google.com/drive/folders/1TZYdh4ERKBXa_-OH0LWxofGAsGWDc3_H?usp=sharing

CheXpert dataset link: https://drive.google.com/drive/folders/1wD_LI0mPlQWNWS47L44w2guuk4aHNmw-?usp=sharing

## Results

The results compared to other various model architectures are shown in the table below.

| Model architecture            || Param (M)  | Training (h) / Testing (s) | Internal validation dataset         | External validation dataset 1       | External validation dataset 2     |
|:----------|:-------------------|:----------:|:-----------:|:---------------------------------------:|:---------------------------------------:|:----------------------------------------:|
|           |                    |            |             |  SPE    　  SEN    　  ACC    　  AUC    |  SPE    　  SEN    　  ACC    　  AUC    |  SPE    　  SEN    　  ACC    　  AUC    |
| CNN       | Inception-v3       | 22.84M     | 31h / 36s   |  0.790  　  0.568  　  0.679  　  0.732  |  0.884  　  0.265  　  0.574  　  0.665  |  0.866  　  0.449  　  0.659  　  0.723  |
|           | ResNet-50          | 24.55M     | 36h / 24s   |  0.806  　  0.564  　  0.685  　  0.749  |  0.902  　  0.279  　  0.591  　  0.639  |  0.828  　  0.453  　  0.642  　  0.721  |
|           | DenseNet-121       | 7.48M      | 32h / 46s   |  0.752  　  0.595  　  0.674  　  0.722  |  0.828  　  0.363  　  0.595  　  0.662  |  0.869  　  0.532  　  0.702  　  0.758  |
|           | EfficientNet-b3    | 11.49M     | 29h / 44s   |  0.790  　  0.557  　  0.674  　  0.741  |  0.888  　  0.270  　  0.579  　  0.655  |  0.884  　  0.506  　  0.696  　  0.655  |
|           | EfficientNet-v2    | 21.23M     | 49h / 48s   |  0.736  　  0.594  　  0.665  　  0.712  |  0.726  　  0.437  　  0.581  　  0.649  |  0.866  　  0.543  　  0.705  　  0.760  |
|           | ConvNeXt           | 28.59M     | 67h / 29s   |  0.751  　  0.544  　  0.648  　  0.699  |  0.833  　  0.400  　  0.616  　  0.689  |  0.914  　  0.415  　  0.666  　  0.736  |
|Transformer| ViT-B              | 88.28M     | 33h / 17s   |  0.816  　  0.448  　  0.632  　  0.677  |  0.902  　  0.200  　  0.551  　  0.618  |**0.978**　  0.102  　  0.542  　  0.633  |
|           | Swin-v2            | 28.35M     | 54h / 37s   |  0.665  　  0.611  　  0.638  　  0.692  |  0.842  　  0.369  　  0.595  　  0.637  |  0.892  　  0.433  　  0.664  　  0.742  |
|           | MLP-Mixer          | 32.18M     | 52h / 24s   |  0.706  　  0.656  　  0.681  　  0.727  |  0.754  　**0.516**　  0.653  　  0.660  |  0.787  　  0.521  　  0.655  　  0.705  |
|           | ResMLP             | 20.21M     | 56h / 16s   |  0.704  　  0.649  　  0.677  　  0.724  |  0.707  　  0.414  　  0.561  　  0.609  |  0.758  　  0.411  　  0.585  　  0.617  |
|           | CoaT               | 11.01M     | 42h / 38s   |  0.709  　  0.651  　  0.680  　  0.734  |  0.795  　  0.335  　  0.565  　  0.609  |  0.724  　  0.555  　  0.640  　  0.698  |
|           | CMT-Ti             | 31.62M     | 29h / 59s   |  0.721  　**0.682**　  0.701  　  0.762  |  0.795  　  0.488  　**0.642**　  0.674  |  0.772  　**0.626**　  0.700  　  0.757  |
|           |**MuSiC-ViT (ours)**| 31.81M     | 39h / 70s   |**0.817**　  0.638  　**0.728  　  0.797**|**0.930**　  0.298  　  0.614  　**0.784**|  0.899  　  0.589  　**0.745  　  0.858**|

*Note: number of model parameters (Param);specificity (SPE); sensitivity (SEN); accuracy (ACC); area under receiver operating characteristics curve (AUC); million (M); hours (h); seconds (s)