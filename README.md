# MuSiC-ViT: Multi-task Siamese Convolutional Vision Transformer for Differentiating Change or No-change of Follow-Up Chest X-Rays

---

![Figure_final_main](https://user-images.githubusercontent.com/46750574/173975319-6f36e80a-627f-4856-921c-6aca52b737c8.png)

## Directory Architecture

Root

|---------- train.json (You have to create yourself.)

|---------- test.json (You have to create yourself.)

|---------- config.py

|---------- datasets_with_UNet.py

|---------- README.md

|---------- test.py

|---------- train.py

|---------- utils.py

|---------- cam_utils.py

|---------- gradcam.py

|---------- runs (if you run the train code, it will be made automatically)

|---------- checkpoints (if you run the train code, it will be made automatically)

## Train

```
CUDA_VISIBLE_DEVICES=0 python train.py --msg=train--aug True --lr 6e-5 --batch_size=20 --print_freq=300 --backbone CMT_Ti
```

## Test

```
CUDA_VISIBLE_DEVICES=0 python test.py --msg=test --batch_size=6 --pretrained checkpoints/2020-10-13_113615_sgd_change+disease+orth_res152_real/20306.pth --backbone CMT_Ti
```

MuSiC-ViT weight link: https://drive.google.com/file/d/1uJsrFra0hsL90Guz5vHuhy1BGebt4gc-/view?usp=sharing

Pre-trained UNet weight link: https://drive.google.com/file/d/1pxf0gTiDZyYFHOKhW16-K09i9Or4n5jx/view?usp=sharing

CheXpert dataset link: https://drive.google.com/drive/folders/1wD_LI0mPlQWNWS47L44w2guuk4aHNmw-?usp=sharing