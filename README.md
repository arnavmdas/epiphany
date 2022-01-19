# Epiphany: predicting Hi-C contact maps from 1D epigenomic signals

## Quick start training
### Clone Repository
```
git clone https://github.com/arnavmdas/epiphany.git
```

### Training
Move to training directory
```
cd epiphany/epiphany
```

Download dataset from google drive

```
mkdir ./Epiphany_dataset
cd ./Epiphany_dataset
wget --no-check-certificate https://drive.google.com/drive/u/2/folders/1UJX6cp-4s0Jbud9jovzuaqnBeORg5R8x GM12878_X.h5
wget --no-check-certificate https://drive.google.com/drive/u/2/folders/1UJX6cp-4s0Jbud9jovzuaqnBeORg5R8x GM12878_y.pickle
cd ..
```


Run training script
```
python3 adversarial.py --wandb
```
