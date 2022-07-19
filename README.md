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
wget --no-check-certificate https://drive.google.com/drive/u/2/folders/1UJX6cp-4s0Jbud9jovzuaqnBeORg5R8x -O GM12878_X.h5
wget --no-check-certificate https://drive.google.com/drive/u/2/folders/1UJX6cp-4s0Jbud9jovzuaqnBeORg5R8x -O GM12878_y.pickle
cd ..
```


Run training script
```
python3 adversarial.py --wandb
```

### Prediction using pretrained models 

- Generate contact map of GM12878 chromosome 3 using pre-trained model at 10kb resolution: [Google colab](https://colab.research.google.com/drive/1DhnboWQvZcltbXKYzHrfm8JSBu9xG4M3?usp=sharing)
- Generate contact map of a certain region on H1ES cell chromosome 8 [chr8:53167500-55167500] with original and perturbed epigenomic signals using pretrained model at 5kb resolution: [Google colab](https://colab.research.google.com/drive/1KjWXWl3OEZXrZGtu-rkG0nw0Odix6syP?authuser=1#scrollTo=opwMnpPFJaC7)