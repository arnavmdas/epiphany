# Helper functions for downstream application

## Pretrained Epiphany model weights

- Trained weights for 10kb resolution model can be obtained from
```
wget -O pretrained_GM12878.pt_model https://wcm.box.com/shared/static/vv8xzxnurfk8ddjwuc9evkhapl6fj0tu.pt_model
```
- Trained weights for 5kb resolution model can be obtained from
```
wget -O pretrained_GM12878_5kb.pt_model https://wcm.box.com/shared/static/wo6bc6elqw0leivm7w9gnyte6sgos5om.pt_model
```

## Colab for application examples 

- Generate contact map of GM12878 chromosome 3 using pre-trained model at 10kb resolution: [Google colab](https://colab.research.google.com/drive/1DhnboWQvZcltbXKYzHrfm8JSBu9xG4M3?usp=sharing)
- Generate contact map of a certain region on H1ES cell chromosome 8 [chr8:53167500-55167500] with original and perturbed epigenomic signals using pretrained model at 5kb resolution: [Google colab](https://colab.research.google.com/drive/1KjWXWl3OEZXrZGtu-rkG0nw0Odix6syP?authuser=1#scrollTo=opwMnpPFJaC7)