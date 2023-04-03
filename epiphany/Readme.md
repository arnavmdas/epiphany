# Epiphany model and training 

Scripts in this folder creates data loader, model structure, and training framework: 
- `data_loader_10kb.py`: takes in `GM12878_X.h5` and `GM12878_y.pickle`, and prepare dataloader for Epiphany training
- `make_data.py`: helper functions for preparing dataloader
- `model_10kb.py`: Epiphany model (for 10kb Hi-C map prediction), including both the encoder and the discriminator
- `adversarial.py`: Epiphany training script
- `utils.py`: other helper functions