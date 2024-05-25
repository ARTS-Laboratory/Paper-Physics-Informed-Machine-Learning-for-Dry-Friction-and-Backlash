## Training Reproduction
Training physics-informed models, analysis, and plotting. This folder contains everything required to train numerical and experimental case studies. Training uses `tensorflow` 2.13.0.

`main.py` imports all scripts and runs them in the correct order. The models are first trained and saved to `/model_saves`. Then, validation tests are performed on the trained models and saved to `/model predictions`. The models and validation tests are used to create plots that are saved in `/plots`. 
