#### installations:

`conda env create -f requirements.yml`

`conda activate discriminative-token`


## Run and Evaluate:

To train and evaluate use:
`python run.py --class_index 283 --train True  --evaluate True`

#### Hyperparameters:
The hyperparameters can be changed in the `config.py` script. Note that the paper results are based on stable-diffusion version 1.4.
