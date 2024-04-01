## installations:

`conda env create -f requirements.yml`

`conda activate discriminative-token`


## Run and Evaluate:

To train and evaluate use:
`python o2m_run.py --class_index 283 --train True  --evaluate True`
or run the shell script:
`bash run.sh`

## Hyperparameters:
The hyperparameters can be changed in the `config.py` script.
