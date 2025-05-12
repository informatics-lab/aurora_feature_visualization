# Aurora Feature Visualisation

**Description/Summary**

## Set up

### Environments
Use the following command to create the **aurora_fv** conda environment.
```bash
conda env create -f requirements.yml
```

### Checkpoints
[Hugging Face Aurora](https://huggingface.co/microsoft/aurora)

```bash
curl https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-small-pretrained.ckpt -o aurora-0.25-small-pretrained.ckpt
```

## Run Scripts
All of the scripts are designed to be run from the root of the repository.

```bash
conda activate aurora_fv
python scripts/experiments/aurora_inference.py
```
