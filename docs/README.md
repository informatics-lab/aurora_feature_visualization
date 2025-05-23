# Aurora Feature Visualisation
This respository implements [Feature Visualisation via Activation Maximisation](https://distill.pub/2017/feature-visualization/) for the [Aurora](https://github.com/microsoft/aurora/blob/main/aurora/model/aurora.py) model.

## Set up

### Environments
Use the following command from the root of the repository to create the **aurora_fv** conda environment:
```bash
conda env create -f requirements.yml
```

## Run Scripts
All of the scripts are designed to be run from the root of the repository and with the aurora_fv conda environment activated using the following command:
```bash
conda activate aurora_fv
```

### Feature Visualisations
To run the script with the default arguments, use the following command:
```bash
python src/feature_visualisation.py
```

Use the --help / -h argument to see the available parameters to select the targeted layer, neuron, save directory, to not display the plot, etc.
```bash
python src/feature_visualisation.py --help
```


### Display Layer Names
To target a specific layer in the model, you need to know what the exact name of that layer is. You can get that by running the following script:
```bash
python src/display_layer_names.py
```
It is important to note that the names of the layers change when using Aurora with autocast (which is turned on by default).

### Calculate Decorrelation
The feature visualisation script uses a decorrelation matrix with the values being hard coded into decorrelation.py. These values were calculated from 1 year of ERA5 data. To recaculate these vaules you can use the following scripts.

```bash
python src/download_data_gcs.py --years 2018 --months 1 2 3 4 5 6 7 8 9 10 11 12 --days 1 15
python src/caculate_era5_decorrelate_gcs.py
```
