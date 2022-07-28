# Towards Online Predictions of the Dreaming Activity: a Deep Learning approach

Source code related to my master thesis: *Towards Online Predictions of the Dreaming Activity: a Deep Learning approach*. Supervised by Prof. Martin Jaggi (EPFL) and Dr. Francesca Siclari (CHUV). This thesis follows the work of [[1]](#1).

## Structure

```
├── requirements.txt         # Library requirements for setting the environment
├── _data_/                  # Toy data used for demo
│   ├── metadata/            
│   ├── processed/
│   └── raw/
│
├── classification/          # Classification of dreams / sleep stages
├── plotting/                # Figure plotting
├── results/                 # Stores the results of pre-training
└── scripts/
    ├── data_processing/     # Data processsing modules
    │   ├── config/
    │   └── ...
    │
    ├── interaction          # Interaction modules
    └── training                
        ├── dataset/         # Dataset modules
        └── representation   # Pre-training modules
            ├── config/
            ├── models/
            └── ...
```

## Setup

You must have `conda` installed on your machine.
1. Create new conda environment at the desired location `path/to/env`: 

        conda create -p path/to/env --file requirements.txt -c conda-forge

2. Activate your environement: 

        conda activate /path/to/env

## Demo

All python scripts for data processing and pre-training are contained in `scripts/`. We provide toy data in `_data_/` for the demo. 

### Data processing

The data processing pipeline is made of four python scripts in series, contained in `scripts/data_processing/`. Each script is configurable via a `yaml` file in `scripts/data_processing/config/`.

1. Transform the raw EEG recordings into frequency-bands power

        python frequency_transform.py -c config/frequency_transform_config.yaml

2. Normalize the frequency-bands power per subject

        python normalization_transform.py -c config/normalization_transform_config.yaml 

3. Project the multi-channel electrodes signal into sequences of two-dimensional frames

        python image_transform.py -c config/image_transform_config.yaml 

4. Create an HDF5 dataset from the frames

        python hdf5_transform.py -c config/hdf5_transform_config.yaml 

The resulting HDF5 dataset is passed to instantiate a `EEG_Image_Batch_Dataset` (from `scripts.training.dataset`), which samples the data in batches for models training.

### Pre-training

With `scripts/training/representation/train_deep.py` you can train a deep network on processed EEG recordings. We provide an example with a Masked Autoencoder [[2]](#2), but other network architectures are available. You can try it with:
        
        python train_deep.py -c config/_config_example_.yaml

The results are saved in `results/_config_example_/`.


## References
<a id="1">[1]</a> 
Siclari Francesca, Baird Benjamin, Perogamvros Lampros et al. (2017). 
The neural correlates of dreaming.
Nature Neuroscience 20, 872–878.

<a id="2">[2]</a> 
Feichtenhofer Christoph, Fan Haoqi, Li Yanghao, He Kaiming (2022). 
Masked Autoencoders As Spatiotemporal Learners. 
