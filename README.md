# MultimodalSDK_loader

This repo aims to serve as a data-parser for the [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) package. It is a plug-and-play solution for feature extraction of CMU-MOSEI, CMU-MOSI and POM datasets. It's main functionality results in pickled tensors of user-defined sequence length, that contain feature vectors for train, validation and test sets of each dataset.<br/>

The code works in three steps:
1. download data
2. word-level alignment across modalities
3. label-level alignment across modalities

Once a step is completed, it caches the output files for future loading.

## Setup

- Clone repo with CMU Multimodal SDK submodule
```
# git version > 2.1.2
git clone --recurse-submodules https://github.com/lobracost/MultimodalSDK_loader.git

# git version < 2.1.2
git clone --recursive https://github.com/lobracost/MultimodalSDK_loader.git

```
- Install requirements
```
cd MultimodalSDK_loader
pip3 install -r requirements.txt 
```

## Usage

#### Command line

Run ```python3 process_dataset.py -n <dataset_name> -s <sequence_length>``` <br/>
where the supported dataset names are cmumosi, cmumosei and pom and the sequence_length variable defines the sequence length of the output tensors.<br/>
<br/>Example:<br/>
```python3 process_dataset.py -n cmumosi -s 20```

#### Python
```
from MultimodalSDK_loader.process_dataset import get_and_process_data

dataset_name = your_dataset_name # eg. "cmumosi"
seq_len = your_sequence_length # eg. 20
tensors = get_and_process_data(dataset_name, seq_len)
```

##### Get feature vectors
In order to get the corresponding feature vectors in a format that is capable for training and testing, run

```
from MultimodalSDK_loader.process_dataset import get_feature_matrix

dataset_name = your_dataset_name # eg. "cmumosi"
seq_len = your_sequence_length # eg. 20
X_train, y_train, X_val, y_val, X_test, y_test = get_feature_matrix(dataset_name, seq_len) 
```
