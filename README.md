# Instructions
## Data processing code

 1. [ECG_MIT_1D_dataprocessing_125Hz](ECG_MIT_1D_dataprocessing_125Hz.ipynb): Process data into 1D 125 Hz train/test (for both inter and intra patient paradigm for [Kachuee paper](https://doi.org/10.1109/ICHI.2018.00092) replication
 2. [ECG_MIT_1D_dataprocessing_360Hz](ECG_MIT_1D_dataprocessing_360Hz.ipynb): Process data into 1D 360 Hz train/test (for both inter and intra patient paradigm for [Romdhane paper](https://doi.org/10.1016/j.compbiomed.2020.103866) replication
 3. [ECG_MIT_2D_STFT_dataprocessing_360Hz](ECG_MIT_2D_STFT_dataprocessing_360Hz.py): Process data into 2D STFT images and label for our proposed method

## Model train/test code

 1. [Kachuee_paper_replication](Kachuee_paper_replication.ipynb): Train and test [Kachuee model](https://doi.org/10.1109/ICHI.2018.00092) for both inter and intra patient paradigm
 2. [Romdhane_paper_replication](Romdhane_paper_replication.ipynb): Train and test [Romdhane model](https://doi.org/10.1016/j.compbiomed.2020.103866) for both inter and intra patient paradigm
 3. [STFT_Resnet](STFT_Resnet.ipynb): Train and test our proposed model for inter patient paradigm

## References
 1. Kachuee paper:  Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018, June). Ecg heartbeat classification: A deep transferable representation. In 2018 IEEE international conference on healthcare informatics (ICHI) (pp. 443-444). IEEE.
 2. Romdhane paper: Romdhane, T. F., & Pr, M. A. (2020). Electrocardiogram heartbeat classification based on a deep convolutional neural network and focal loss. _Computers in Biology and Medicine_, _123_, 103866.