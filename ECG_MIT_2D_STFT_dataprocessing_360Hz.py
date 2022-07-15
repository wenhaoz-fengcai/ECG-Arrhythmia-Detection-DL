import numpy as np
import h5py
import os
import librosa
import librosa.display
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import csv
import multiprocessing
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Handle data path
from pathlib import Path

# Read and display data from Physionet
import wfdb
import pprint
import collections
from IPython.display import clear_output

# Data manipulation and plotting
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.signal import butter,filtfilt

# Divide data into train and test set and save to HDF5
import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import collections
from tqdm import tqdm


def translate_beat_type(beat_type_arr, peak_loc_arr):
    n_type = ['N', 'L', 'R', 'e', 'j']
    s_type = ['A', 'a', 'J', 'S']
    v_type = ['V', 'E']
    f_type = ['F']
    q_type = ['/', 'f', 'Q']

    not_used = ['B', 'r', 'n', '?']

    encoded_label = []
    peak_loc_label = []
    ignored_label = []

    for beat_type, peak_loc in zip(beat_type_arr, peak_loc_arr):
        if beat_type in n_type:
            encoded_label.append(0)
        elif beat_type in s_type:
            encoded_label.append(1)
        elif beat_type in v_type:
            encoded_label.append(2)
        elif beat_type in f_type:
            encoded_label.append(3)
        elif beat_type in q_type:
            encoded_label.append(4)
        else:
            ignored_label.append(beat_type)
            continue
        peak_loc_label.append(peak_loc)

    encoded_label = np.array(encoded_label)
    peak_loc_label = np.array(peak_loc_label)

    return encoded_label, peak_loc_label, ignored_label


## Noted: This part of data is done incorrectly. The dataset should be split in train and test using records as follow instead of random split. The reason for this is from the paper:
## How the Choice of Samples for Building Arrhythmia Classifiers ImpactTheir Performances
## Train dataset record number: 101, 106, 108, 109, 112, 114, 115,116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215,220, 223, and 230,
## Test dataset record number: 100, 103, 105, 11, 113, 117, 121, 123, 200, 202, 210, 212, 213,214, 219, 221, 222, 228, 231, 232, and 234
## read data and annotation

# progress bar to check if the program running
def update_progress(progress):
    bar_length = 50
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def normalize(arr):
    arr_range = np.ptp(arr)
    arr_min = np.min(arr)
    norm_arr = (arr - arr_min) / float(arr_range)
    return norm_arr


def check_data_group(file):
    data_group = file['data']
    total = len(data_group.keys())
    index = 0
    total = []
    peaks = []
    for key in data_group.keys():
        temp_data = list(file['label/{}'.format(key)])
        total += temp_data
        temp_data = list(file['peak/{}'.format(key)])
        peaks += temp_data

    print(np.unique(total))
    print(collections.Counter(total))
    print(len(total) - collections.Counter(total)[5])
    print(len(peaks))
    print(len(total))

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def check_for_valid_beat(peak_location, label, max_length, threshold=180):
    if label == 5:
        return False
    if peak_location - threshold + 1 < 0:
        return False
    if peak_location + threshold + 1 > max_length:
        return False

    return True

def generate_stft_img(ecg_sample, file_name, n_fft=512, hop_length=1, pad_mode='wrap', window='hamming'):
    # cmap=LinearSegmentedColormap.from_list('gyr',["g", "y", "r"], N=256)

    my_dpi=300
    f = plt.figure(figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)

    plt.ioff()

    D = np.abs(librosa.stft(ecg_sample, n_fft=n_fft, win_length=n_fft, hop_length=1, pad_mode='constant', window='hamming'))

    db = librosa.amplitude_to_db(D,ref=np.max)
    plt.axis('off')
    # img = librosa.display.specshow(db, sr=360, hop_length=1, y_axis='log', x_axis='time', cmap=cmap)
    img = librosa.display.specshow(db, sr=360, hop_length=1, y_axis='log', x_axis='time', cmap='gray')

    plt.savefig(file_name)
    f.clf()
    plt.close(f)


def find_local_peak_location(beat_locs, ecg_sample, sample_freq):
    interval = 10 * sample_freq
    max_length = len(ecg_sample)
    correct_locs = []
    try:
        for i in range(len(beat_locs)):
            start_base_line = beat_locs[i] - interval / 2
            stop_base_line = beat_locs[i] + interval / 2

            # left end of data
            if start_base_line < 0:
                stop_base_line += abs(start_base_line)
                start_base_line = 0

            # right end of data
            if stop_base_line >= max_length:
                start_base_line -= abs(stop_base_line - max_length)
                stop_base_line = max_length

            # local mean base line
            base_line = np.mean(ecg_sample[int(start_base_line):int(stop_base_line)])

            local_sample = None

            # left end of data
            if beat_locs[i] - 2 < 0:
                local_sample = ecg_sample[beat_locs[i]: beat_locs[i] + 5]

            # right end of data
            elif beat_locs[i] + 3 > max_length:
                local_sample = ecg_sample[beat_locs[i] - 5: beat_locs[i]]

            # normal case
            else:
                local_sample = ecg_sample[beat_locs[i] - 2: beat_locs[i] + 3]

            diff = [abs(x - base_line) for x in local_sample]
            max_loc = np.argmax(diff)
            new_loc = beat_locs[i] - 2 + max_loc
            correct_locs.append(new_loc)
    except:
        ## error case print stuff
        print(beat_locs[i])
        print(local_sample)
        print(base_line)
        print(diff)
        print(max_loc)
    correct_locs = np.array(correct_locs)
    return correct_locs

if __name__ == '__main__':

    project_dir = Path("./")
    # main directory that store data
    data_dir = project_dir / 'data'
    os.makedirs(data_dir, exist_ok=True)

    img_dir = project_dir / 'img'
    os.makedirs(img_dir, exist_ok = True)

    # directory that store original MIT-BIH data

    mit_arrh_dir = data_dir / 'mit-bih-arrhythmia-database-1.0.0'
    #os.makedirs(mit_arrh_dir, exist_ok=True)

    # directory that store processed data
    process_dir = data_dir / 'processed_data'
    temp_dir = process_dir / 'temp'
    os.makedirs(process_dir, exist_ok=True)

    raw_data_dir = process_dir
    os.makedirs(raw_data_dir, exist_ok=True)

    file = open(mit_arrh_dir/'RECORDS')
    record_list = file.read().splitlines()
    file.close()
    print(record_list)

    with h5py.File(raw_data_dir / 'mit_raw_data.h5', 'w') as file:
        for i in range(len(record_list)):
            summary = wfdb.rdrecord(str(mit_arrh_dir / record_list[i])).__dict__

            ## only get the second signal which is ii - chest ECG
            anno = wfdb.rdann(str(mit_arrh_dir / record_list[i]), extension='atr', summarize_labels=True).__dict__
            peaks = anno['sample']
            labels = anno['symbol']
            encoded_labels, peak_labels, ignored_label = translate_beat_type(labels, peaks)

            file.create_dataset('data/{:03d}'.format(i), data=summary['p_signal'][:, 0])
            file.create_dataset('peak/{:03d}'.format(i), data=peak_labels)
            file.create_dataset('label/{:03d}'.format(i), data=encoded_labels)

            update_progress((i + 1) / float(len(record_list)))

    with h5py.File(raw_data_dir / 'mit_raw_data.h5', 'r') as file:
        check_data_group(file)

    filtered_ecg_data = []
    with h5py.File(raw_data_dir / 'mit_raw_data.h5', 'r') as file:
        ## only get the second signal which is ii - chest ECG
        data_group = file['data']
        total = len(data_group.keys())
        index = 0
        for key in data_group.keys():
            print(key)
            temp_data = file['data/{}'.format(key)]
            filter_data = butter_highpass_filter(temp_data, 0.25, 360, 2)
            filtered_ecg_data.append(filter_data)
            update_progress((index + 1) / float(total))
            index += 1

    ecg_data_norm = []
    for i in range(len(filtered_ecg_data)):
        signal = normalize(filtered_ecg_data[i])
        ecg_data_norm.append(signal)
        update_progress( (i + 1) / float(len(filtered_ecg_data)))

    ecg_correct_peak_location = []
    with h5py.File(raw_data_dir / 'mit_raw_data.h5', 'r') as file:
        ## only get the second signal which is ii - chest ECG
        data_group = file['peak']
        total = len(data_group.keys())
        index = 0
        for key in data_group.keys():
            temp_peak_data = file['peak/{}'.format(key)]
            corrected_loc = find_local_peak_location(temp_peak_data, ecg_data_norm[index], 360)
            ecg_correct_peak_location.append(corrected_loc)
            index += 1
            update_progress((index) / float(len(ecg_data_norm)))

    with h5py.File(raw_data_dir / 'mit_raw_data.h5', 'r') as raw_file:
        with h5py.File(process_dir / 'mit_processed_1st_stage_data.h5', 'w') as processed_file:
            ecg_label = []
            data_group = raw_file['label']
            total = len(data_group.keys())
            for key in data_group.keys():
                ecg_label.append(raw_file['label/{}'.format(key)][:])

            for i in range(len(ecg_label)):
                processed_file.create_dataset('data/{:03d}'.format(int(record_list[i])), data=ecg_data_norm[i])
                processed_file.create_dataset('peak/{:03d}'.format(int(record_list[i])),
                                              data=ecg_correct_peak_location[i])
                processed_file.create_dataset('label/{:03d}'.format(int(record_list[i])), data=ecg_label[i])

    with h5py.File(process_dir / 'mit_processed_1st_stage_data.h5', 'r') as processed_1st_file:
        with h5py.File(process_dir / 'mit_processed_2nd_stage_data.h5', 'w') as processed_2nd_file:
            data_group = processed_1st_file['peak']
            total = len(data_group.keys())
            for key in tqdm(data_group.keys(), total=total):
                ecg_peak = processed_1st_file['peak/{}'.format(key)][:]
                ecg_data = processed_1st_file['data/{}'.format(key)][:]
                ecg_label = processed_1st_file['label/{}'.format(key)][:]

                selected_peak = []
                selected_label = []
                selected_data = []

                for i in range(len(ecg_peak)):
                    if check_for_valid_beat(ecg_peak[i], ecg_label[i], len(ecg_data), threshold=432):
                        selected_peak.append(ecg_peak[i])
                        selected_label.append(ecg_label[i])
                        selected_data.append(ecg_data[ecg_peak[i] - 431: ecg_peak[i] + 433])

                #             print(len(selected_data))
                for i in range(len(selected_data)):
                    #                 print('data/{:03d}/{:05d}'.format(int(record_list[i]), i))
                    processed_2nd_file.create_dataset('data/{}/{:05d}'.format(key, i), data=selected_data[i])

                processed_2nd_file.create_dataset('peak/{}'.format(key), data=np.array(selected_peak))
                processed_2nd_file.create_dataset('label/{}'.format(key), data=np.array(selected_label))

    n_fft = 512
    hop_length = 1
    pad_mode = 'wrap'
    window = 'hamming'

    bw_dir = data_dir / ('2D_BW')
    img_dir = bw_dir / 'img'
    label_dir = bw_dir / 'label'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    with h5py.File(process_dir / 'mit_processed_2nd_stage_data.h5', 'r') as file:
        patient_group = file['data']
        total = len(patient_group.keys())
        index = 0
        for key in tqdm(patient_group.keys(), total=total):
            index += 1
            data_group = file['data/{}'.format(key)]
            file_dir = img_dir / key
            # print("Process image for patient " + key, index, "out of", total)
            os.makedirs(file_dir, exist_ok=True)
            pool = multiprocessing.Pool()

            ecg_data = []
            file_names = []
            for data_key in data_group.keys():
                ecg_data.append(file['data/{}/{}'.format(key, data_key)][:])
                file_names.append(file_dir / '{}.png'.format(data_key))
                # generate_stft_img(ecg_data, file_name, n_fft, hop_length, pad_mode, window)
                # pool.apply_async(generate_stft_img, args=(ecg_data, file_name, n_fft, hop_length, pad_mode, window,))

            for single_data, file_name in zip(ecg_data, file_names):
                # print("Assign task to pool for " + str(file_name))
                # generate_stft_img(single_data, file_name, n_fft, hop_length, pad_mode, window)
                pool.apply_async(generate_stft_img, args=(single_data, file_name, n_fft, hop_length, pad_mode, window,))
            pool.close()
            pool.join()

            # break
            ecg_label = file['label/{}'.format(key)][:]
            with open(label_dir / '{}.csv'.format(key), mode='w') as label_file:
                writer = csv.writer(label_file, delimiter=',')
                writer.writerow(ecg_label)