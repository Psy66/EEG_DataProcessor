import mne
import numpy as np


def min_max_normalisation(raw):
    original_annotations = raw.annotations.copy()

    ecg_channel = None
    if 'ECG  ECG' in raw.ch_names:
        ecg_channel = raw.copy().pick_channels(['ECG  ECG'])
        raw = raw.drop_channels(['ECG  ECG'])

    data_by_channels = raw.get_data()
    normalized_data = []

    for data_by_channel in data_by_channels:
        min_val = data_by_channel.min()
        max_val = data_by_channel.max()
        if max_val - min_val == 0:
            raise ZeroDivisionError('Диапазон значений канала равен нулю, нормализация невозможна.')
        normalized_data_by_channel = (data_by_channel - min_val) / (max_val - min_val)
        normalized_data.append(normalized_data_by_channel)
    normalized_data = np.array(normalized_data)

    info = raw.info.copy()
    for ch in info['chs']:
        ch['kind'] = mne.io.constants.FIFF.FIFFV_MISC_CH
        ch['unit'] = mne.io.constants.FIFF.FIFF_UNIT_NONE

    normalized_raw = mne.io.RawArray(normalized_data, info)

    if ecg_channel is not None:
        normalized_raw.add_channels([ecg_channel])

    normalized_raw.set_annotations(original_annotations)

    return normalized_raw
