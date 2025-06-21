import os

import mne

from edf_preproc.crop_raw_edf import crop_raw_edf
from edf_preproc.edf_filters import notch_filter, bandpass_filter, sigma_3_filter
from edf_preproc.ica import ica_filter
from edf_preproc.min_max_normalisation import min_max_normalisation
import logging

logger = logging.getLogger(__name__)

class EdfPreprocessor:
    def __init__(self, bandpass_filter=None, notch_filter=None, segment_min_duration=1.0, bad_channel_threshold=0.8):
        self.bandpass_filter = self._validate_filter(bandpass_filter, default=[0.5, 45], name='bandpass_filter')
        self.notch_filter_freqs = self._validate_filter(notch_filter, default=[50, 60], name='notch_filter')
        self.segment_min_duration = self._validate_positive_float(segment_min_duration, 'segment_min_duration')
        self.bad_channel_threshold = self._validate_fraction(bad_channel_threshold, 'bad_channel_threshold')

    @classmethod
    def from_config(cls, processing_config):
        return cls(
            bandpass_filter=processing_config['bandpass_filter'],
            notch_filter=processing_config['notch_filter'],
            segment_min_duration=processing_config['segment_min_duration'],
            bad_channel_threshold=processing_config['bad_channel_threshold']
        )

    @staticmethod
    def _validate_filter(value, default, name):
        if value is None:
            return default
        if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
            raise ValueError(f'{name} должен быть списком из двух чисел, например [низкая_частота, высокая_частота]')
        return value

    @staticmethod
    def _validate_positive_float(value, name):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'{name} должен быть положительным числом')
        return float(value)

    @staticmethod
    def _validate_fraction(value, name):
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError(f'{name} должен быть числом от 0 до 1')
        return float(value)

    def preprocess_raw(self, raw):
        # 1. Обрезаем концы
        cropped_raw = crop_raw_edf(raw, 5.0)
        # 2. Фильтры (notch, полосной, 3-sigma)
        notch_filtered_raw = notch_filter(cropped_raw, self.notch_filter_freqs)
        bandpass_filtered_raw = bandpass_filter(notch_filtered_raw, l_freq=self.bandpass_filter[0],
                                                h_freq=self.bandpass_filter[1])
        sigma_3_filtered = sigma_3_filter(bandpass_filtered_raw)

        # 3. ICA
        # ica_filtered_raw = ica_filter(sigma_3_filtered)

        # 4. MinMax нормализация
        # min_max_normalised_raw = min_max_normalisation(ica_filtered_raw)
        min_max_normalised_raw = min_max_normalisation(sigma_3_filtered)

        return min_max_normalised_raw

    def edf_preprocess(self, edf_file_path, output_folder='./temp/preprocessed_edf'):
        if not edf_file_path.lower().endswith('.edf'):
            raise ValueError('Файл должен иметь расширение .edf')

        if not os.path.exists(edf_file_path):
            raise FileNotFoundError(f'Файл не найден: {edf_file_path}')

        if not os.path.isdir(output_folder):
            raise FileNotFoundError(f'Папка для сохранения не существует: {output_folder}')

        raw = mne.io.read_raw_edf(edf_file_path, preload=True)
        preprocessed_raw = self.preprocess_raw(raw)

        output_path = os.path.join(output_folder, os.path.basename(edf_file_path))
        preprocessed_raw.export(output_path, fmt='edf', overwrite=False)
        logger.info(f'Файл {os.path.basename(edf_file_path)} предобработан и сохранён: {output_path}')
