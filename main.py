import yaml
import os
import logging
import pandas as pd
import shutil

from edf_segmentor.EdfSegmentor import EdfSegmentor
from utils.utils import get_sha256, clear_dir
from file_storage.synology_api import SynologyAPI, FileListMode
from edf_segmentor.edf_split import edf_split, create_block_csvs, export_blocks, split_edf_into_subblocks
from edf_preproc.edf_preproc import EdfPreprocessor
from hdf5.make_h5 import HDF5Manager, process_subblocks

import mne

mne.set_log_level('ERROR')

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG_PATH = "./config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_and_validate_target_file(api, target, download_dir, overwrite):
    remote_path = target["file_path"]
    filename = target["file_name"]
    expected_sha256 = target["file_checksum"]
    local_path = f'{download_dir}/{filename}'

    if os.path.exists(local_path) and not overwrite:
        logger.info(f"Файл {filename} уже существует, пропускаем загрузку.")
        return local_path, True

    local_path = api.download_file(remote_path, download_dir, overwrite=overwrite)
    actual_sha256 = get_sha256(local_path)

    if actual_sha256 != expected_sha256:
        logger.warning(f"Контрольная сумма не совпадает: {filename}. Удаление.")
        os.remove(local_path)
        return None, False

    logger.info(f"Файл {filename} успешно загружен и проверен.")
    return local_path, True


def process_single_target(config, target, synology_api, edf_preprocessor):
    filename = target["file_name"]
    base_filename = filename.replace('.edf', '')

    edf_download_dir = config['segmentation']['edf_dir']
    overwrite_on_download = config['storage']['overwrite_downloads']

    target_edf_local_path, is_edf_valid = download_and_validate_target_file(synology_api, target, edf_download_dir,
                                                                            overwrite_on_download)

    if not is_edf_valid:
        return

    seg_config = config["segmentation"]

    edf_preprocessor.edf_preprocess(target_edf_local_path, seg_config['cleaned_edf'])

    segmentor = EdfSegmentor()

    segments_csv_path = segmentor.create_segment_csv(edf_download_dir, filename, seg_config["csv_dir"])
    segments_dir_path = segmentor.split_edf_to_segments(segments_csv_path,
                                                        f'{seg_config['cleaned_edf']}/{filename}',
                                                        seg_config["segments_dir"],
                                                        base_filename)
    blocks_dir_path = segmentor.split_segment_to_blocks(seg_config["blocks_dir"], base_filename,
                                                        segments_dir_path, seg_config["block_duration"])


def process_edfs(config):
    edf_preprocessor = EdfPreprocessor.from_config(config['processing'])
    synology_api = SynologyAPI.from_config(config['storage'])
    targets = pd.read_csv(config['targets']['targets_csv'])

    ops_limit = 1  # Временно ограничим для отладки
    ops_count = 0
    for i, target in targets.iterrows():
        if ops_count == ops_limit:
            break
        ops_count += 1

        process_single_target(config, target, synology_api, edf_preprocessor)


def prepare_dataset(config):
    hdf5_manager = HDF5Manager(base_dir="temp/download_upload/hdf5_output")

    process_subblocks(
        subblocks_dir="temp/preprocessing/output_blocks",
        mapping_csv="info_data/original/mapping.csv",
        hdf5_manager=hdf5_manager,
    )


def main(action):
    config = get_config()

    if action == "process_edfs":
        process_edfs(config)
    elif action == "prepare_dataset":
        prepare_dataset(config)
    else:
        print("Несуществующая функция")


if __name__ == "__main__":
    main("process_edfs")
    # main("prepare_dataset")
