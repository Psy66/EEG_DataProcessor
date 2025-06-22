import yaml
import os
import logging
from datetime import datetime
import pandas as pd
import shutil

from edf_segmentor.EdfSegmentor import EdfSegmentor
from utils.utils import get_sha256, clear_dir
from file_storage.synology_api import SynologyAPI, FileListMode
from edf_segmentor.edf_split import edf_split, create_block_csvs, export_blocks, split_edf_into_subblocks
from edf_preproc.edf_preproc import EdfPreprocessor
from hdf5.make_h5 import HDF5Manager, process_subblocks

import mne
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning)

config_path = "./config.yaml"


def get_config():
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger():
    config = get_config()
    logs_dir = config['logging']['logs_dir']

    log_filename = datetime.now().strftime(f"{logs_dir}/log_%Y-%m-%d__%H-%M-%S.log")

    # Настройка логгера
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


logger = setup_logger()


def download_and_validate_target_file(api, target, download_dir, overwrite):
    remote_path = target["file_path"]
    filename = target["file_name"]
    expected_sha256 = target["file_checksum"]
    local_path = f'{download_dir}/{filename}'

    if os.path.exists(local_path) and not overwrite:
        logger.info(f"Файл {filename} уже существует, пропускаем загрузку.")
        return local_path, True

    try:
        local_path = api.download_file(remote_path, download_dir, overwrite=overwrite)
    except FileNotFoundError as e:
        logger.info(e)
        return None, False

    actual_sha256 = get_sha256(local_path)

    if actual_sha256 != expected_sha256:
        logger.warning(f"Контрольная сумма не совпадает: {filename}. Удаление.")
        os.remove(local_path)
        return None, False

    logger.info(f"Файл {filename} успешно проверен на целостность.")
    return local_path, True


def process_single_target(config, target, synology_api, edf_preprocessor):
    filename = target["file_name"]
    base_filename = filename.replace('.edf', '')

    seg_config = config["segmentation"]
    storage_config = config["storage"]

    edf_download_dir = seg_config['edf_dir']
    overwrite_on_download = storage_config['overwrite_downloads']

    target_edf_local_path, is_edf_valid = download_and_validate_target_file(synology_api, target, edf_download_dir,
                                                                            overwrite_on_download)

    if not is_edf_valid:
        return

    logger.info(f'Начинаем обработку файла {filename}')
    cleaned_edf_path = edf_preprocessor.edf_preprocess(target_edf_local_path, seg_config['cleaned_edf'])
    logger.info(f'Обработка файла {filename} завершена')

    logger.info(f'Начинаем сегментацию файла {filename}')
    segmentor = EdfSegmentor()

    segments_csv_path = segmentor.create_segment_csv(edf_download_dir, filename, seg_config["csv_dir"])
    segments_dir_path = segmentor.split_edf_to_segments(segments_csv_path,
                                                        f'{seg_config['cleaned_edf']}/{filename}',
                                                        seg_config["segments_dir"],
                                                        base_filename)
    logger.info(f'Сегментация файла {filename} завершена')

    storage_output_path = f'{storage_config['output_path']}/{base_filename}'
    logger.info(f'Загружаем сегменты {segments_dir_path} в {storage_output_path}')

    synology_api.upload_folder(segments_dir_path, storage_output_path, storage_config['create_remote_parents'],
                               storage_config['overwrite_uploads'])

    logger.info(f'Загрузка в {storage_output_path} завершена')

    logger.info(f'Очищаем служебные файлы')

    # Удаляем исходный edf файл
    # Удаляем cleaned_edf
    # Удаляем output_csv
    # Удаляем output_segments
    os.remove(target_edf_local_path)
    os.remove(cleaned_edf_path)
    os.remove(segments_csv_path)
    shutil.rmtree(segments_dir_path)

    logger.info(f'Служебные файлы очищены. Обработка {filename} завершена')


def process_edfs(config):
    edf_preprocessor = EdfPreprocessor.from_config(config['processing'])
    synology_api = SynologyAPI.from_config(config['storage'])
    targets = pd.read_csv(config['targets']['targets_csv'])

    targets_len = targets.shape[0]

    processed_files = synology_api.get_files_list(config['storage']['output_path'], FileListMode.DIR)
    unprocessed_files = [
        target for _, target in targets.iterrows()
        if f'{config["storage"]["output_path"]}/{target["file_name"].replace(".edf", "")}' not in processed_files
    ]

    processed_files_len = len(processed_files)
    unprocessed_files_len = len(unprocessed_files)

    logger.info(
        f"На данный момент обработано {processed_files_len} из {targets_len} файлов, осталось {unprocessed_files_len}"
    )

    ops_limit = 7  # Временно ограничим для отладки
    ops_count = 0
    processed_files_count = 0
    for i, target in targets.iterrows():
        if ops_count == ops_limit:
            break
        ops_count += 1

        storage_dir_path = f'{config['storage']['output_path']}/{target["file_name"].replace('.edf', '')}'
        if storage_dir_path in processed_files:
            logger.info(
                f'Файл {target["file_name"]} уже имеет директорию в хранилище {storage_dir_path}. Обработка пропущена.')
            continue

        process_single_target(config, target, synology_api, edf_preprocessor)
        processed_files_count += 1

        logger.info(
            f"Обработано {processed_files_len + processed_files_count} из {targets_len} файлов, осталось {unprocessed_files_len - processed_files_count}"
        )

    logger.info('Обработка и загрузка сегментов завершена!')


def prepare_dataset(config):
    hdf5_manager = HDF5Manager(base_dir="temp/download_upload/hdf5_output")

    process_subblocks(
        subblocks_dir="temp/preprocessing/output_blocks",
        mapping_csv="info_data/original/mapping.csv",
        hdf5_manager=hdf5_manager,
    )

    # segmentor = EdfSegmentor()
    # blocks_dir_path = segmentor.split_segment_to_blocks(seg_config["blocks_dir"], base_filename,
    #                                                     segments_dir_path, seg_config["block_duration"])


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
