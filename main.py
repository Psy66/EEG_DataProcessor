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


def process_edf(config):
    # 1. Получи "список целей"
    # 2. Скачай файл-цель из списка
    # 3. Примени к нему фильтры
    # 4. Залей в облако
    # 5. Повтори, пока не обработаешь всё
    # 6. Удалить:
    #   1. исходный edf
    #   2. очищенный edf
    #   3. папку с сегментами (после загрузки)
    #   4. папку с блоками (после их внесения в h5 файл)


    EDF_PREPROCESSOR = EdfPreprocessor.from_config(config)

    EDF_DOWNLOAD_DIR = "temp/download_upload/edf_input"
    EDF_PROCESSED_DIR = "temp/download_upload/edf_output"

    storage_config = config["storage"]

    OVERWRITE_DOWNLOADS = storage_config["overwrite_downloads"]
    OVERWRITE_UPLOADS = storage_config["overwrite_uploads"]
    CREATE_REMOTE_PARENTS = storage_config["create_remote_parents"]

    OUTPUT_PATH = storage_config["output_path"]

    SYNOLOGY_BASE_URL = f"{storage_config['protocol']}://{storage_config['host']}:{storage_config['port']}"
    SYNOLOGY_LOGIN = storage_config["username"]
    SYNOLOGY_PASSWORD = storage_config["password"]

    SID = storage_config["sid"] if "sid" in storage_config else None

    if SID:
        synology_api = SynologyAPI(base_url=SYNOLOGY_BASE_URL, sid=SID)
    else:
        synology_api = SynologyAPI(
            base_url=SYNOLOGY_BASE_URL,
            username=SYNOLOGY_LOGIN,
            password=SYNOLOGY_PASSWORD,
        )
        synology_api.auth()

    targets = pd.read_csv("./info_data/prepared/studies.csv")

    # В целях тестирования - запускаем только на 5 первых файлах
    OPS_COUNT = 0
    OPS_LINIT = 1

    for index, target in targets.iterrows():
        if OPS_COUNT == OPS_LINIT:
            break
        OPS_COUNT += 1

        # Получаем путь до файла в облачном хранилище
        remote_filepath = target["file_path"]
        filename = target["file_name"]
        base_filename = filename.replace('.edf', '')

        if os.path.exists(f"{EDF_DOWNLOAD_DIR}/{filename}") and not OVERWRITE_DOWNLOADS:
            logger.info(
                f"Файл {filename} уже существует в директории {EDF_DOWNLOAD_DIR}, отмена загрузки"
            )
        else:
            # Качаем его
            local_filepath = synology_api.download_file(
                remote_filepath=remote_filepath,
                output_dir=EDF_DOWNLOAD_DIR,
                overwrite=OVERWRITE_DOWNLOADS,
            )

            # Проверяем SHA-256
            # Если не совпадает - удаляем загруженный битый файл и идём дальше
            remote_file_sha256 = target["file_checksum"]
            local_file_sha256 = get_sha256(local_filepath)

            if remote_file_sha256 != local_file_sha256:
                logger.warning(
                    f"Целостность файла {filename}, нарушена. Начинаю удаление..."
                )
                os.remove(local_filepath)
                logger.warning(f"Битый файл {EDF_DOWNLOAD_DIR}/{filename} удалён")
            else:
                logger.info(f"Целостность файла {filename} подтверждена\n")

        segmentation_config = config["segmentation"]

        EDF_DIR = segmentation_config["edf_dir"]
        CLEANED_EDF = segmentation_config["cleaned_edf"]
        CSV_DIR = segmentation_config["csv_dir"]
        SEGMENTS_DIR = segmentation_config["segments_dir"]
        BLOCKS_DIR = segmentation_config["blocks_dir"]
        BLOCK_DURATION = segmentation_config["block_duration"]

        # После того, как убедились, что с файлом всё хорошо - начинаем его обработку
        EDF_PREPROCESSOR.edf_preprocess(
            edf_file_path=f"{EDF_DOWNLOAD_DIR}/{filename}",
            output_folder=CLEANED_EDF,
        )

        segmentor = EdfSegmentor(
            edf_input_dir=EDF_DIR,
            csv_output_dir=CSV_DIR,
            segments_dir=SEGMENTS_DIR,
            blocks_output_dir=BLOCKS_DIR,
            block_duration=BLOCK_DURATION
        )

        # Путь к файлу с информацией о сегментах .edf файла
        segments_csv_path = segmentor.create_segment_csv(EDF_DOWNLOAD_DIR, filename, CSV_DIR)

        # Путь к папке с сегментами
        segments_dir_path = segmentor.split_edf_to_segments(f'{CSV_DIR}/{base_filename}_segments.csv',
                                                            f'{CLEANED_EDF}/{filename}', SEGMENTS_DIR,
                                                            base_filename)

        # Путь к папке с блоками
        blocks_dir_path = segmentor.split_segment_to_blocks(BLOCKS_DIR, base_filename,
                                                            f'{SEGMENTS_DIR}/{base_filename}', BLOCK_DURATION)

def prepare_dataset(config):
    hdf5_manager = HDF5Manager(base_dir="temp/download_upload/hdf5_output")

    process_subblocks(
        subblocks_dir="temp/preprocessing/output_blocks",
        mapping_csv="info_data/original/mapping.csv",
        hdf5_manager=hdf5_manager,
    )


def main(action):
    config = get_config()

    if action == "process_edf":
        process_edf(config)
    elif action == "prepare_dataset":
        prepare_dataset(config)
    else:
        print("Несуществующая функция")


if __name__ == "__main__":
    main("process_edf")
    # main("prepare_dataset")
