import yaml
import os
import logging
import pandas as pd

from edf_segmentor.EdfSegmentor import EdfSegmentor
from utils.utils import get_sha256, clear_dir
from file_storage.synology_api import SynologyAPI, FileListMode
from edf_segmentor.edf_split import edf_split, create_block_csvs, export_blocks, split_edf_into_subblocks
from edf_preproc.edf_preproc import EdfPreprocessor
from hdf5.make_h5 import HDF5Manager, process_subblocks

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

        # После того, как убедились, что с файлом всё хорошо - начинаем его обработку
        EDF_PREPROCESSOR.edf_preprocess(
            edf_file_path=f"{EDF_DOWNLOAD_DIR}/{filename}",
            output_folder="./temp/preprocessing/cleaned_edf",
        )

        segmentation_config = config["segmentation"]

        EDF_DIR = segmentation_config["edf_dir"]
        CLEANED_EDF = segmentation_config["cleaned_edf"]
        CSV_DIR = segmentation_config["csv_dir"]
        SEGMENTS_DIR = segmentation_config["segments_dir"]
        BLOCKS_DIR = segmentation_config["blocks_dir"]
        BLOCK_DURATION = segmentation_config["block_duration"]

        segmentor = EdfSegmentor(
            edf_dir=CLEANED_EDF,
            csv_dir=CSV_DIR,
            segments_dir=SEGMENTS_DIR,
            blocks_dir=BLOCKS_DIR,
            block_duration=BLOCK_DURATION
        )

        segmentor.split_edfs_from_edf_input_dir()

        # После успешной обработки удаляем исходный файл и промежуточные данные
        logger.info(f"Файл {filename} успешно разбит на сегменты...")
        # os.remove(f'{EDF_DOWNLOAD_DIR}/{filename}')
        # Очищаем output_csv
        # clear_dir(EDF_DIR)
        # clear_dir(CSV_DIR)
        # clear_dir(CLEANED_EDF)
        # Очищаем cleaned_edf
        logger.info(f"Исходный файл {EDF_DOWNLOAD_DIR}/{filename} и промежуточные данные были удалены удалены")

        # Загружаем содержимое папки с сегментами на сервер
        # res_path = f'temp/preprocessing/output_segments/{filename.replace('.edf', '')}'
        # synology_api.upload_folder(res_path, f'{OUTPUT_PATH}/{filename.replace('.edf', '')}', True, False)

        # Очищаем локальную папку с сегментами
        # clear_dir(res_path)

        logger.info(f"Сегменты загружены")


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
