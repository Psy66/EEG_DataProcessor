import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from mne.io import read_raw_edf


class HDF5Manager:
    def __init__(self, base_dir: str):
        """
        Инициализирует менеджер HDF5 файлов.
        Args:
            base_dir (str): Базовая директория для хранения HDF5 файлов.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def add_subblocks_to_hdf5(
        self, diagnosis, block_type, patient_id, data, metadata, global_attributes
    ):
        """
        Добавляет подблоки в HDF5 файл.

        :param diagnosis: Диагноз пациента (например, "F32").
        :param block_type: Тип блока (например, "CLOSED_EYES").
        :param patient_id: Идентификатор пациента (например, "085574").
        :param data: Данные подблоков (numpy array, shape=(n_blocks, n_channels, n_samples_per_block)).
        :param metadata: Метаданные пациента (словарь с ключами: gender, age_cat, source_file).
        :param global_attributes: Глобальные атрибуты файла (диагноз, метка блока, частота дискретизации и т.д.).
        """

        # Формируем имя файла
        filename = f"{diagnosis}_{block_type}.h5"
        file_path = os.path.join(self.base_dir, filename)

        # Открываем файл в режиме append
        with h5py.File(file_path, "a") as hdf_file:
            if patient_id not in hdf_file:
                patient_group = hdf_file.create_group(patient_id)
                patient_group.create_dataset("data", data=data, dtype="float32")

                metadata_group = patient_group.create_group("metadata")
                metadata_group.attrs["gender"] = metadata["gender"]
                metadata_group.attrs["age_cat"] = metadata["age_cat"]
                metadata_group.create_dataset(
                    "source_files",
                    data=np.array([metadata["source_file"]]),
                    maxshape=(None,),
                    dtype="S100",
                )
            else:
                patient_group = hdf_file[patient_id]

                # Обновляем data
                existing_data = patient_group["data"]
                new_data = np.concatenate([existing_data[:], data], axis=0)
                del patient_group["data"]
                patient_group.create_dataset("data", data=new_data, dtype="float32")

                # Обновляем source_files
                metadata_group = patient_group["metadata"]
                source_ds = metadata_group["source_files"]
                existing_sources = list(source_ds[:])
                if metadata["source_file"] not in existing_sources:
                    updated_sources = existing_sources + [metadata["source_file"]]
                    del metadata_group["source_files"]
                    metadata_group.create_dataset(
                        "source_files",
                        data=np.array(updated_sources),
                        dtype="S100",
                        maxshape=(None,),
                    )

            # Обновляем глобальные атрибуты
            for key, value in global_attributes.items():
                hdf_file.attrs[key] = value


def get_gender_from_raw(raw):
    if "subject_info" in raw.info and raw.info["subject_info"]:
        sex_code = raw.info["subject_info"].get("sex")
        if sex_code == 1:
            return "M"
        elif sex_code == 2:
            return "F"
        else:
            return "N"
    return "N"


def calculate_age_cat(raw):
    """
    Рассчитывает возраст и возрастную категорию на основе meas_date и birthday.

    :param meas_date: Дата записи в формате "YYYY-MM-DD HH:MM:SS UTC".
    :param birthday: Дата рождения в формате "YYYY-MM-DD".
    :return: Кортеж (возраст, возрастная категория).
    """
    # Возможные возрастные категории
    age_categories = [
        (0, 3, "0-3 years"),
        (3, 6, "3-6 years"),
        (6, 9, "6-9 years"),
        (9, 12, "9-12 years"),
        (12, 14, "12-14 years"),
        (14, 18, "14-18 years"),
        (18, 25, "18-25 years"),
        (25, 30, "25-30 years"),
        (30, 40, "30-40 years"),
        (40, float("inf"), "40+ years"),  # Все, что выше 40
    ]

    meas_date = raw.info["meas_date"]
    birthday = raw.info["subject_info"].get("birthday")

    try:
        meas_date = meas_date.date()

        age = meas_date.year - birthday.year

        if (meas_date.month, meas_date.day) < (birthday.month, birthday.day):
            age -= 1

        for start, end, category in age_categories:
            if start <= age < end:
                return int(age), category
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        return None, "Unknown"

    return category


def process_subblocks(subblocks_dir: str, mapping_csv: str, hdf5_manager: HDF5Manager):
    """
    Обрабатывает все подблоки в указанной директории и сохраняет их в HDF5 файлы.

    :param subblocks_dir: Директория с подблоками.
    :param mapping_csv: Путь к CSV-файлу с диагнозами пациентов.
    :param hdf5_manager: Экземпляр HDF5Manager для работы с HDF5 файлами.
    """

    # Чтение mapping.csv
    mapping_df = pd.read_csv(mapping_csv)
    mapping_dict = dict(zip(mapping_df["New Name"], mapping_df["Diag_Code"]))

    # Обход папок в subblocks
    for patient_folder in os.listdir(subblocks_dir):
        patient_folder_path = os.path.join(subblocks_dir, patient_folder)
        if not os.path.isdir(patient_folder_path):
            continue

        # Получаем диагноз
        patient_id = patient_folder
        diagnosis = mapping_dict.get(patient_id + ".edf")
        if not diagnosis:
            print(f"Diagnosis not found for patient {patient_id}. Skipping...")
            continue

        # Обработка файлов в папке пациента
        for file_name in os.listdir(patient_folder_path):
            file_path = os.path.join(patient_folder_path, file_name)

            block_type = file_name.split("_")[1]

            # Загружаем данные из EDF файлов (подблоков)
            try:
                raw = read_raw_edf(file_path, preload=True)
                data = raw.get_data()
                data = np.expand_dims(data, axis=0)
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
                continue

            age, age_cat = calculate_age_cat(raw)

            # Формируем метаданные
            metadata = {
                "gender": get_gender_from_raw(raw).encode("utf-8"),
                "age_cat": age_cat.encode("utf-8") if age_cat else b"Unknown",
                "source_file": file_name.encode("utf-8"),
            }

            # Глобальные атрибуты
            global_attributes = {
                "diagnosis": diagnosis,
                "segment_label": block_type,
                "sampling_rate": raw.info["sfreq"],
                "n_channels": raw.info["nchan"],
                "channel_names": raw.info["ch_names"],
                "block_length": data.shape[-1]
                / raw.info["sfreq"],  # Длина блока в секундах
                "creation_date": datetime.now().isoformat(),
            }

            # Добавляем данные в HDF5 файл
            hdf5_manager.add_subblocks_to_hdf5(
                diagnosis=diagnosis,
                block_type=block_type,
                patient_id=patient_id,
                data=data,
                metadata=metadata,
                global_attributes=global_attributes,
            )

        print(f"Processed patient {patient_id} with diagnosis {diagnosis}")


if __name__ == "__main__":
    subblocks_dir = "temp/subblocks"
    mapping_csv = "mapping.csv"
    hdf5_base_dir = "hdf5"

    hdf5_manager = HDF5Manager(base_dir=hdf5_base_dir)

    process_subblocks(subblocks_dir, mapping_csv, hdf5_manager)
