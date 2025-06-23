import re
import os
import csv
import mne
import logging

import warnings

warnings.filterwarnings(
    "ignore",
    message="Omitted [0-9]+ annotation\\(s\\) that were outside data range.",
    category=RuntimeWarning,
)

warnings.filterwarnings(
    "ignore",
    message="EDF format requires equal-length data blocks, so [0-9.]+ seconds of edge values were appended to all channels when writing the final block.",
    category=RuntimeWarning,
)

logger = logging.getLogger(__name__)


class EdfSegmentor:
    MIN_BLOCK_DURATION = 5.0

    TRANSLATIONS = {
        "Фоновая запись": "Baseline",
        "Открывание глаз": "EyesOpen",
        "Закрывание глаз": "EyesClosed",
        "Без стимуляции": "AfterStim",
        "Фотостимуляция": "PhoticStim",
        "После фотостимуляции": "PostPhotic",
        "Встроенный фотостимулятор": "Photic",
        "Встроенный слуховой стимулятор": "Auditory",
        "Остановка стимуляции": "AfterStim",
        "Гипервентиляция": "Hypervent",
        "После гипервентиляции": "PostHypervent",
        "Бодрствование": "Awake",
    }

    EXCLUDED_NAMES = {
        "stimFlash",
        "stimAudio",
        "Артефакт",
        "Начало печати",
        "Окончание печати",
        "Эпилептиформная активность",
        '''Комплекс "острая волна - медленная волна"''',
        "Множественные спайки и острые волны",
        "Разрыв записи",
    }

    def seconds_to_min_sec_ms(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{minutes:02}:{secs:02}.{milliseconds:03}"

    def time_str_to_seconds(self, time_str):
        minutes, rest = time_str.split(":")
        seconds, milliseconds = rest.split(".")
        return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

    def clean_event_name(self, name: str):
        """
        Очистка имен событий.
        """
        cleaned_name = re.sub(r"\[.*?\]|\(.*?\)", "", name).strip()
        if cleaned_name in EdfSegmentor.EXCLUDED_NAMES or name in EdfSegmentor.EXCLUDED_NAMES:
            return None

        for ru_name, en_name in EdfSegmentor.TRANSLATIONS.items():
            if ru_name in cleaned_name:
                return en_name

        return cleaned_name

    def create_segments_csvs(self, edfs_input_dir, csv_output_dir):
        os.makedirs(csv_output_dir, exist_ok=True)

        for filename in os.listdir(edfs_input_dir):
            if not filename.endswith(".edf"):
                continue

            self.create_segment_csv(edfs_input_dir, filename, csv_output_dir)

    def create_segment_csv(self, input_dir, filename, output_csv_dir):
        edf_file = os.path.join(input_dir, filename)
        raw = mne.io.read_raw_edf(edf_file, preload=True)

        annotations = raw.annotations
        onsets = annotations.onset
        descriptions = annotations.description
        # total_duration = raw.times[-1]
        total_duration = raw.n_times / raw.info["sfreq"]

        blocks = []
        i = 0
        while i < len(descriptions):
            desc = descriptions[i]
            clean_desc = self.clean_event_name(desc)
            if clean_desc is None:
                i += 1
                continue

            start = onsets[i]
            j = i + 1
            while j < len(descriptions):
                next_desc = descriptions[j]
                clean_next_desc = self.clean_event_name(next_desc)
                if clean_next_desc is None:
                    j += 1
                else:
                    break

            end = onsets[j] if j < len(onsets) else total_duration
            duration = end - start

            blocks.append((clean_desc, start, duration))
            i = j

        base_filename = os.path.splitext(filename)[0]
        csv_path = f"{output_csv_dir}/{base_filename}_segments.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Название", "Время начала", "Длительность"])
            for name, start, duration in blocks:
                writer.writerow(
                    [
                        name,
                        self.seconds_to_min_sec_ms(start),
                        self.seconds_to_min_sec_ms(duration),
                    ]
                )

        logger.info(f"CSV создан: {csv_path}")
        return csv_path

    def split_edfs_to_segments(self, edfs_input_dir, csv_output_dir, segments_output_dir):
        os.makedirs(segments_output_dir, exist_ok=True)

        for filename in os.listdir(edfs_input_dir):
            if not filename.endswith(".edf"):
                continue

            edf_file = os.path.join(edfs_input_dir, filename)
            base_filename = os.path.splitext(filename)[0]
            csv_file = os.path.join(csv_output_dir, f"{base_filename}_segments.csv")
            if not os.path.exists(csv_file):
                logger.info(f"Пропущен (нет CSV): {filename}")
                continue

            if self.is_csv_empty(csv_file):
                logger.info(f"Пропущен (пустой CSV): {csv_file}")
                continue

            self.split_edf_to_segments(csv_file, edf_file, segments_output_dir, base_filename)

    def is_csv_empty(self, csv_file):
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Проверка содержимого csv файла
            rows = list(reader)
            if not rows:
                return True
            return False

    def split_edf_to_segments(self, csv_file, edf_file, output_dir, base_filename):
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            raw = mne.io.read_raw_edf(edf_file, preload=True)

            file_output_dir = f'{output_dir}/{base_filename}'
            os.makedirs(file_output_dir, exist_ok=True)

            for i, row in enumerate(rows):
                block_name = row["Название"]
                start_sec = self.time_str_to_seconds(row["Время начала"])
                duration_sec = self.time_str_to_seconds(row["Длительность"])
                end_sec = min(start_sec + duration_sec, raw.times[-1])

                raw_block = raw.copy().crop(tmin=start_sec, tmax=end_sec)
                safe_name = (
                    block_name.replace(" ", "_").replace("(", "").replace(")", "")
                )
                # out_path = os.path.join(
                #     file_output_dir, f"{i + 1:02d}_{safe_name}_{base_filename}.edf"
                # )
                out_path = f'{file_output_dir}/{i + 1:02d}_{safe_name}_{base_filename}.edf'

                raw_block.export(
                    out_path, fmt="edf", physical_range="auto", overwrite=True
                )
                logger.info(f"Сохранён блок: {out_path}")

            return file_output_dir

    def split_segments_to_blocks(self, segments_input_dir: str, blocks_output_dir: str, block_duration: float):
        """
        Разделяет EDF-файлы на подблоки фиксированной длины. Последний подблок отбрасывается, если он короче block_duration.

        :param segments_input_dir: Директория, содержащая папки с EDF-файлами.
        :param blocks_output_dir: Директория для сохранения подблоков.
        :param block_duration: Длительность каждого подблока (в секундах).
        """

        block_duration = block_duration - 0.002

        # Создаем выходные директории
        os.makedirs(blocks_output_dir, exist_ok=True)

        # Перебираем все папки в input_dir
        for folder_name in os.listdir(segments_input_dir):
            folder_path = os.path.join(segments_input_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            logger.info(f"Обработка папки: {folder_name}")
            self.split_segment_to_blocks(blocks_output_dir, folder_name, folder_path, block_duration)

    def split_segment_to_blocks(self, output_dir, folder_name, folder_path, block_duration):
        blocks_folder = f'{output_dir}/{folder_name}'

        # Создает выходную директорию для подблоков
        os.makedirs(blocks_folder, exist_ok=True)

        # Перебираем все .edf файлы в папке.
        for filename in os.listdir(folder_path):
            if not filename.endswith(".edf"):
                continue

            edf_file = os.path.join(folder_path, filename)

            # Загружаем данные из EDF файла
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True)
            except Exception as e:
                logger.info(f"Ошибка при загрузке файла {filename}: {e}")
                continue

            # Общая длительность записи
            total_duration = raw.n_times / raw.info["sfreq"]

            # Имя базового файла без расширения
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            logger.info(
                f"Обработка файла: {filename} (длительность: {total_duration:.2f} сек)"
            )

            # Разделяем данные на подблоки
            current_time = 0
            block_index = 1

            while current_time + block_duration <= total_duration:
                next_time = current_time + block_duration

                try:
                    sub_block = raw.copy().crop(tmin=current_time, tmax=next_time)
                except ValueError as e:
                    logger.info(f"Ошибка при обрезке блока: {e}")
                    break

                # Генерируем имя файла для блока
                out_path = f"{blocks_folder}/{base_filename}_block_{block_index:02d}.edf"

                # Сохраняем подблок
                try:
                    sub_block.export(
                        out_path, fmt="edf", physical_range="auto", overwrite=True
                    )
                    logger.info(f"Сохранён блок: {out_path}")
                except Exception as e:
                    logger.info(f"Ошибка при экспорте блока: {e}")
                    break

                current_time = next_time
                block_index += 1

        return blocks_folder
