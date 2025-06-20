import re
import os
import csv
import mne
import logging

logger = logging.getLogger(__name__)


class EdfSegmentor:
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

    def __init__(self, edf_dir: str, csv_dir: str, segments_dir: str, blocks_dir: str,
                 block_duration: float):
        """
        Инициализирует экземпляр класса EdfSegmentor для поэтапного разбиения EDF-файлов
        на сегменты (блоки по аннотациям) и подблоки фиксированной длины.

        Args:
            edf_dir (str): Путь к директории с исходными EDF-файлами.
            csv_dir (str): Путь к директории для сохранения CSV-файлов с конфигурацией блоков (по аннотациям).
            segments_dir (str): Путь к директории для сохранения экспортированных сегментов.
            blocks_dir (str): Путь к директории для сохранения блоков фиксированной длины.
            block_duration (str): Продолжительность длины блока (соотв. минимальная длина сегмента).
        """
        self._edf_dir = edf_dir
        self._csv_dir = csv_dir
        self._segments_dir = segments_dir
        self._blocks_dir = blocks_dir
        self._block_duration = block_duration

    # По сути - главный функционал класса
    def split_edfs_from_edf_input_dir(self):
        # Вызов функции для создания CSV с сегментами по аннотациям
        self.create_segments_csvs(edf_input_dir=self._edf_dir, csv_output_dir=self._csv_dir)

        # Вызов функции разделения файлов на сегменты согласно CSV файлу с сегментами
        self.export_segments(edf_input_dir=self._edf_dir, csv_input_dir=self._csv_dir,
                             segments_output_dir=self._segments_dir)

        # Вызов функции для разбиения сегментов на блоки
        self.split_edf_into_blocks(segments_input_dir=self._segments_dir, blocks_output_dir=self._blocks_dir,
                                   block_duration=self._block_duration)

    def time_str_to_seconds(self, time_str: str):
        minutes, rest = time_str.split(":")
        seconds, milliseconds = rest.split(".")
        return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

    def clean_event_name(self, name: str):
        cleaned_name = re.sub(r"\[.*?\]|\(.*?\)", "", name).strip()
        if cleaned_name in EdfSegmentor.EXCLUDED_NAMES or name in EdfSegmentor.EXCLUDED_NAMES:
            return None

        for ru_name, en_name in EdfSegmentor.TRANSLATIONS.items():
            if ru_name in cleaned_name:
                return en_name

        return cleaned_name

    def seconds_to_min_sec_ms(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{minutes:02}:{secs:02}.{milliseconds:03}"

    def create_segments_csvs(self, edf_input_dir, csv_output_dir):
        os.makedirs(self._csv_dir, exist_ok=True)

        for filename in os.listdir(self._edf_dir):
            if not filename.endswith(".edf"):
                continue

            edf_file = os.path.join(self._edf_dir, filename)
            raw = mne.io.read_raw_edf(edf_file, preload=True)

            annotations = raw.annotations
            onsets = annotations.onset
            descriptions = annotations.description
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
            csv_path = os.path.join(self._csv_dir, f"{base_filename}_blocks.csv")

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

    def export_segments(self, edf_input_dir, csv_input_dir, segments_output_dir):
        os.makedirs(segments_output_dir, exist_ok=True)

        for filename in os.listdir(edf_input_dir):
            if not filename.endswith(".edf"):
                continue

            edf_file = os.path.join(edf_input_dir, filename)
            raw = mne.io.read_raw_edf(edf_file, preload=True)
            base_filename = os.path.splitext(filename)[0]
            csv_file = os.path.join(csv_input_dir, f"{base_filename}_blocks.csv")

            if not os.path.exists(csv_file):
                print(f"Пропущен (нет CSV): {filename}")
                continue

            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Проверка содержимого csv файла
                rows = list(reader)
                if not rows:
                    print(f"Пропущен (пустой CSV): {csv_file}")
                    continue

                file_output_path = os.path.join(segments_output_dir, base_filename)
                os.makedirs(file_output_path, exist_ok=True)

                for i, row in enumerate(rows):
                    block_name = row["Название"]
                    start_sec = self.time_str_to_seconds(row["Время начала"])
                    duration_sec = self.time_str_to_seconds(row["Длительность"])
                    end_sec = min(start_sec + duration_sec, raw.times[-1])

                    raw_block = raw.copy().crop(tmin=start_sec, tmax=end_sec)
                    safe_name = (
                        block_name.replace(" ", "_").replace("(", "").replace(")", "")
                    )
                    out_path = os.path.join(
                        file_output_path, f"{i + 1:02d}_{safe_name}_{base_filename}.edf"
                    )

                    raw_block.export(
                        out_path, fmt="edf", physical_range="auto", overwrite=True
                    )
                    print(f"Сохранён блок: {out_path}")

    def split_edf_into_blocks(self, segments_input_dir, blocks_output_dir, block_duration):

        block_duration = block_duration - 0.002

        # Создаем выходные директории
        os.makedirs(blocks_output_dir, exist_ok=True)

        # Перебираем все папки в input_dir
        for folder_name in os.listdir(segments_input_dir):
            folder_path = os.path.join(segments_input_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            logger.info(f"Обработка папки: {folder_name}")

            # Создает выходную директорию для подблоков
            subblocks_folder = os.path.join(blocks_output_dir, folder_name)
            os.makedirs(subblocks_folder, exist_ok=True)

            # Перебираем все .edf файлы в папке.
            for filename in os.listdir(folder_path):
                if not filename.endswith(".edf"):
                    continue

                edf_file = os.path.join(folder_path, filename)

                # Загружаем данные из EDF файла
                try:
                    raw = mne.io.read_raw_edf(edf_file, preload=True)
                except Exception as e:
                    print(f"Ошибка при загрузке файла {filename}: {e}")
                    continue

                # Общая длительность записи
                total_duration = raw.n_times / raw.info["sfreq"]

                # Имя базового файла без расширения
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                print(
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
                        logger.error(f"Ошибка при обрезке блока: {e}")
                        break

                    # Генерируем имя файла для подблока
                    out_path = os.path.join(
                        subblocks_folder,
                        f"{base_filename}_subblock_{block_index:02d}.edf",
                    )

                    # Сохраняем подблок
                    try:
                        sub_block.export(
                            out_path, fmt="edf", physical_range="auto", overwrite=True
                        )
                        logger.info(f"Сохранён подблок: {out_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при экспорте подблока: {e}")
                        break

                    current_time = next_time
                    block_index += 1
