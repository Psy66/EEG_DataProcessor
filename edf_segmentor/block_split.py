import re
import os
import csv
import mne

import warnings

warnings.filterwarnings(
    "ignore",
    message="Omitted [0-9]+ annotation\(s\) that were outside data range.",
    category=RuntimeWarning,
)

warnings.filterwarnings(
    "ignore",
    message="EDF format requires equal-length data blocks, so [0-9.]+ seconds of edge values were appended to all channels when writing the final block.",
    category=RuntimeWarning,
)

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


def seconds_to_min_sec_ms(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{minutes:02}:{secs:02}.{milliseconds:03}"


def time_str_to_seconds(time_str):
    minutes, rest = time_str.split(":")
    seconds, milliseconds = rest.split(".")
    return int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def clean_event_name(name: str):
    """
    Очистка имен событий.
    """
    cleaned_name = re.sub(r"\[.*?\]|\(.*?\)", "", name).strip()
    if cleaned_name in EXCLUDED_NAMES or name in EXCLUDED_NAMES:
        return None

    for ru_name, en_name in TRANSLATIONS.items():
        if ru_name in cleaned_name:
            return en_name

    return cleaned_name


def create_block_csvs(input_dir=".", output_csv_dir="output_csv", skip_labels=None):
    if skip_labels is None:
        skip_labels = list(EXCLUDED_NAMES)

    os.makedirs(output_csv_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".edf"):
            continue

        edf_file = os.path.join(input_dir, filename)
        raw = mne.io.read_raw_edf(edf_file, preload=True)

        annotations = raw.annotations
        onsets = annotations.onset
        descriptions = annotations.description
        total_duration = raw.times[-1]

        blocks = []
        i = 0
        while i < len(descriptions):
            desc = descriptions[i]
            clean_desc = clean_event_name(desc)
            if clean_desc is None:
                i += 1
                continue

            start = onsets[i]
            j = i + 1
            while j < len(descriptions):
                next_desc = descriptions[j]
                clean_next_desc = clean_event_name(next_desc)
                if clean_next_desc is None:
                    j += 1
                else:
                    break

            end = onsets[j] if j < len(onsets) else total_duration
            duration = end - start

            blocks.append((clean_desc, start, duration))
            i = j

        base_filename = os.path.splitext(filename)[0]
        csv_path = os.path.join(output_csv_dir, f"{base_filename}_blocks.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Название", "Время начала", "Длительность"])
            for name, start, duration in blocks:
                writer.writerow(
                    [
                        name,
                        seconds_to_min_sec_ms(start),
                        seconds_to_min_sec_ms(duration),
                    ]
                )

        print(f"CSV создан: {csv_path}")


def export_blocks(
        input_dir=".",
        output_csv_dir="output_csv",
        output_dir="splitted_blocks",
):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".edf"):
            continue

        edf_file = os.path.join(input_dir, filename)
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        base_filename = os.path.splitext(filename)[0]
        csv_file = os.path.join(output_csv_dir, f"{base_filename}_blocks.csv")

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

            file_output_dir = os.path.join(output_dir, base_filename)
            os.makedirs(file_output_dir, exist_ok=True)

            for i, row in enumerate(rows):
                block_name = row["Название"]
                start_sec = time_str_to_seconds(row["Время начала"])
                duration_sec = time_str_to_seconds(row["Длительность"])
                end_sec = start_sec + duration_sec

                raw_block = raw.copy().crop(tmin=start_sec, tmax=end_sec)
                safe_name = (
                    block_name.replace(" ", "_").replace("(", "").replace(")", "")
                )
                out_path = os.path.join(
                    file_output_dir, f"{i + 1:02d}_{safe_name}_{base_filename}.edf"
                )

                raw_block.export(
                    out_path, fmt="edf", physical_range="auto", overwrite=True
                )
                print(f"Сохранён блок: {out_path}")


def split_edf_into_subblocks(
        input_dir: str, output_dir: str, min_duration: float = MIN_BLOCK_DURATION
):
    """
    Разделяет блоки из папок на подблоки заданной длительности.
    :param input_dir: Директория, содержащая папки с блоками.
    :param output_dir: Директория для сохранения подблоков.
    :param min_duration: Минимальная длительность подблока (в секундах).
    """

    min_duration = min_duration - 0.002

    # Создаем выходные директории
    os.makedirs(output_dir, exist_ok=True)

    # Перебираем все папки в input_dir
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        # Проверяем, что это действительно папка
        if not os.path.isdir(folder_path):
            continue

        print(f"Обработка папки: {folder_name}")

        # Создаем выходную директорию для подблоков
        subblocks_folder = os.path.join(output_dir, folder_name)
        os.makedirs(subblocks_folder, exist_ok=True)

        # Перебираем все .edf файлы в папке
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

            total_duration = raw.times[-1]  # Общая длительность записи

            # Имя базового файла (без расширения)
            base_filename = os.path.splitext(os.path.basename(filename))[0]

            print(
                f"Обработка файла: {filename} (длительность: {total_duration:.2f} сек)"
            )

            # Разделяем данные на подблоки
            current_time = 0
            block_index = 1

            while current_time < total_duration:
                # Вычисляем время окончания подблока
                next_time = min(current_time + min_duration, total_duration)

                subblock_duration = next_time - current_time
                if subblock_duration < min_duration:
                    print(
                        f"Подблок {block_index} слишком короткий ({subblock_duration:.2f} сек) и будет пропущен"
                    )
                    break  # Прерываем цикл, так как последний подблок уже обработан

                # Обрезаем данные для подблока
                try:
                    sub_block = raw.copy().crop(tmin=current_time, tmax=next_time)
                except ValueError as e:
                    print(f"Ошибка при обрезке блока: {e}")
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
                    # sub_block.save(out_path, overwrite=True)
                    print(f"Сохранён подблок: {out_path}")
                except Exception as e:
                    print(f"Ошибка при экспорте подблока: {e}")
                    break

                # Переходим к следующему подблоку
                current_time = next_time
                block_index += 1
