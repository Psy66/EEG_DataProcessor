# from edf_processing.edf_split.split.block_split import create_block_csvs, export_blocks
from edf_segmentor.block_split import create_block_csvs, export_blocks, split_edf_into_subblocks
import logging


logger = logging.getLogger(__name__)


def edf_split(input_dir="temp/edf/input",
              output_csv_dir="temp/splitted_blocks/config",
              output_blocks_dir="temp/splitted_blocks",
              output_subblocks_dir="temp/subblocks"):
    """
        Выполняет поэтапное разбиение EDF-файлов на блоки (сегменты) и подблоки (блоки).

        Функция выполняет три последовательные операции:
        1. Генерация CSV-файлов с конфигурацией блоков на основе аннотаций.
        2. Экспорт блоков в отдельные EDF-файлы по информации из CSV.
        3. Дополнительное разбиение экспортированных блоков (сегментов) на подблоки (блоки).

        Args:
            input_dir (str): Путь к директории с исходными EDF-файлами.
            output_csv_dir (str):  Путь к директории, куда сохраняются сгенерированные CSV-файлы с конфигурацией блоков.
            output_blocks_dir (str):  Путь к директории, куда сохраняются экспортированные блоки в формате EDF.
            output_subblocks_dir (str): Путь к директории, куда сохраняются подблоки. По умолчанию: "temp/subblocks".

        Returns:
            None

        Логгирует информацию о завершении разбиения.
        """
    # input_dir = edf_path
    # output_csv_dir = "temp/splitted_blocks/config"
    # output_blocks_dir = "temp/splitted_blocks"
    # output_subblocks_dir = "temp/subblocks"


    # Вызов функции для создания CSV с блоками по аннотациям
    create_block_csvs(
        input_dir=input_dir,
        output_csv_dir=output_csv_dir,
        skip_labels=None,
    )

    # Вызов функции для сохранения блоков по CSV
    export_blocks(
        input_dir=input_dir, output_csv_dir=output_csv_dir, output_dir=output_blocks_dir
    )

    split_edf_into_subblocks(
        input_dir=output_blocks_dir, output_dir=output_subblocks_dir
    )

    logger.info("Разбиение на блоки завершено")


if __name__ == "__main__":
    edf_path = "eedfs"
    edf_split(edf_path)
