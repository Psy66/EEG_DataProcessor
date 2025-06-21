import requests
from functools import wraps
from enum import Enum
import os
import logging

logger = logging.getLogger(__name__)


def requires_auth(method):
    """
    Декоратор для методов, требующих авторизации. Проверяет наличие в классе заполненного поля _sid.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_sid', None):
            raise PermissionError(
                f'Метод \'{method.__name__}\' требует авторизации. Вызовите auth() перед его использованием и повторите попытку.')
        return method(self, *args, **kwargs)

    return wrapper


class FileListMode(Enum):
    SHORT = 'short'
    FULL = 'full'
    DIR = 'dir'


# Написан в соответствии с документацией и опытом использования Synology File Station API
# https://global.download.synology.com/download/Document/Software/DeveloperGuide/Package/FileStation/All/enu/Synology_File_Station_API_Guide.pdf


class SynologyAPI:

    def __init__(self, base_url: str, username: str = None, password: str = None, sid: str = None):
        self._base_url = base_url
        self._username = username
        self._password = password
        self._sid = sid
        self._error_codes = {
            119: '_sid не действителен или истёк. Требуется повторная авторизация',
            408: 'Указанный вами файл или директория не найдены в удалённом хранилище',
            400: 'Указанный вами параметр(-ы) или операция(-и) запроса не валидны'
        }

    @classmethod
    def from_config(cls, storage_config):

        synology_base_url = f"{storage_config['protocol']}://{storage_config['host']}:{storage_config['port']}"
        synology_login = storage_config["username"]
        synology_password = storage_config["password"]

        sid = storage_config["sid"] if "sid" in storage_config else None

        if sid:
            synology_api = SynologyAPI(base_url=synology_base_url, sid=sid)
        else:
            synology_api = SynologyAPI(
                base_url=synology_base_url,
                username=synology_login,
                password=synology_password,
            )
            synology_api.auth()

        return synology_api

    def get_api_info(self) -> dict:
        """
        Получает информацию о доступных методах API Synology.

        Returns:
            dict: Словарь с описанием доступных API.
        """

        url = f'{self._base_url}/webapi/query.cgi'
        params = {
            'api': 'SYNO.API.Info',
            'version': '1',
            'method': 'query'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                raise RuntimeError(f'Запрос вернул ошибку: {data}')

            return data.get('data', {})

        except requests.RequestException as e:
            raise ConnectionError(f'Ошибка подключения к Synology API: {e}')

        except ValueError:
            raise ValueError('Не удалось декодировать JSON-ответ от сервера')

    def auth(self) -> bool:
        """
        Аутентифицирует пользователя и сохраняет _sid для дальнейшего использования API.

        Returns:
            bool: True, если авторизация успешна (от сервера был получен _sid). False в противном случае.
        """

        url = f'{self._base_url}/webapi/auth.cgi'
        params = {
            'api': 'SYNO.API.Auth',
            'version': '7',
            'method': 'login',
            'account': self._username,
            'passwd': self._password,
            'session': 'FileStation',
            'format': 'sid'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                error_code = data.get('error', {}).get('code')
                if error_code in self._error_codes:
                    raise RuntimeError(f'Авторизация не удалась: {self._error_codes[error_code]}')
                raise RuntimeError(f'Авторизация не удалась: {data}')

            try:
                self._sid = data['data']['sid']
                return True
            except KeyError:
                raise RuntimeError(f'SID не был получен при авторизации: {data}')

        except requests.RequestException as e:
            raise ConnectionError(f'Ошибка подключения к Synology API: {e}')

        except ValueError:
            raise ValueError('Не удалось декодировать JSON-ответ от сервера')

    @requires_auth
    def get_files_list(self, path: str, mode: FileListMode = FileListMode.SHORT) -> list:
        """
        Получает список файлов в директории.
        Требует авторизации (наличия валидного _sid).

        Args:
            path (str): Путь к директории.
            mode (FileListMode): Режим вывода — краткий (возвращает одноуровневый список файлов) или полный (возвращает
            полный ответ).

        Returns:
            list: Список файлов (в виде строк-путей или словарей с подробной информацией о файлах).
        """

        url = f'{self._base_url}/webapi/entry.cgi'
        params = {
            'api': 'SYNO.FileStation.List',
            'version': '1',
            'method': 'list',
            'folder_path': path,
            '_sid': self._sid,
            'additional': 'size,time'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                error_code = data.get('error', {}).get('code')
                if error_code in self._error_codes:
                    raise PermissionError(f'Ошибка: {self._error_codes[error_code]}')
                raise RuntimeError(f'Ошибка получения списка файлов по пути {path}. Код: {error_code}. Ответ: {data}')

            files = data['data']['files']
            if mode == FileListMode.FULL:
                return files
            elif mode == FileListMode.SHORT:
                return [file.get('path') for file in files]
            elif mode == FileListMode.DIR:
                return [file.get('path') for file in files if file.get('isdir')]
            else:
                raise ValueError(f'Недопустимый режим: {mode}')

        except requests.RequestException as e:
            raise ConnectionError(f'Ошибка подключения к Synology API: {e}')

        except ValueError:
            raise ValueError('Не удалось декодировать JSON-ответ от сервера')

    @requires_auth
    def download_file(self, remote_filepath: str, output_dir: str, overwrite: bool = False) -> str:
        """
        Скачивает файл из хранилища Synology (remote_path) и сохраняет его локально (local_path).
        Требует авторизации (наличия валидного _sid).

        Args:
            remote_filepath (str): Путь к файлу в хранилище Synology (например, '/home/Drive/edf/81275.edf').
            output_dir (str): Директория, в которую нужно сохранить файл на локальной машине.
            overwrite (bool): True - перезаписывать файл, если существует.

        Returns:
            str: Путь к сохранённому локально файлу.
        """

        filename = os.path.basename(remote_filepath)
        output_dir = os.path.abspath(output_dir)
        output_path = os.path.abspath(os.path.join(output_dir, filename))

        remote_dir = os.path.dirname(remote_filepath)
        remote_files = set(self.get_files_list(remote_dir))
        if remote_filepath not in remote_files:
            raise FileNotFoundError(f'Файл \'{remote_filepath}\' не найден в хранилище Synology')

        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f'Целевая локальная директория \'{output_dir}\' не существует.')

        if os.path.isfile(output_path) and not overwrite:
            logger.info(f'Файл {filename} уже существует в указанной директории, отмена загрузки')
            return output_path

        url = f'{self._base_url}/webapi/entry.cgi'
        params = {
            'api': 'SYNO.FileStation.Download',
            'version': '2',
            'method': 'download',
            'mode': 'download',
            'path': remote_filepath,
            '_sid': self._sid
        }

        try:
            with requests.get(url, params=params, stream=True) as response:
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    error_data = response.json()
                    if not error_data.get('success'):
                        error_code = error_data.get('error', {}).get('code')
                        if error_code in self._error_codes:
                            raise PermissionError(f'Ошибка: {self._error_codes[error_code]}')
                        raise RuntimeError(f'Ошибка скачивания файла. Код: {error_code}. Ответ: {error_data}')
                    raise RuntimeError(f'Неожиданный JSON-ответ: {error_data}')

                total_size = int(response.headers.get('Content-Length', 0))
                total_size_mb = total_size / (1024 * 1024)
                downloaded = 0
                chunk_size = 8192
                last_shown_percent = 0

                with open(output_path, 'wb') as f:
                    logger.info(f'Файл {filename} ({total_size_mb:.2f} MB) - загрузка начата')
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            downloaded_mb = downloaded / (1024 * 1024)

                            if total_size_mb:
                                percent = int(downloaded_mb * 100 / total_size_mb)
                                if percent >= last_shown_percent + 10:
                                    logger.info(
                                        f'Скачано {percent}% ({downloaded_mb:.2f} MB из {total_size_mb:.2f} MB)')
                                    last_shown_percent = percent
                            else:
                                logger.info(f'Скачано {downloaded_mb:.2f} MB')

                    logger.info(f'Файл {filename} загружен успешно')
                    return output_path

        except requests.RequestException as e:
            raise ConnectionError(f'Ошибка подключения к Synology API: {e}')

        except ValueError:
            raise ValueError('Не удалось декодировать JSON-ответ от сервера')

        except OSError as e:
            raise RuntimeError(f'Ошибка записи файла на диск: {e}')

    @requires_auth
    def check_writing_permission(self, remote_dir_path: str, filename: str, size: int, overwrite: bool = False) -> bool:
        """
        Проверяет, есть ли разрешение на запись файла в указанную директорию Synology.
        Требует авторизации (наличия валидного _sid).

        Args:
            remote_dir_path (str): Путь к директории в хранилище Synology.
            filename (str): Имя файла, который планируется записать.
            size (int): Размер файла в байтах.
            overwrite (bool): True - перезаписывать ли файл, если он уже существует.

        Returns:
            bool: True, если запись разрешена. False в противном случае.
        """

        url = f'{self._base_url}/webapi/entry.cgi'
        params = {
            'api': 'SYNO.FileStation.CheckPermission',
            'version': '3',
            'method': 'write',
            'path': remote_dir_path,
            'filename': filename,
            'size': size,
            'overwrite': 'true' if overwrite else 'false',
            '_sid': self._sid
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                error_code = data.get('error', {}).get('code')
                if error_code in self._error_codes:
                    raise PermissionError(f'Ошибка: {self._error_codes[error_code]}')
                raise RuntimeError(
                    f'Ошибка при проверке допуска на запись {filename} в {remote_dir_path}. Код: {error_code}. Ответ: {data}')

            # blSkip == false - означает, что запись возможна
            # blSkip == true - что невозможна
            is_writing_allowed = not data['data']['blSkip']
            if is_writing_allowed:
                logger.info(f'Запись файла {filename} в директорию {remote_dir_path} возможна')
            else:
                logger.info(f'Запись файла {filename} в директорию {remote_dir_path} невозможна')

            return is_writing_allowed

        except requests.RequestException as e:
            raise ConnectionError(f'Ошибка подключения к Synology API: {e}')

        except ValueError:
            raise ValueError('Не удалось декодировать JSON-ответ от сервера')

    # Не смотря на документацию - для успешной загрузки через API нужно применять не POST а GET
    # с телом запроса (что вообще-то противоречит спецификации HTTP, но работает только так)
    @requires_auth
    def upload_file(self, local_file_path: str, remote_dir_path: str, create_parents: bool = False,
                    overwrite: bool = False) -> bool:
        """
        Загружает файл в хранилище Synology.

        Args:
            local_file_path (str): Путь к локальному файлу, загружаемому в хранилище.
            remote_dir_path (str): Путь к удалённой директории, в которую будет загружен файл.
            create_parents (bool): True - будут созданы недостающие директории.
            overwrite (bool): True - существующий в хранилище файл будет перезаписан при загрузке.

        Returns:
            bool: True, если загрузка прошла успешно.
         """

        filename = os.path.basename(local_file_path)
        full_remote_path = remote_dir_path + '/' + filename

        # Сперва проверяем существование директории. Если её нет, а create_parents = False - ошибка
        files_list = []

        try:
            files_list = self.get_files_list(remote_dir_path)
        except PermissionError:
            if not create_parents:
                logger.warning(
                    f'Целевая указанная директоия {remote_dir_path} не существует. Невозможно загрузить файл {filename}. '
                    f'Для создания несуществующих директорий укажите create_parents=True')
                return False

        # Затем проверяем существование файла. Если он есть, а overwrite = False - ошибка
        if full_remote_path in files_list and not overwrite:
            logger.warning(f'Файл {filename} уже существует в директории {remote_dir_path}. Загрузка отменена.'
                  f'Для перезаписи существующих файлов укажите overwrite=True')
            return False

        if not os.path.isfile(local_file_path):
            logger.warning(f'Локальный файл {local_file_path} не найден. Загрузка отменена')
            return False

        url = f'{self._base_url}/webapi/entry.cgi'
        params = {
            'api': 'SYNO.FileStation.Upload',
            'version': '2',
            'method': 'upload',
            'path': remote_dir_path,
            'create_parents': 'true' if create_parents else 'false',
            'overwrite': 'true' if overwrite else 'false',
            '_sid': self._sid
        }

        files = [
            ('filename', (f'{filename}', open(f'{local_file_path}', 'rb'), 'application/octet-stream'))
        ]

        try:
            response = requests.get(url, params=params, files=files)
            response.raise_for_status()
            data = response.json()

            if not data.get('success'):
                error_code = data.get('error', {}).get('code')
                if error_code in self._error_codes:
                    raise PermissionError(f'Ошибка: {self._error_codes[error_code]}')
                raise RuntimeError(
                    f'Ошибка при попытке записи {filename} в {remote_dir_path}. Код: {error_code}. Ответ: {data}')

            # blSkip == false - означает, что запись прошла успешно
            # blSkip == true - что запись отменена (возможно, файл уже существует)
            is_writing_succeed = not data['data']['blSkip']
            if is_writing_succeed:
                logger.info(f'Файл {filename} успешно загружен в директорию {remote_dir_path}')
            else:
                logger.warning(f'Загрузка {filename} в директорию {remote_dir_path} не удалась')

            return is_writing_succeed

        except requests.RequestException as e:
            raise ConnectionError(f'Ошибка подключения к Synology API: {e}')

        except ValueError:
            raise ValueError('Не удалось декодировать JSON-ответ от сервера')

    @requires_auth
    def upload_folder(self, local_folder_path: str, remote_folder_path: str,
                      create_parents: bool = False, overwrite: bool = False) -> bool:
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file = os.path.join(root, file)
                self.upload_file(local_file, remote_folder_path, create_parents, overwrite)
