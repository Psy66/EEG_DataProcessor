# Т. к. на данный момент нет возможности брать данные напрямую из БД - был взят дамп из неё, и сохранён
# В формате .csv в директории original

# patients.csv
# Из БД была взята информация о записях, имеющих:
# - Монополярный монтаж (montage = Monopolar)
# - 21 канал (channel_count = 21)
# - 19 ЭЭГ каналов (eeg_channel_count = 19)

# Всего вышло 610 записей

# mapping.csv
# Содержит информацию о диагнозах для каждого .edf файла

# prepare - соединяет все данные из .csv файлов в единый файл и преобразует некоторые поля,
# меняя данные на актуальные, для облегчения дальнейшей обработки

import pandas as pd

def prepare():
    NEW_PATH_PREFIX = '/home/Drive/EEGBASE/Valeoton/EDF'

    studies = pd.read_csv('./original/studies.csv')
    diagnoses = pd.read_csv('./original/mapping.csv')

    studies['file_path'] = studies['file_path'].apply(lambda x: NEW_PATH_PREFIX + '/' + x.split('/')[-1])
    studies['file_name'] = studies['file_path'].apply(lambda x: x.split('/')[-1])

    studies_diagnoses = pd.merge(studies, diagnoses, left_on='file_name', right_on='New Name', how='inner')
    studies_diagnoses.drop(columns=['New Name'], inplace=True)
    studies_diagnoses.rename(columns={'Diag_Code': 'diag_code'}, inplace=True)

    studies_diagnoses.to_csv('./prepared/studies.csv', index=False)


if __name__ == '__main__':
    prepare()