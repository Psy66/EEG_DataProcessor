import h5py


def explore_hdf5(file_path):
    def recursively_print(name, obj):
        print(f"\nPath: {name}")

        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: shape={obj.shape}, dtype={obj.dtype}")
            if obj.shape[0] <= 5:  # Покажем первые значения для небольших массивов
                print(f"🔍 Data preview: {obj[()]}")
        elif isinstance(obj, h5py.Group):
            print("Group")

        # Атрибуты объекта
        for key, value in obj.attrs.items():
            print(f"Attr: {key} = {value}")

    with h5py.File(file_path, "r") as f:
        print(f"\nExploring HDF5 file: {file_path}")

        # Печать глобальных атрибутов (корневой группы)
        print("\nGlobal Attributes:")
        for key, value in f.attrs.items():
            print(f"Attr: {key} = {value}")

        # Печать содержимого всех групп/датасетов
        f.visititems(recursively_print)


explore_hdf5(r"C:\Users\Max\Desktop\EEG_DataProcessor\temp\download_upload\hdf5_output\F95.2_Baseline.h5")


def explore_data_blocks(file_path):
    print(f"\nExploring data blocks in: {file_path}")

    with h5py.File(file_path, "r") as f:
        for source_file_id in f:
            group = f[source_file_id]
            print(f"\nSource file: {source_file_id}")

            # Данные
            data = group.get("data")
            if data is not None:
                print(f"Data shape: {data.shape} (blocks, channels, samples)")

            # Метаданные
            meta = group.get("metadata")
            if meta:
                gender = meta.attrs.get("gender", b"N")
                age_cat = meta.attrs.get("age_cat", b"Unknown")

                # Декодируем атрибуты
                gender = gender.decode("utf-8") if isinstance(gender, bytes) else gender
                age_cat = (
                    age_cat.decode("utf-8") if isinstance(age_cat, bytes) else age_cat
                )

                print(f"Gender: {gender}, Age Category: {age_cat}")

                # Показываем все source_files
                if "source_files" in meta:
                    sources = [s.decode("utf-8") for s in meta["source_files"][:]]
                    print(f"Source Files ({len(sources)}):")
                    for s in sources:
                        print(f"   - {s}")
                else:
                    print("Source Files: None")


explore_data_blocks(r"C:\Users\Max\Desktop\EEG_DataProcessor\temp\download_upload\hdf5_output\F95.2_Baseline.h5")
