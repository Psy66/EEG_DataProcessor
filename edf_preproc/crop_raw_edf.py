def crop_raw_edf(raw, crop_time=5.0):
    duration = raw.times[-1]

    # 1. Обрезаем сигнал
    cropped_raw = raw.crop(tmin=crop_time, tmax=duration - crop_time)

    # 2. Обрезаем и сдвигаем аннотации
    cropped_annotations = cropped_raw.annotations
    cropped_annotations.onset -= crop_time  # Сдвиг

    # 3. Применяем новые аннотации к обрезанному сигналу
    cropped_raw.set_annotations(cropped_annotations)

    return cropped_raw
