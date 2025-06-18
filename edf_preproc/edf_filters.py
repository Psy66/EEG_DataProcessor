import mne
import numpy as np


def notch_filter(raw, freqs):
    filtered_raw = raw.notch_filter(freqs=freqs)
    return filtered_raw


def bandpass_filter(raw, l_freq, h_freq):
    filtered_raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    return filtered_raw


def sigma_3_filter(raw):
    events, titles = mne.events_from_annotations(raw)

    # Исключаем события с названием 'stimFlash'
    blacklist_titles = ['stimFlash']
    whitelist_numbers = [
        eid for title, eid in titles.items() if title not in blacklist_titles
    ]

    # Фильтруем события
    events_mask = np.isin(events[:, -1], whitelist_numbers)
    clear_events = events[events_mask]

    # Получаем временные метки событий
    event_times = raw.times[clear_events[:, 0]]
    event_times = np.append(event_times, raw.times[-1])
    event_times = np.insert(event_times, 0, raw.times[0])

    segments = []
    original_annotations = raw.annotations.copy()

    for i in range(len(event_times) - 1):
        start = event_times[i]
        end = event_times[i + 1]

        # segment = raw.copy().crop(tmin=start, tmax=end)
        # data = segment.get_data()

        data = raw.get_data(tmin=start, tmax=end)

        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)

        mask = np.abs(data - mean) > 3 * std
        data[mask] = np.broadcast_to(mean, data.shape)[mask]

        for ch in range(data.shape[0]):
            bad_idx = np.where(mask[ch])[0]
            good_idx = np.where(~mask[ch])[0]
            if len(good_idx) >= 2:
                data[ch, bad_idx] = np.interp(bad_idx, good_idx, data[ch, good_idx])

        cleaned = mne.io.RawArray(data, raw.info)
        segments.append(cleaned)

    raw_cleaned_combined = mne.concatenate_raws(segments)
    raw_cleaned_combined.set_annotations(original_annotations)

    return raw_cleaned_combined
