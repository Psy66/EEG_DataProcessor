import mne

def ica_filter(raw, n_components=0.99, random_state=42):
    raw_copy = raw.copy()

    picks = mne.pick_types(raw_copy.info, eeg=True)

    # Фильтрация для обучения ICA (1–45 Гц)
    raw_for_ica = raw_copy.filter(1.0, 45.0)

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter='auto')
    ica.fit(raw_for_ica, picks=picks)

    eog_channels = ['EEG FP1-A1', 'EEG FP2-A2']
    eog_inds, scores = ica.find_bads_eog(raw_copy, ch_name=eog_channels)
    ica.exclude = eog_inds

    raw_cleaned = raw_copy
    ica.apply(raw_cleaned)

    return raw_cleaned
