from itertools import product
import numpy as np


def get_alphabet_encoding(window, threshold):
    changes = window
    alp_window = []
    for change in changes:
        if abs(change) < threshold:
            alp_window.append('A')
        elif change >= threshold:
            alp_window.append('B')
        else:
            alp_window.append('C')
    return alp_window


def get_BOG(alp_window, possible_trigramms):
    ngramms_size = len(possible_trigramms[0])
    n_gramms = {g: 0 for g in possible_trigramms}
    n_gramms_size = len(possible_trigramms[0])
    for ind in range(len(alp_window) - (n_gramms_size - 1)):
        n_gramm = tuple(alp_window[ind:ind+ngramms_size])
        n_gramms[n_gramm] += 1
    return n_gramms


def normalize_ngramms(windows_dataset,
                      gramms_size,
                      records_for_idf_calc):

    possible_ngramms = list(product('ABC', repeat=gramms_size))

    wind_with_gramm_count = {t: 0 for t in possible_ngramms}

    number_of_windows = 0
    for record in records_for_idf_calc:
        for window in windows_dataset[record]:
            number_of_windows += 1
            for g in possible_ngramms:
                if window.ngramms_count[g] > 0:
                    wind_with_gramm_count[g] += 1

    idf = {t: np.log(number_of_windows /
                     (1 + wind_with_gramm_count[t]))
           for t in possible_ngramms}

    for record in windows_dataset:
        for window in windows_dataset[record]:
            window.normalize_ngramms(idf)


def build_alphabet_dataset(windows_dataset,
                           gramms_size=3):
    """
    :param windows_dataset: dict "record_name": [list of windows]
    """

    possible_ngramms = list(product('ABC', repeat=gramms_size))

    number_of_windows = 0

    for record in windows_dataset:
        number_of_windows += len(windows_dataset[record])

    for record in windows_dataset:
        for window in windows_dataset[record]:
            window.ngramms_count = get_BOG(
                window.alphabet_encoding,
                possible_ngramms
            )