from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# we only consider task of discrimination of AFIB beats from normal
# that's why we delete all windows with
STATE_TO_LABEL = {'(N': 0, '(AFIB': 1, '(AFL': 2, '(J': 2, 'start': 2}


def get_rr_peaks_indices(record, max_bpm=230):
    """
    :param record_name:
    :param database:
    :param max_bpm:
    :return: list of timestamps for
    """

    qrs_inds = processing.gqrs_detect(sig=record.p_signal[:, 0],
                                      fs=record.fs)
    search_radius = int(record.fs * 60 / max_bpm)
    corrected_peak_inds = processing.correct_peaks(
        record.p_signal[:, 0], peak_inds=qrs_inds,
        search_radius=search_radius, smooth_window_size=150)

    result = []
    discarded_count = 0
    prev_pi = -1
    for pi in corrected_peak_inds:
        if pi != prev_pi:
            result.append(pi)
        else:
            discarded_count += 1
        prev_pi = pi
    # returns r-r peaks timestamps
    result = np.array(result) / record.fs

    return result, discarded_count


def visualize_peaks(sig, peak_inds, fs, title,
                    from_peak=0, to_peak=-1,
                    figsize=(20, 10), saveto=None):

    start_indice = peak_inds[from_peak]
    end_indice = peak_inds[to_peak]

    fig, ax_left = plt.subplots(figsize=figsize)

    ax_left.plot(np.arange(start_indice, end_indice + 1) / fs,
                 sig[start_indice:end_indice+1],
                 color='#3979f0', label='Signal'
                 )
    ax_left.plot(np.array(peak_inds[from_peak:to_peak]) / fs,
                 sig[peak_inds[from_peak:to_peak]],
                 'rx', marker='*',
                 color='red', label='Peak',
                 markersize=12
                 )
    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()


def produce_peaks_windows(rr_peaks, window_size, annotation):
    """
    rr_peaks_indices - list of timestaps for r-peaks
    annotation - wfdb annotation object for the record from which got
    """
    # indices of the r peaks which separates ecg states
    states_borders = np.array(annotation.sample) // annotation.fs
    states_borders = np.append(states_borders, rr_peaks[-1] + 1)
    states_borders = np.array(states_borders) / annotation.fs
    states_sequence = ['start'] + annotation.aux_note
    labels_sequence = [STATE_TO_LABEL[s] for s in states_sequence]

    state_ind = 0
    current_label = labels_sequence[state_ind]
    current_border = states_borders[state_ind]

    window = deque([], window_size)
    windows = []
    for peak in rr_peaks:

        if peak >= current_border:
            state_ind += 1
            current_label = labels_sequence[state_ind]
            current_border = states_borders[state_ind]

        window.append((peak, current_label))
        if len(window) == window_size:
            windows.append(window)

    final_windows = []
    for w in windows:
        labels = [p[1] for p in w]
        max_label = max(labels)
        if max_label != 2:
            final_windows.append([[pi[0] for pi in w], max_label])

    return final_windows


def get_rate_window(peaks_window):
    x = [peaks_window[i] - peaks_window[i-1]
         for i in range(1, len(peaks_window))]
    x = np.array(x)
    return x


def get_changes_window(rate_window):
    x = [rate_window[i] / rate_window[i - 1] - 1
         for i in range(1, len(rate_window))]
    return np.array(x)
