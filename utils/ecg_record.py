from utils import ecg_processing
import wfdb
import os
import sys
import logging
import pickle
import numpy as np
import utils.alphabet as alphabet


class ECGWindow:

    def __init__(self, peaks_window):
        peaks_window, label = peaks_window
        self.label = label
        self.heart_rate_window =\
            ecg_processing.get_rate_window(peaks_window)
        self.changes_window =\
            ecg_processing.get_changes_window(self.heart_rate_window)

        self.alphabet_encoding = []
        self.ngramms_count = {}
        self.ngramms_idf = {}

        self.mean_hr = self.heart_rate_window.mean()
        self.median_hr = np.median(self.heart_rate_window)
        self.variance_hr = self.heart_rate_window.var()

        self.mean_ch = self.changes_window.mean()
        self.variance_ch = self.changes_window.var()
        self.median_ch = np.median(self.changes_window)
        self.mean_abs_ch = np.mean(np.abs(self.changes_window))
        self.max_ch = np.max(self.changes_window)
        self.min_ch = np.min(self.changes_window)
        self.sum_ch_abs = np.sum(self.changes_window)

    def dict(self):
        dict_ = {"mean_hr": self.mean_hr,
                 "median_hr": self.median_hr,
                 "variance_hr": self.variance_hr,
                 "mean_ch": self.mean_ch,
                 "variance_ch": self.variance_ch,
                 "median_ch": self.median_ch,
                 "mean_abs_ch": self.mean_abs_ch,
                 "max_ch": self.max_ch,
                 "min_ch": self.min_ch,
                 "sum_ch_abs": self.sum_ch_abs,
                 "label": self.label
                }
        if len(self.ngramms_idf) > 0:
            source_ngramms = self.ngramms_idf
        else:
            source_ngramms = self.ngramms_count

        ngramms = {"".join(n_gramm): source_ngramms[n_gramm]
                   for n_gramm in source_ngramms}

        dict_ = {**dict_, **ngramms}

        return dict_

    def prepare_alphabet(self, alphabet_threshold):
        self.alphabet_encoding = alphabet.get_alphabet_encoding(self.changes_window, alphabet_threshold)

    def normalize_ngramms(self, idf):
        norm = 0
        for ngramm in self.ngramms_count:
            self.ngramms_idf[ngramm] =\
                self.ngramms_count[ngramm] * idf[ngramm]
            norm += self.ngramms_idf[ngramm] ** 2
        norm = norm ** 0.5
        for ngramm in self.ngramms_idf:
            self.ngramms_idf[ngramm] /= norm


class ECGRPeaksRecord:

    def __init__(self, name, database):

        self.name = name
        self.database = database
        self.rr_peaks = None
        self.annotation = None
        self.windows = None

    def download(self):
        logging.info("downloading record: {}".format(self.name))
        record = wfdb.rdrecord(self.name,
                               pn_dir=self.database,
                               channels=[0])

        self.annotation = wfdb.io.rdann(self.name,
                                        pn_dir=self.database,
                                        extension='atr')

        self.rr_peaks, discarded_count = \
            ecg_processing.get_rr_peaks_indices(record)

        logging.info("peak detection for record {} is finished"
                     " number of duplicated peaks {} number of detected peaks {}".
                     format(self.name, discarded_count, len(self.rr_peaks)))

    def generate_windows(self, windows_size):
        peaks_windows = ecg_processing.produce_peaks_windows(
            self.rr_peaks,
            windows_size,
            self.annotation
        )
        windows = [ECGWindow(p) for p in peaks_windows]
        return windows


def download_database(
        data_folder,
        database='afdb',
        ):
    records = [ECGRPeaksRecord(r, database)
               for r in wfdb.get_record_list(database)]
    logging.info("saving records")
    for r in records:
        try:
            r.download()
        except Exception:
            exc_type, _, traceback = sys.exc_info()
            logging.info("failed to download record:{};\n"
                         "Exception:{}\nTraceback:{}\n\n".
                         format(r.name, exc_type, traceback)
                         )
            continue
        path_to_record = os.path.join(data_folder, r.name)
        with open(path_to_record, "wb") as f:
            pickle.dump(r, f)


def restore_records(data_folder):
    files = os.listdir(data_folder)
    files = [os.path.join(data_folder, f) for f in files]
    records = []
    for fpath in files:
        with open(fpath, 'rb') as f:
            records.append(pickle.load(f))

    return records


def build_windows_dataset(records_folder, windows_size):
    records = restore_records(records_folder)
    windows = dict()
    for record in records:
        windows[record.name] = record.generate_windows(windows_size)
    return windows
