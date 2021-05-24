import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
import utils.ecg_record as ecg_record
import utils.alphabet as alphabet


def windows_cv_iter(rec_to_ind, n_folds, train_folds):
    #records_list = list(rec_to_ind.keys())
    a, b, c, d = train_folds
    records_split_ind_iter = [[a + b + c, d],
                              [a + b + d, c],
                              [a + c + d, b],
                              [b + c + d, a]]

    for split_indices in records_split_ind_iter:

        #train_records = [records_list[ind] for ind in split_indices[0]]
        #test_records = [records_list[ind] for ind in split_indices[1]]
        train_records = split_indices[0]
        test_records = split_indices[1]

        test_windows_indices = []
        train_windows_indices = []
        for r in train_records:
            train_windows_indices += rec_to_ind[r]

        for r in test_records:
            test_windows_indices += rec_to_ind[r]

        yield test_windows_indices, train_windows_indices


def windows_to_dataframe(windows_dataset,
                         records_set):
    record_to_ind = {}
    dataset = []
    windows_count = 0
    for record in records_set:
        record_to_ind[record] = []
        for window in windows_dataset[record]:
            dataset.append(window.dict())
            record_to_ind[record].append(windows_count)
            windows_count += 1
    dataset = pd.DataFrame(dataset)
    #dataset.to_csv("C:/Users/nikit/Downloads/dataset32")
    return dataset, record_to_ind


def dataframe_to_numpy(dataframe):
    features = dataframe.columns.drop("label")
    x = dataframe[features].to_numpy()
    y = dataframe["label"].to_numpy()
    return x, y


def evaluate_estmator(estimator, x, y, test_fold, rec_to_ind):
    test_windows_indices = []
    for r in test_fold:
        test_windows_indices += rec_to_ind[r]

    prediction = estimator.predict(x[test_windows_indices])
    prediction_proba = estimator.predict_proba(x[test_windows_indices])
    #for svm should use the next line and change for auc
    #prediction_proba = estimator.decision_function(x[test_windows_indices])

    metrics = dict()

    metrics['auc'] = roc_auc_score(np.array(y[test_windows_indices], dtype=np.int),
                                   prediction_proba[:, 1])
    #metrics['auc'] = roc_auc_score(np.array(y[test_windows_indices], dtype=np.int),
    #                               prediction_proba)

    metrics["sensitivity"] = recall_score(prediction, y[test_windows_indices], pos_label=1)
    metrics["accuracy"] = accuracy_score(prediction, y[test_windows_indices])
    metrics["specificity"] = recall_score(prediction, y[test_windows_indices], pos_label=0)

    return metrics


def cv_search(windows_dataset,
              records_set,
              estimator,
              estimator_params_grid,
              n_folds,
              n_jobs=-1):
    dataset, record_to_ind = \
        windows_to_dataframe(windows_dataset,
                             records_set)
    x, y = dataframe_to_numpy(dataset)

    a = ['04015', '04126', '04936', '07879', '08405']
    b = ['04043', '04048', '07859', '07910']
    c = ['04746', '05261', '08215', '08378', '08455']
    d = ['04908', '06426', '07162', '08219', '08434']
    e = ['05091', '05121', '06453', '06995']
    folds = [a, b, c, d, e]
    results = dict()
    cs = estimator_params_grid['c']
    for c in cs:
        res_ = dict()
        for f in range(n_folds):
            test_fold = folds[f]
            train_folds = [folds[f-1], folds[f-2], folds[f-3], folds[f-4]]

            # record_cv_splitter = windows_cv_iter(
            #     record_to_ind, n_folds, train_folds)

            # random_search = GridSearchCV(
            #     estimator,
            #     param_grid=estimator_params_grid,
            #     n_jobs=n_jobs,
            #     scoring='roc_auc',
            #     cv=record_cv_splitter,
            #     verbose=3,
            #     refit=True
            # )

            # random_search.fit(x, y)

            train_windows_indices = []
            for train_fold in train_folds:
                for r in train_fold:
                    train_windows_indices += record_to_ind[r]

            model = LogisticRegression(C=c, max_iter=1000, n_jobs=-1)
            model.fit(x[train_windows_indices], y[train_windows_indices])

            test_scores = evaluate_estmator(model, x, y, test_fold, record_to_ind)
            res_['fold' + str(f)] = test_scores
            #params['fold' + str(f)] = random_search.best_estimator_
        results["c=" + str(c)] = res_

    return results

def cv_eval(windows_dataset,
            gramms_size,
            estimator,
            estimator_params_grid,
            test_records_tuples,
            n_folds):
    scores = {}
    # for test_records_tuple in test_records_tuples:
    #     logging.info("performing cv for ")
    #     train_records =\
    #         windows_dataset.keys() - test_records_tuple
    train_records = windows_dataset.keys()
    if gramms_size > 0:
        alphabet.normalize_ngramms(
            windows_dataset,
            gramms_size,
            train_records
        )
    metrics = \
        cv_search(
            windows_dataset,
            train_records,
            estimator,
            estimator_params_grid,
            n_folds
        )
    #best_params = best_estimator.get_params()
    # test_dataframe, _ = windows_to_dataframe(
    #     windows_dataset,
    #     test_records_tuple
    # )
    # x, y = dataframe_to_numpy(test_dataframe)

    # test_metrics = evaluate_estmator(best_estimator, x, y)

    #scores[test_records_tuple] = \
        #{"test_metrics": test_metrics,
    scores['scores'] = \
            {"metrics": metrics}

    return scores


def cross_val(
        estimator,
        estimator_params_grid,
        windows_sizes,
        alp_thresholds,
        gramms_sizes,
        test_records_tuples,
        n_folds,
        records_dir="records"):
    results = {}
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    with open('windows.txt', 'r+') as wd, open('results.txt', 'a') as fout:
        done_sizes = wd.readline().strip().split()
        done = set(done_sizes)
        for window_size in windows_sizes:
            if str(window_size) in done:
                continue

            logging.info(
                "building windows dataset"
                " for windows size: {}".format(window_size))
            windows_dataset = ecg_record.build_windows_dataset(
                records_dir, window_size)
            
            for alp_threshold in alp_thresholds:
                logging.info("preparing alphabet"
                            " encoding for treshold:"
                            " {}".format(alp_threshold))
                for record in windows_dataset:
                    for window in windows_dataset[record]:
                        window.prepare_alphabet(alp_threshold)
                for gramms_size in gramms_sizes:
                    logging.info("preparing ngramms of size:"
                                " {}".format(gramms_size))
                    alphabet.build_alphabet_dataset(
                        windows_dataset,
                        gramms_size=gramms_size
                    )
                    key = (window_size, alp_threshold, gramms_size)
                    logging.info("performing cross val for params:"
                                "ws={};alp_treshold={};"
                                "ngramms_size={}".format(window_size,
                                                        alp_threshold,
                                                        gramms_size))
                    results[key] = cv_eval(
                        windows_dataset,
                        gramms_size,
                        estimator,
                        estimator_params_grid,
                        test_records_tuples,
                        n_folds=n_folds,
                    )
                    print(key, ':', results[key], sep=" ", file=fout)
            print(str(window_size) + ' ', end="", file=wd)
        return results
