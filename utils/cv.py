import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
import utils.ecg_record as ecg_record
import utils.alphabet as alphabet


def windows_cv_iter(rec_to_ind, n_folds):
    cv = KFold(random_state=420, n_splits=n_folds, shuffle=True)
    records_list = list(rec_to_ind.keys())
    records_split_ind_iter = cv.split(records_list)

    for split_indices in records_split_ind_iter:

        train_records = [records_list[ind] for ind in split_indices[0]]
        test_records = [records_list[ind] for ind in split_indices[1]]

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
    return dataset, record_to_ind


def dataframe_to_numpy(dataframe):
    features = dataframe.columns.drop("label")
    x = dataframe[features].to_numpy()
    y = dataframe["label"].to_numpy()
    return x, y


def evaluate_estmator(estimator, x, y):
    prediction = estimator.predict(x)
    prediction_proba = estimator.predict_proba(x)
    #prediction_proba = estimator.decision_function(x)

    metrics = dict()

    metrics['auc'] = roc_auc_score(np.array(y, dtype=np.int),
                                   prediction_proba[:, 1])
    #metrics['auc'] = roc_auc_score(np.array(y, dtype=np.int),
    #                               prediction_proba)

    metrics["sensitivity"] = recall_score(prediction, y, pos_label=1)
    metrics["accuracy"] = accuracy_score(prediction, y)
    metrics["specificity"] = recall_score(prediction, y, pos_label=0)

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

    record_cv_splitter = windows_cv_iter(
        record_to_ind, n_folds)

    random_search = GridSearchCV(
        estimator,
        param_grid=estimator_params_grid,
        n_jobs=n_jobs,
        scoring='f1',
        cv=record_cv_splitter,
        verbose=3,
        refit=True
    )

    random_search.fit(x, y)
    print('!!!')
    print(random_search.cv_results_)
    print('!!!')

    train_scores = evaluate_estmator(random_search, x, y)
    #estimator.fit(x, y)
    #train_scores = evaluate_estmator(estimator, x, y)

    return random_search.best_estimator_, train_scores
    #return estimator, train_scores


def cv_eval(windows_dataset,
            gramms_size,
            estimator,
            estimator_params_grid,
            test_records_tuples,
            n_folds):
    scores = {}
    for test_records_tuple in test_records_tuples:
        logging.info("performing cv for ")
        train_records =\
            windows_dataset.keys() - test_records_tuple

        if gramms_size > 0:
            alphabet.normalize_ngramms(
                windows_dataset,
                gramms_size,
                train_records
            )
        best_estimator, train_metrics = \
            cv_search(
                windows_dataset,
                train_records,
                estimator,
                estimator_params_grid,
                n_folds
            )
        best_params = best_estimator.get_params()
        test_dataframe, _ = windows_to_dataframe(
            windows_dataset,
            test_records_tuple
        )
        x, y = dataframe_to_numpy(test_dataframe)

        test_metrics = evaluate_estmator(best_estimator, x, y)

        scores[test_records_tuple] = \
            {"test_metrics": test_metrics,
             "train_metrics": train_metrics,
             "best_params": best_params}

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
    for window_size in windows_sizes:
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
    return results
