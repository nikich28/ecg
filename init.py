import utils.alphabet as alphabet
import utils.cv as cv
import utils.ecg_processing as proc
import utils.ecg_record as record 
import pickle
import os
import logging
import yaml
import sys
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier



with open(r'params.yml') as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)
path = params_list["path"]
ws = params_list['windows']
trs = params_list['tresholds']
ngrs = params_list['ngramms']
est_params = params_list["est_params"]

#record.download_database(path)

#format: estimator, est_params, windows_sizes, alphabet tresholds, ngramms sizes, -, -, path to records 
#res = cv.cross_val(XGBClassifier(), {}, [32], [0.2], [3], {}, 5, "C:/Users/nikit/Downloads/ecg-master/ecg-master/records")
res = cv.cross_val(XGBClassifier(), est_params, ws, trs, ngrs, {}, 5, path)
print(res) #also results are in results.txt file
