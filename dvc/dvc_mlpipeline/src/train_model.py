# -*- coding: utf-8 -*-
# @Time    : 22.05.23 10:16
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : train_model.py
# @Software: PyCharm

from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import sys
import pandas as pd
import numpy as np
import yaml


def train_model(yaml_path):
    with open(yaml_path) as config_file:
        config_parms = yaml.safe_load(config_file)

    training_data = os.path.join(config_parms["split"]["dir"], config_parms["split"]["train_file"])
    testing_data = os.path.join(config_parms["split"]["dir"], config_parms["split"]["test_file"])


    target = [config_parms["base"]["target_col"]]

    train = pd.read_csv(training_data)
    test = pd.read_csv(testing_data)

    y_train = train[target]
    y_test = test[target]

    x_train = train.drop(target, axis=1)
    x_test = test.drop(target, axis=1)

    random_state = config_parms["base"]["random_state"]
    n_est = config_parms["train"]["n_est"]

    rfc = RandomForestClassifier(random_state=random_state, n_estimators=n_est)
    rfc.fit(x_train, y_train.values.ravel())

    model_dir = config_parms["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    with open(model_dir + "/trained_tuned_model.pkl", "wb") as f:
        pickle.dump(rfc, f)

if __name__=="__main__":
    print(f"******************************* Training started *******************************")
    train_model(yaml_path="params.yaml")
    print(f"******************************* Training complete *******************************")

