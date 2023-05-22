# -*- coding: utf-8 -*-
# @Time    : 21.05.23 23:34
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : data_split.py
# @Software: PyCharm


import pandas as pd
import argparse
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def read_split_train_test(yaml_path):
    with open(yaml_path) as config_file:
        config_parms = yaml.safe_load(config_file)
    local_data_src = os.path.join(config_parms["preprocess"]["dir"], config_parms["preprocess"]["preprocessed_file"])
    df = pd.read_csv(local_data_src)
    random_state = config_parms["base"]["random_state"]
    train_test_split_ratio = config_parms["split"]["split_ratio"]

    train_data, test_data = train_test_split(
        df,
        test_size=train_test_split_ratio,
        random_state=random_state
    )
    os.makedirs(config_parms["split"]["dir"], exist_ok=True)
    train_data_path = os.path.join(config_parms["split"]["dir"], config_parms["split"]["train_file"])

    train_data.to_csv(train_data_path, index=False)

    test_data_path = os.path.join(config_parms["split"]["dir"], config_parms["split"]["test_file"])
    test_data.to_csv(test_data_path, index=False)

if __name__=="__main__":
    print(f"******************************* Train test split started *******************************")
    read_split_train_test(yaml_path="params.yaml")
    print(f"******************************* Train test split complete *******************************")

