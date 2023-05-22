# -*- coding: utf-8 -*-
# @Time    : 21.05.23 23:34
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : process.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import yaml
import os

def pre_processing(yaml_path):
    with open(yaml_path) as config_file:
        config_parms = yaml.safe_load(config_file)
    # print(config_parms["data_source"]["local_path"])
    local_data_src = config_parms["data_source"]["local_path"]
    df = pd.read_csv(local_data_src)
    # mark > 60% as good quality
    df['quality'] = np.where(df['quality'] > 6.0, 1, 0)
    processed_Data = df.dropna()
    os.makedirs(config_parms["preprocess"]["dir"], exist_ok=True)
    preprocess_dir = os.path.join(config_parms["preprocess"]["dir"], config_parms["preprocess"]["preprocessed_file"])
    processed_Data.to_csv(preprocess_dir, index=False)

if __name__=="__main__":
    print(f"******************************* Preprocessing started *******************************")
    pre_processing(yaml_path="params.yaml")
    print(f"******************************* Preprocessing complete *******************************")

