# -*- coding: utf-8 -*-
# @Time    : 21.05.23 23:33
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : evaluate.py
# @Software: PyCharm

import json
import math
import os
import pickle
import sys

import pandas as pd
from sklearn import metrics
from dvclive import Live
from matplotlib import pyplot as plt
import yaml
def evaluate(yaml_path, model, data, category, live):
    with open(yaml_path) as cofig_file:
        config_parms = yaml.safe_load(cofig_file)
    target = [config_parms["base"]["target_col"]]
    y = data[target]

    x = data.drop(target, axis=1)

    predictions_by_class = model.predict_proba(x)

    predictions = predictions_by_class[:, 1]

    # Use dvclive to log a few simple metrics...
    avg_prec = metrics.average_precision_score(y['quality'].values, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)
    if not live.summary:
        live.summary = {"avg_prec": {}, "roc_auc": {}}
    live.summary["avg_prec"][category] = avg_prec
    live.summary["roc_auc"][category] = roc_auc

    # ... and plots...
    live.log_sklearn_plot("roc", y, predictions, name=f"roc/{category}")

    precision, recall, prc_thresholds = \
        metrics.precision_recall_curve(y, predictions)
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    prc_dir = os.path.join("eval", "prc")
    os.makedirs(prc_dir, exist_ok=True)
    prc_file = os.path.join(prc_dir, f"{category}.json")
    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )

    # ... confusion matrix plot
    live.log_sklearn_plot("confusion_matrix",
                          y.squeeze(),
                          predictions_by_class.argmax(-1),
                          name=f"cm/{category}"
                          )

    return ""


if __name__ == "__main__":
    print(f"******************************* Evaluation started *******************************")
    param_yaml_path = "params.yaml"
    with open(param_yaml_path) as config_file:
        config_params = yaml.safe_load(config_file)
    model_path = config_params["model_dir"]
    with open(model_path + "/trained_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)

    train_data_path = os.path.join(config_params["split"]["dir"], config_params["split"]["train_file"])
    train_data = pd.read_csv(train_data_path)

    test_data_path = os.path.join(config_params["split"]["dir"], config_params["split"]["test_file"])
    test_data = pd.read_csv(test_data_path)

    # Evaluate
    EVAL_PATH = "eval"
    live = Live(os.path.join(EVAL_PATH, "live"), dvcyaml=False)
    evaluate(param_yaml_path, model, train_data, "train", live)
    evaluate(param_yaml_path, model, test_data, "test", live)
    live.make_summary()


    # Dump feature importance image and show it with your plots.
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    importances = model.feature_importances_
    x = train_data.drop('quality', axis=1)
    feature_names = [f"feature {i}" for i in range(x.shape[1])]
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
    axes.set_ylabel("Mean decrease in impurity")
    forest_importances.plot.bar(ax=axes)

    fig.savefig(os.path.join(EVAL_PATH, "importance.png"))
    print(f"******************************* Evaluation complete *******************************")











