# -*- coding: utf-8 -*-
# @Time    : 22.05.23 10:01
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : pipeline.py
# @Software: PyCharm

import os
import webbrowser, yaml
if __name__ =="__main__":

    if not os.path.exists(".dvc"):
        # DVC not initialize DVC dvc init --subdir as git is already initialize in parent dir
        os.system('dvc init --subdir')
    #add local dvc_remote directory as remote. The dvc_remote directory should be create beforehand
    os.system("dvc remote add -d local ./dvc_remote --force")

    # Call data preprocessing
    stage_1 = "dvc stage add -n data_preprocessing -p data_source.local_path -d src/process.py -o data/preprocessed python src/process.py "

    os.system(stage_1)
    print("Stage 1 completed")

    # Call train_test_split
    stage_2 = "dvc stage add -n train_test_preprocessed_split -p split.dir,split.train_file,split.test_file -d src/data_split.py -d data/preprocessed -o data/split python src/data_split.py data/split"

    print(stage_2)

    os.system(stage_2)
    print("Stage 2 completed")

    # Train model
    stage_3 = "dvc stage add --force -n train -p split.dir,split.train_file,split.test_file,base.random_state,base.target_col,train.n_est,model_dir -d src/train_model.py -d data/split -o model/trained_tuned_model.pkl python src/train_model.py data/features model/trained_tuned_model.pkl"

    print(stage_3)
    os.system(stage_3)
    print("Stage 3 completed")

    #evaluate
    stage_4 = "dvc stage add -n evaluate -d src/evaluate.py -d model/trained_tuned_model.pkl -d data/split -M eval/live/metrics.json -O eval/live/plots -O eval/prc -o eval/importance.png python src/evaluate.py model/trained_tuned_model.pkl data/split"

    print(stage_4)
    os.system(stage_4)
    print("Stage 4 completed")

    print("####################################################################################")
    print("######################## RUNNING COMPLETE PIPELINE       ###########################")
    os.system("dvc repro")
    print("################### RUNNING COMPLETE PIPELINE COMPLETED ###########################")
    print("####################################################################################")

    with open("dvc.yaml") as config_file:
        config_parms = yaml.safe_load(config_file)



    updateyml={'plots': [{'ROC': {'template': 'simple',
            'x': 'fpr',
            'y': {'eval/live/plots/sklearn/roc/train.json': 'tpr',
             'eval/live/plots/sklearn/roc/test.json': 'tpr'}}},
          {'Confusion-Matrix': {'template': 'confusion',
            'x': 'actual',
            'y': {'eval/live/plots/sklearn/cm/train.json': 'predicted',
             'eval/live/plots/sklearn/cm/test.json': 'predicted'}}},
          {'Precision-Recall': {'template': 'simple',
            'x': 'recall',
            'y': {'eval/prc/train.json': 'precision',
             'eval/prc/test.json': 'precision'}}},
          'eval/importance.png']}

    config_parms = {**updateyml, **config_parms, }
    print(config_parms)
    with open("dvc.yaml", "w") as f:
        yaml.dump(config_parms, f)

    # Generate plots auc-roc
    file = os.popen("dvc plots show")
    output = file.read()
    webbrowser.open_new_tab(output)

    #push to remote
    #note we've set local remote
    os.system("dvc push")
""":Other commands
dvc metrics show
dvc plots show

dvc metrics diff
dvc plots diff

dvc remove <name>
dvc remove train  

"""