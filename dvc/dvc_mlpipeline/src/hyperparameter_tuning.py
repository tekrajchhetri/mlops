# -*- coding: utf-8 -*-
# @Time    : 21.05.23 23:34
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : https://tekrajchhetri.com/
# @File    : hyperparameter_tuning.py
# @Software: PyCharm

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def hyperparameter_tuning(x_train, y_train):

    hyperparameter_grid = {'n_estimators': [5, 10, 25, 55, 105,200],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(10, 100, num=10)],
                   'min_samples_split': [2, 4, 6, 8, 10],
                   'min_samples_leaf': [1, 2, 3, 4, 5] ,
                   'bootstrap': [True, False]
                   }


    clf = RandomForestClassifier()
    tuning_model = RandomizedSearchCV(estimator=clf,
                                      param_distributions=hyperparameter_grid,
                                      n_iter=101,
                                      cv=10, verbose=2, random_state=45,
                                      n_jobs=6
                                      )
    tuning_model.fit(x_train, y_train)

    print(f'Random grid: {hyperparameter_grid}')

    print(f'Best Parameters: {tuning_model.best_params_}')

    best_params = tuning_model.best_params_

    n_estimators = best_params['n_estimators']
    max_features = best_params['max_features']
    min_samples_leaf = best_params['min_samples_leaf']
    max_depth = best_params['max_depth']
    min_samples_split = best_params['min_samples_split']
    bootstrap = best_params['bootstrap']

    tuned_model = RandomForestClassifier(n_estimators=n_estimators,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=max_features,
                                         max_depth=max_depth,
                                         bootstrap=bootstrap)
    tuned_model.fit(x_train, y_train)
    return tuned_model, best_params