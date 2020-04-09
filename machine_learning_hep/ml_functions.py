#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
Methods to: choose, train and apply ML models
            load and save ML models
"""
import pickle

import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
import xgboost as xgb

import templates.templates_keras as templates_keras
import templates.templates_scikit as templates_scikit
import templates.templates_xgboost as templates_xgboost
from logger import get_logger

pd.options.mode.chained_assignment = None

def getclf_scikit(model_config):

    logger = get_logger()
    logger.debug("Load scikit models")

    if "scikit" not in model_config:
        logger.debug("No scikit models found")
        return [], []

    classifiers = []
    names = []
    grid_search_params = []

    for c in model_config["scikit"]:
        if model_config["scikit"][c]["activate"]:
            try:
                model = getattr(templates_scikit, c)(model_config["scikit"][c]["central_params"])
                classifiers.append(model)
                names.append(c)
                grid_search_params.append(model_config["scikit"][c].get("grid_search", {}))
                logger.info("Added scikit model %s", c)
            except AttributeError:
                logger.critical("Could not load scikit model %s", c)

    return classifiers, names, grid_search_params


def getclf_xgboost(model_config):

    logger = get_logger()
    logger.debug("Load xgboost models")

    if "xgboost" not in model_config:
        logger.debug("No xgboost models found")
        return [], []

    classifiers = []
    names = []
    grid_search_params = []

    for c in model_config["xgboost"]:
        if model_config["xgboost"][c]["activate"]:
            try:
                model = getattr(templates_xgboost, c)(model_config["xgboost"][c]["central_params"])
                classifiers.append(model)
                names.append(c)
                grid_search_params.append(model_config["xgboost"][c].get("grid_search", {}))
                logger.info("Added xgboost model %s", c)
            except AttributeError:
                logger.critical("Could not load xgboost model %s", c)

    return classifiers, names, grid_search_params


def getclf_keras(model_config, length_input):

    logger = get_logger()
    logger.debug("Load keras models")

    if "keras" not in model_config:
        logger.debug("No keras models found")
        return [], []

    classifiers = []
    names = []

    for c in model_config["keras"]:
        if model_config["keras"][c]["activate"]:
            try:
                classifiers.append(KerasClassifier(build_fn=lambda name=c: \
                    getattr(templates_keras, name)(model_config["keras"][name], length_input), \
                                               epochs=model_config["keras"][c]["epochs"], \
                                               batch_size=model_config["keras"][c]["batch_size"], \
                                               verbose=0))
                names.append(c)
                logger.info("Added keras model %s", c)
            except AttributeError:
                logger.critical("Could not load keras model %s", c)

    return classifiers, names, []


def set_num_trees(classifiers_, x_train_, y_train_, nfold_, num_early_stopping_, seed_):
    logger = get_logger()
    logger.info("Estimating number of trees")
    new_models = []

    for clf in classifiers_:
        params = clf.get_params(deep=True)
        num_trees = params["n_estimators"]
        xgb_params = clf.get_xgb_params()
        train_data = xgb.DMatrix(x_train_, label=y_train_, silent=True, nthread=4)

        cv_results = xgb.cv(xgb_params, train_data, num_boost_round=num_trees, nfold=nfold_,
                            stratified=True, metrics='auc', seed=seed_,
                            early_stopping_rounds=num_early_stopping_)

        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        mean_std = cv_results['test-auc-std'][boost_rounds]
        logger.info("ROC_AUC %.5f +/- %.5f for %d rounds", mean_auc, mean_std, boost_rounds)

        #cv_results_err = xgb.cv(xgb_params, train_data, num_boost_round=num_trees, nfold=nfold_,
        #                        stratified=True, metrics='error', seed=seed_,
        #                        early_stopping_rounds=num_early_stopping_)

        #mean_err = cv_results_err['test-error-mean'].min()
        #boost_rounds_err = cv_results_err['test-error-mean'].idxmin()
        #mean_std_err = cv_results_err['test-error-std'][boost_rounds_err]
        #logger.info("ERROR %.5f +/- %.5f for %d rounds", mean_err, mean_std_err, boost_rounds_err)

        #cv_results_rmse = xgb.cv(xgb_params, train_data, num_boost_round=num_trees, nfold=nfold_,
        #                         stratified=True, metrics='rmse', seed=seed_,
        #                         early_stopping_rounds=num_early_stopping_)

        #mean_rmse = cv_results_rmse['test-rmse-mean'].min()
        #boost_rounds_rmse = cv_results_rmse['test-rmse-mean'].idxmin()
        #mean_std_rmse = cv_results_rmse['test-rmse-std'][boost_rounds_rmse]
        #logger.info("RMSE %.5f +/- %.5f for %d rounds", mean_rmse,
        #             mean_std_rmse, boost_rounds_rmse)

        #NB: Doesn't seem to work properly
        #cv_results_mae = xgb.cv(xgb_params, train_data, num_boost_round=num_trees, nfold=nfold_,
        #                        stratified=True, metrics='mae', seed=seed_,
        #                        early_stopping_rounds=num_early_stopping_)

        #mean_mae = cv_results_mae['test-mae-mean'].min()
        #boost_rounds_mae = cv_results_mae['test-mae-mean'].idxmin()
        #mean_std_mae = cv_results_mae['test-mae-std'][boost_rounds_mae]
        #logger.info("MAE %.5f +/- %.5f for %d rounds", mean_mae, mean_std_mae, boost_rounds_mae)

        params["n_estimators"] = boost_rounds
        clf.set_params(**params)
        new_models.append(clf)

    return new_models


def fit(names_, classifiers_, x_train_, y_train_):
    trainedmodels_ = []
    for _, clf in zip(names_, classifiers_):
        clf.fit(x_train_, y_train_)
        trainedmodels_.append(clf)
    return trainedmodels_


def test(ml_type, names_, trainedmodels_, test_set_, mylistvariables_, myvariablesy_):
    x_test_ = test_set_[mylistvariables_]
    y_test_ = test_set_[myvariablesy_].values.reshape(len(x_test_),)
    test_set_[myvariablesy_] = pd.Series(y_test_, index=test_set_.index)
    for name, model in zip(names_, trainedmodels_):
        y_test_prediction = []
        y_test_prob = []
        y_test_prediction = model.predict(x_test_)
        y_test_prediction = y_test_prediction.reshape(len(y_test_prediction),)
        test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)

        if ml_type == "BinaryClassification":
            y_test_prob = model.predict_proba(x_test_)[:, 1]
            test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
    return test_set_


def apply(ml_type, names_, trainedmodels_, test_set_, mylistvariablestraining_):
    x_values = test_set_[mylistvariablestraining_]
    for name, model in zip(names_, trainedmodels_):
        y_test_prediction = []
        y_test_prob = []
        y_test_prediction = model.predict(x_values)
        y_test_prediction = y_test_prediction.reshape(len(y_test_prediction),)
        test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)

        if ml_type == "BinaryClassification":
            y_test_prob = model.predict_proba(x_values)[:, 1]
            test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
    return test_set_


def savemodels(names_, trainedmodels_, folder_, suffix_):
    for name, model in zip(names_, trainedmodels_):
        if "keras" in name:
            architecture_file = folder_+"/"+name+suffix_+"_architecture.json"
            weights_file = folder_+"/"+name+suffix_+"_weights.h5"
            arch_json = model.model.to_json()
            with open(architecture_file, 'w') as json_file:
                json_file.write(arch_json)
            model.model.save_weights(weights_file)
        if "scikit" in name:
            fileoutmodel = folder_+"/"+name+suffix_+".sav"
            pickle.dump(model, open(fileoutmodel, 'wb'), protocol=4)
        if "xgboost" in name:
            fileoutmodel = folder_+"/"+name+suffix_+".sav"
            pickle.dump(model, open(fileoutmodel, 'wb'), protocol=4)
            fileoutmodel = fileoutmodel.replace(".sav", ".model")
            model.save_model(fileoutmodel)


def readmodels(names_, folder_, suffix_):
    trainedmodels_ = []
    for name in names_:
        fileinput = folder_+"/"+name+suffix_+".sav"
        model = pickle.load(open(fileinput, 'rb'))
        trainedmodels_.append(model)
    return trainedmodels_
