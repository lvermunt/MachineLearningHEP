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
Methods to: perform gridsearch
"""
import pickle
from os.path import join as osjoin

import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from utilities import print_dict, dump_yaml_from_dict
from logger import get_logger
from ml_functions import savemodels
from utilities import openfile

def get_scorers(score_names):
    """Construct dictionary of scorers

    Args:
        score_names: tuple of names. Available names see below
    Returns:
        dictionary mapping scorers to names
    """

    scorers = {}
    for sn in score_names:
        if sn == "AUC":
            scorers["AUC"] = make_scorer(roc_auc_score, needs_threshold=True)
        elif sn == "Accuracy":
            scorers["Accuracy"] = make_scorer(accuracy_score)

    return scorers


def do_gridsearch(names, classifiers, grid_params, x_train, y_train, nkfolds, out_dirs, ncores=-1):
    """Hyperparameter grid search for a list of classifiers

    Given a list of classifiers, do a hyperparameter grid search based on a corresponding
    set of parameters

    Args:
        names: iteratable of classifier names
        classifiers: iterable of classifiers
        grid_params: iterable of parameters used to perform the grid search
        x_train: feature dataframe
        y_train: targets dataframe
        nkfolds: int, cross-validation generator or an iterable
        out_dirs: Write parameters and pickle of summary dataframe
        ncores: number of cores to distribute jobs to
    Returns:
        lists of grid search models, the best model and scoring dataframes
    """

    logger = get_logger()

    for clf_name, clf, gps, out_dir in zip(names, classifiers, grid_params, out_dirs):
        if not gps:
            logger.info("Nothing to be done for grid search of model %s", clf_name)
            continue
        logger.info("Grid search for model %s with following parameters:", clf_name)
        print_dict(gps)

        # To work for probabilities. This will call model.decision_function or
        # model.predict_proba as it is done for the nominal ROC curves as well to decide on the
        # performance
        scoring = get_scorers(gps["scoring"])

        grid_search = GridSearchCV(clf, gps["params"], cv=nkfolds, refit=gps["refit"],
                                   scoring=scoring, n_jobs=ncores, verbose=2,
                                   return_train_score=True)
        grid_search.fit(x_train, y_train)
        cvres = grid_search.cv_results_

        # Save the results as soon as we have them in case something goes wrong later
        # (would be quite unfortunate to loose grid search results...)
        out_file = osjoin(out_dir, "results.pkl")
        pickle.dump(pd.DataFrame(cvres), openfile(out_file, "wb"), protocol=4)
        # Parameters
        dump_yaml_from_dict(gps, osjoin(out_dir, "parameters.yaml"))
        savemodels((clf_name,), (grid_search.best_estimator_,), out_dir, "")
