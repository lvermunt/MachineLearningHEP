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
main script for doing data processing, machine learning and analysis
"""
import argparse
import subprocess
import sys
from os.path import exists

import warnings
import yaml
from pkg_resources import resource_stream
# To set batch mode immediately
from ROOT import gROOT  # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.logger import configure_logger, get_logger
from machine_learning_hep.multiprocesser import MultiProcesser
from machine_learning_hep.optimiser import Optimiser
from machine_learning_hep.processer import Processer
from machine_learning_hep.processerdhadrons import ProcesserDhadrons
from machine_learning_hep.processerdhadrons_mult import ProcesserDhadrons_mult
from machine_learning_hep.utilities import checkdirlist, checkdir
from machine_learning_hep.utilities import checkmakedirlist, checkmakedir

from machine_learning_hep.optimiser_hipe4ml import Optimiserhipe4ml

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import logging
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler) # pylint: disable=protected-access
    absl.logging._warn_preinit_stderr = False # pylint: disable=protected-access
except Exception as e: # pylint: disable=broad-except
    print("##############################")
    print("Failed to fix absl logging bug", e)
    print("##############################")


def do_entire_analysis(data_config: dict, data_param: dict, data_model: dict, run_param: dict): # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    # Disable any graphical stuff. No TCanvases opened and shown by default
    gROOT.SetBatch(True)

    logger = get_logger()
    logger.info("Do analysis chain")

    # If we are here we are interested in the very first key in the parameters database
    for k in data_param.keys():
        case = k
        break

    dodownloadalice = data_config["download"]["alice"]["activate"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    domergingmc = data_config["merging"]["mc"]["activate"]
    domergingdata = data_config["merging"]["data"]["activate"]
    doskimmingmc = data_config["skimming"]["mc"]["activate"]
    doskimmingdata = data_config["skimming"]["data"]["activate"]
    domergingperiodsmc = data_config["mergingperiods"]["mc"]["activate"]
    domergingperiodsdata = data_config["mergingperiods"]["data"]["activate"]
    doml = data_config["ml_study"]["activate"]
    docorrelation = data_config["ml_study"]['docorrelation']
    dotraining = data_config["ml_study"]['dotraining']
    dotesting = data_config["ml_study"]['dotesting']
    doapplytodatamc = data_config["ml_study"]['doapplytodatamc']
    docrossvalidation = data_config["ml_study"]['docrossvalidation']
    dolearningcurve = data_config["ml_study"]['dolearningcurve']
    doroc = data_config["ml_study"]['doroc']
    doroctraintest = data_config["ml_study"]['doroctraintest']
    doboundary = data_config["ml_study"]['doboundary']
    doimportance = data_config["ml_study"]['doimportance']
    dogridsearch = data_config["ml_study"]['dogridsearch']
    doefficiencyml = data_config["ml_study"]['doefficiency']
    dosignifopt = data_config["ml_study"]['dosignifopt']
    doscancuts = data_config["ml_study"]["doscancuts"]
    doplotdistr = data_config["ml_study"]["doplotdistr"]
    doapplydata = data_config["mlapplication"]["data"]["doapply"]
    doapplymc = data_config["mlapplication"]["mc"]["doapply"]
    domergeapplydata = data_config["mlapplication"]["data"]["domergeapply"]
    domergeapplymc = data_config["mlapplication"]["mc"]["domergeapply"]
    docontinueapplydata = data_config["mlapplication"]["data"]["docontinueafterstop"]
    docontinueapplymc = data_config["mlapplication"]["mc"]["docontinueafterstop"]
    dohistomassmc = data_config["analysis"]["mc"]["histomass"]
    dohistomassdata = data_config["analysis"]["data"]["histomass"]
    doefficiency = data_config["analysis"]["mc"]["efficiency"]

    typean = data_config["analysis"]["type"]

    dirpklmc = data_param[case]["multi"]["mc"]["pkl"]
    dirpklevtcounter_allmc = data_param[case]["multi"]["mc"]["pkl_evtcounter_all"]
    dirpklskmc = data_param[case]["multi"]["mc"]["pkl_skimmed"]
    dirpklmlmc = data_param[case]["multi"]["mc"]["pkl_skimmed_merge_for_ml"]
    dirpklmltotmc = data_param[case]["multi"]["mc"]["pkl_skimmed_merge_for_ml_all"]
    dirpkldata = data_param[case]["multi"]["data"]["pkl"]
    dirpklevtcounter_alldata = data_param[case]["multi"]["data"]["pkl_evtcounter_all"]
    dirpklskdata = data_param[case]["multi"]["data"]["pkl_skimmed"]
    dirpklmldata = data_param[case]["multi"]["data"]["pkl_skimmed_merge_for_ml"]
    dirpklmltotdata = data_param[case]["multi"]["data"]["pkl_skimmed_merge_for_ml_all"]
    dirpklskdecmc = data_param[case]["mlapplication"]["mc"]["pkl_skimmed_dec"]
    dirpklskdec_mergedmc = data_param[case]["mlapplication"]["mc"]["pkl_skimmed_decmerged"]
    dirpklskdecdata = data_param[case]["mlapplication"]["data"]["pkl_skimmed_dec"]
    dirpklskdec_mergeddata = data_param[case]["mlapplication"]["data"]["pkl_skimmed_decmerged"]

    dirresultsdata = data_param[case]["analysis"][typean]["data"]["results"]
    dirresultsmc = data_param[case]["analysis"][typean]["mc"]["results"]
    dirresultsdatatot = data_param[case]["analysis"][typean]["data"]["resultsallp"]
    dirresultsmctot = data_param[case]["analysis"][typean]["mc"]["resultsallp"]

    binminarray = data_param[case]["ml"]["binmin"]
    binmaxarray = data_param[case]["ml"]["binmax"]
    raahp = data_param[case]["ml"]["opt"]["raahp"]
    mltype = data_param[case]["ml"]["mltype"]
    training_vars = data_param[case]["variables"]["var_training"]

    mlout = data_param[case]["ml"]["mlout"]
    mlplot = data_param[case]["ml"]["mlplot"]

    proc_type = data_param[case]["analysis"][typean]["proc_type"]

    domloption = data_param[case]["hipe4ml"]["dohipe4ml"]
    opti_hyperpar_hipe4ml = data_param[case]["hipe4ml"]["hyper_par_opt"]["do_hyp_opt"]
    hipe4ml_hyper_pars = data_param[case]["hipe4ml"]["hipe4ml_hyper_pars"]

    #creating folder if not present
    counter = 0
    if doconversionmc is True:
        counter = counter + checkdirlist(dirpklmc)

    if doconversiondata is True:
        counter = counter + checkdirlist(dirpkldata)

    if doskimmingmc is True:
        checkdirlist(dirpklskmc)
        counter = counter + checkdir(dirpklevtcounter_allmc)

    if doskimmingdata is True:
        counter = counter + checkdirlist(dirpklskdata)
        counter = counter + checkdir(dirpklevtcounter_alldata)

    if domergingmc is True:
        counter = counter + checkdirlist(dirpklmlmc)

    if domergingdata is True:
        counter = counter + checkdirlist(dirpklmldata)

    if domergingperiodsmc is True:
        counter = counter + checkdir(dirpklmltotmc)

    if domergingperiodsdata is True:
        counter = counter + checkdir(dirpklmltotdata)

    if doml is True:
        counter = counter + checkdir(mlout)
        counter = counter + checkdir(mlplot)

    if docontinueapplymc is False:
        if doapplymc is True:
            counter = counter + checkdirlist(dirpklskdecmc)

        if domergeapplymc is True:
            counter = counter + checkdirlist(dirpklskdec_mergedmc)

    if docontinueapplydata is False:
        if doapplydata is True:
            counter = counter + checkdirlist(dirpklskdecdata)

        if domergeapplydata is True:
            counter = counter + checkdirlist(dirpklskdec_mergeddata)

    if dohistomassmc is True:
        counter = counter + checkdirlist(dirresultsmc)
        counter = counter + checkdir(dirresultsmctot)

    if dohistomassdata is True:
        counter = counter + checkdirlist(dirresultsdata)
        counter = counter + checkdir(dirresultsdatatot)

    if counter < 0:
        sys.exit()
    # check and create directories

    if doconversionmc is True:
        checkmakedirlist(dirpklmc)

    if doconversiondata is True:
        checkmakedirlist(dirpkldata)

    if doskimmingmc is True:
        checkmakedirlist(dirpklskmc)
        checkmakedir(dirpklevtcounter_allmc)

    if doskimmingdata is True:
        checkmakedirlist(dirpklskdata)
        checkmakedir(dirpklevtcounter_alldata)

    if domergingmc is True:
        checkmakedirlist(dirpklmlmc)

    if domergingdata is True:
        checkmakedirlist(dirpklmldata)

    if domergingperiodsmc is True:
        checkmakedir(dirpklmltotmc)

    if domergingperiodsdata is True:
        checkmakedir(dirpklmltotdata)

    if doml is True:
        checkmakedir(mlout)
        checkmakedir(mlplot)

    if docontinueapplymc is False:
        if doapplymc is True:
            checkmakedirlist(dirpklskdecmc)

        if domergeapplymc is True:
            checkmakedirlist(dirpklskdec_mergedmc)

    if docontinueapplydata is False:
        if doapplydata is True:
            checkmakedirlist(dirpklskdecdata)

        if domergeapplydata is True:
            checkmakedirlist(dirpklskdec_mergeddata)

    if dohistomassmc is True:
        checkmakedirlist(dirresultsmc)
        checkmakedir(dirresultsmctot)

    if dohistomassdata is True:
        checkmakedirlist(dirresultsdata)
        checkmakedir(dirresultsdatatot)

    proc_class = Processer
    if proc_type == "Dhadrons":
        print("Using new feature for Dhadrons")
        proc_class = ProcesserDhadrons
    if proc_type == "Dhadrons_mult":
        print("Using new feature for Dhadrons_mult")
        proc_class = ProcesserDhadrons_mult

    mymultiprocessmc = MultiProcesser(case, proc_class, data_param[case], typean, run_param, "mc")
    mymultiprocessdata = MultiProcesser(case, proc_class, data_param[case], typean, run_param,\
                                        "data")

    #perform the analysis flow
    if dodownloadalice == 1:
        subprocess.call("../cplusutilities/Download.sh")

    if doconversionmc == 1:
        mymultiprocessmc.multi_unpack_allperiods()

    if doconversiondata == 1:
        mymultiprocessdata.multi_unpack_allperiods()

    if doskimmingmc == 1:
        mymultiprocessmc.multi_skim_allperiods()

    if doskimmingdata == 1:
        mymultiprocessdata.multi_skim_allperiods()

    if domergingmc == 1:
        mymultiprocessmc.multi_mergeml_allperiods()

    if domergingdata == 1:
        mymultiprocessdata.multi_mergeml_allperiods()

    if domergingperiodsmc == 1:
        mymultiprocessmc.multi_mergeml_allinone()

    if domergingperiodsdata == 1:
        mymultiprocessdata.multi_mergeml_allinone()

    if doml is True and domloption == 1:
        index = 0
        for binmin, binmax in zip(binminarray, binmaxarray):
            myopt = Optimiser(data_param[case], case, typean,
                              data_model[mltype], binmin, binmax,
                              raahp[index], training_vars[index])
            if docorrelation is True:
                myopt.do_corr()
            if dotraining is True:
                myopt.do_train()
            if dotesting is True:
                myopt.do_test()
            if doapplytodatamc is True:
                myopt.do_apply()
            if docrossvalidation is True:
                myopt.do_crossval()
            if dolearningcurve is True:
                myopt.do_learningcurve()
            if doroc is True:
                myopt.do_roc()
            if doroctraintest is True:
                myopt.do_roc_train_test()
            if doplotdistr is True:
                myopt.do_plot_model_pred()
            if doimportance is True:
                myopt.do_importance()
            if dogridsearch is True:
                myopt.do_grid()
            if doboundary is True:
                myopt.do_boundary()
            if doefficiencyml is True:
                myopt.do_efficiency()
            if dosignifopt is True:
                myopt.do_significance()
            if doscancuts is True:
                myopt.do_scancuts()
            index = index + 1

    if doapplydata is True:
        mymultiprocessdata.multi_apply_allperiods()
    if doapplymc is True:
        mymultiprocessmc.multi_apply_allperiods()
    if domergeapplydata is True:
        mymultiprocessdata.multi_mergeapply_allperiods()
    if domergeapplymc is True:
        mymultiprocessmc.multi_mergeapply_allperiods()
    if dohistomassmc is True:
        mymultiprocessmc.multi_histomass()
    if dohistomassdata is True:
        mymultiprocessdata.multi_histomass()
    if doefficiency is True:
        mymultiprocessmc.multi_efficiency()

    if doml is True and domloption == 2:
        index = 0
        for binmin, binmax in zip(binminarray, binmaxarray):
            myopthipe4ml = Optimiserhipe4ml(data_param[case], binmin, binmax,
                                            training_vars[index],
                                            hipe4ml_hyper_pars[index])

            if opti_hyperpar_hipe4ml is True:
                myopthipe4ml.do_hipe4mlhyperparopti()
            else:
                myopthipe4ml.set_hipe4ml_modelpar()

            myopthipe4ml.do_hipe4mltrain()
            myopthipe4ml.do_hipe4mlplot()
            index = index + 1


def load_config(user_path: str, default_path: tuple) -> dict:
    """
    Quickly extract either configuration given by user and fall back to package default if no user
    config given.
    Args:
        user_path: path to YAML file
        default_path: tuple were to find the resource and name of resource
    Returns:
        dictionary built from YAML
    """
    logger = get_logger()
    stream = None
    if user_path is None:
        stream = resource_stream(default_path[0], default_path[1])
    else:
        if not exists(user_path):
            logger_string = f"The file {user_path} does not exist."
            logger.fatal(logger_string)
        stream = open(user_path)
    return yaml.load(stream, yaml.FullLoader)
    #return yaml.safe_load(stream)

def main():
    """
    This is used as the entry point for ml-analysis.
    Read optional command line arguments and launch the analysis.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="activate debug log level")
    parser.add_argument("--log-file", dest="log_file", help="file to print the log to")
    parser.add_argument("--run-config", "-r", dest="run_config",
                        help="the run configuration to be used")
    parser.add_argument("--database-analysis", "-d", dest="database_analysis",
                        help="analysis database to be used")
    parser.add_argument("--database-ml-models", dest="database_ml_models",
                        help="ml model database to be used")
    parser.add_argument("--database-run-list", dest="database_run_list",
                        help="run list database to be used")
    parser.add_argument("--analysis", "-a", dest="type_ana",
                        help="choose type of analysis")

    args = parser.parse_args()

    configure_logger(args.debug, args.log_file)

    # Extract which database and run config to be used
    pkg_data = "machine_learning_hep.data"
    pkg_data_run_config = "machine_learning_hep.submission"
    run_config = load_config(args.run_config, (pkg_data_run_config, "default_complete.yml"))
    case = run_config["case"]
    if args.type_ana is not None:
        run_config["analysis"]["type"] = args.type_ana

    db_analysis_default_name = f"database_ml_parameters_{case}.yml"
    print(args.database_analysis)
    db_analysis = load_config(args.database_analysis, (pkg_data, db_analysis_default_name))
    db_ml_models = load_config(args.database_ml_models, (pkg_data, "config_model_parameters.yml"))
    db_run_list = load_config(args.database_run_list, (pkg_data, "database_run_list.yml"))

    # Run the chain
    do_entire_analysis(run_config, db_analysis, db_ml_models, db_run_list)
