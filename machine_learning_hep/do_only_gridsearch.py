from os.path import exists
import pickle
import argparse
import yaml
from pkg_resources import resource_stream
from machine_learning_hep.logger import configure_logger, get_logger
from machine_learning_hep.grid_search import do_gridsearch, read_grid_dict, perform_plot_gridsearch

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
    return yaml.safe_load(stream)

def do_grid_search_macro(data_config: dict, data_param: dict, data_model: dict, grid_param: dict, run_param: dict):

    logger = get_logger()

    # If we are here we are interested in the very first key in the parameters database
    for k in data_param.keys():
        case = k
        break

    typean = data_config["analysis"]["type"]
    print("typean =", typean)
    mltype = data_param[case]["ml"]["mltype"]
    print("mltype =", mltype)

    analysisdb = grid_param[mltype]
    names_cv, clf_cv, par_grid_cv, refit_cv, var_param, par_grid_cv_keys = read_grid_dict(analysisdb)

    print("Opening df_xtrain")
    df_xtrain = pickle.load(open("/data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1/mlout_TrainingInput/xtrain_BsITS2_dfselection_pt_cand_7.0_10.0.pkl", 'rb'))
    print("Opening df_ytrain")
    df_ytrain = pickle.load(open("/data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1/mlout_TrainingInput/ytrain_BsITS2_dfselection_pt_cand_7.0_10.0.pkl", 'rb'))
    nkfolds = data_param[case]["ml"]["nkfolds"]
    ncorescross = data_param[case]["ml"]["ncorescrossval"]
    _, _, dfscore = do_gridsearch(names_cv, clf_cv, par_grid_cv, refit_cv, df_xtrain, df_ytrain, nkfolds, ncorescross)

    perform_plot_gridsearch(names_cv, dfscore, par_grid_cv, par_grid_cv_keys, var_param, "./GridSearch/", "bin_710", 0.1)

def do_only_gridsearch():

    print("Starting macro")
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="activate debug log level")
    parser.add_argument("--log-file", dest="log_file", help="file to print the log to")
    parser.add_argument("--run-config", "-r", dest="run_config",
                        help="the run configuration to be used")
    parser.add_argument("--database-analysis", "-d", dest="database_analysis",
                        help="analysis database to be used")
    parser.add_argument("--database-ml-models", dest="database_ml_models",
                        help="ml model database to be used")
    parser.add_argument("--database-ml-gridsearch", dest="database_ml_gridsearch",
                        help="ml gridsearch database to be used")
    parser.add_argument("--database-run-list", dest="database_run_list",
                        help="run list database to be used")
    parser.add_argument("--analysis", "-a", dest="type_ana",
                        help="choose type of analysis")

    args = parser.parse_args()

    configure_logger(args.debug, args.log_file)

    pkg_data = "machine_learning_hep.data"
    pkg_data_run_config = "machine_learning_hep"
    run_config = load_config(args.run_config, (pkg_data_run_config, "default_complete.yaml"))
    case = run_config["case"]
    print("case =", case)
    if args.type_ana is not None:
        run_config["analysis"]["type"] = args.type_ana

    db_analysis_default_name = f"database_ml_parameters_{case}.yml"
    db_analysis = load_config(args.database_analysis, (pkg_data, db_analysis_default_name))
    db_ml_models = load_config(args.database_ml_models, (pkg_data, "config_model_parameters.yml"))
    db_ml_gridsearch = load_config(args.database_ml_gridsearch,
                                   (pkg_data, "database_ml_gridsearch.yml"))
    db_run_list = load_config(args.database_run_list, (pkg_data, "database_run_list.yml"))

    do_grid_search_macro(run_config, db_analysis, db_ml_models, db_ml_gridsearch, db_run_list)

do_only_gridsearch()
