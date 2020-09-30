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

from os.path import dirname, join
import sys
from glob import glob
import multiprocessing as mp
import yaml
import pandas as pd
import numpy as np
import pickle
from lz4 import frame
from root_numpy import fill_hist
from ROOT import TFile, gRandom, TH1F, TH2F

from machine_learning_hep.bitwise import filter_bit_df
from machine_learning_hep.utilities import derive
from machine_learning_hep.utilities import openfile
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities import list_folders, createlist

def read_database(path, overwrite_path=None):
    data_param = None
    with open(path, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    case = list(data_param.keys())[0]
    data_param = data_param[case]
    if overwrite_path:
        overwrite_db = None
        with open(overwrite_path, 'r') as param_config:
            overwrite_db = yaml.load(param_config, Loader=yaml.FullLoader)
        modify_dictionary(data_param, overwrite_db)
    return case, data_param

#############
# FUNCTIONS #
#############

def _callback(e):
    print(e)

def multi_proc(function, argument_list, kw_argument_list, maxperchunk, max_n_procs=10):
        chunks_args = [argument_list[x:x+maxperchunk] \
                  for x in range(0, len(argument_list), maxperchunk)]
        if not kw_argument_list:
            kw_argument_list = [{} for _ in argument_list]
        chunks_kwargs = [kw_argument_list[x:x+maxperchunk] \
                  for x in range(0, len(kw_argument_list), maxperchunk)]
        res = None
        for chunk_args, chunk_kwargs in zip(chunks_args, chunks_kwargs):
            print("Processing new chunck size=", maxperchunk)
            pool = mp.Pool(max_n_procs)
            res = [pool.apply_async(function, args=args, kwds=kwds, error_callback=_callback) for args, kwds in zip(chunk_args, chunk_kwargs)]
            pool.close()
            pool.join()

        res_list = None
        try:
            res_list = [r.get() for r in res]
        except Exception as e:
            print("EXCEPTION")
            pass
        return res_list


def fill_from_pickles(file_paths, histo_params, cols=None, query=None, queryrunlist=None, querybit=None, skim_func=None, skim_func_args=None, queries=None, merge_on=None):
    print(f"Process files {file_paths}")

    dfs = [pickle.load(openfile(f, "rb")) for f in file_paths]
    df = dfs[0]
    if len(dfs) > 1:
        if merge_on and len(merge_on) != len(dfs) - 1:
            print(f"ERROR: merge_on must be {len(dfs) - 1} however found to be {len(merge_on)}")
            sys.exit(1)

        for df_, on in zip(dfs[1:], merge_on):
            # Recursively merge dataframes
            df = pd.merge(df, df_, on=on)

    if query:
        # Apply common query
        df = df.query(query)
    if queryrunlist:
        df = selectdfrunlist(df, databases["runselection"][queryrunlist], "run_number")
    if querybit:
        df = filter_bit_df(df, "cand_type", querybit)
    if cols:
        # Select already columns which are needed in the following
        df = df[cols]

    if skim_func:
        # Skim the dataframe according to user function
        df = skim_func(df, skim_func_args)


    histos = []
    if not queries:
        queries = [None] * len(histo_params)

    if len(queries) != len(histo_params):
        print("ERROR: Need as many queries as histogram parameters")
        sys.exit(1)

    for hp, qu in zip(histo_params, queries):
        n_cols = len(hp[0])
        if n_cols > 2:
            print(f"ERROR: Cannot handle plots with dimension > 2")
            sys.exit(1)
        histo_func = TH1F if n_cols == 1 else TH2F

        df_fill = df
        if qu:
            # If there is an additional query for this histogram apply it to dataframe
            df_fill = df.query(qu)

        # Arrange for 1D or 2D plotting
        fill_with = df_fill[hp[0][0]] if n_cols == 1 else df_fill[hp[0]].to_numpy()

        histo_name = "_".join(hp[0])
        histo = histo_func(histo_name, histo_name, *hp[1])

        weights = df_fill[hp[2]] if len(hp) == 3 else None
        fill_hist(histo, fill_with, weights=weights)
        histo.SetDirectory(0)
        histos.append(histo)

    return histos


def only_one_evt(df, dupl_cols):
    return df.drop_duplicates(dupl_cols)

#######
# SET #
#######
CASE = "data" #"mc" #"data"
CAND = "Ds" # D0
trigger_type = "MB"

INV_MASS_CAND = {"Ds": 1.969, "Lc": 2.2864, "D0": 1.864}
database_paths = {"Ds": {"MB": "/home/lvermunt/test/MachineLearningHEP/machine_learning_hep/data/database_ml_parameters_Dspp_years.yml",
                         "HMSPD": "/home/lvermunt/test/MachineLearningHEP/machine_learning_hep/data/database_ml_parameters_Dspp.yml",
                         "HMV0": "/home/lvermunt/test/MachineLearningHEP/machine_learning_hep/data/database_ml_parameters_Dspp.yml"},
                  "Lc": {"MB": "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/data_prod_20200304/database_ml_parameters_LcpK0spp_0304.yml",
                         "HMV0": "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/data_prod_20200304/database_ml_parameters_LcpK0spp_0304_HM_V0.yml"},
                  "D0": {"MB": "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/data_prod_20200417/database_ml_parameters_D0pp_0417.yml",
                         "HMV0": "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/data_prod_20200417/database_ml_parameters_D0pp_0417_HM_V0.yml"}}

if CASE == "mc":
    query_all = [None, "is_ev_rej == 0", "is_ev_rej == 0", "is_ev_rej == 0", None, "is_ev_rej == 0", "is_ev_rej == 0", "is_ev_rej == 0"]
elif trigger_type == "MB":
    query_all = ["trigger_hasbit_INT7", "is_ev_rej == 0 and trigger_hasbit_INT7", "is_ev_rej == 0 and trigger_hasbit_INT7", "is_ev_rej == 0 and trigger_hasbit_INT7", \
                 "trigger_hasbit_INT7", "is_ev_rej == 0 and trigger_hasbit_INT7", "is_ev_rej == 0 and trigger_hasbit_INT7", "is_ev_rej == 0 and trigger_hasbit_INT7"]
elif trigger_type == "HMSPD":
    query_all = ["trigger_hasbit_HighMultSPD", "is_ev_rej == 0 and trigger_hasbit_HighMultSPD", "is_ev_rej == 0 and trigger_hasbit_HighMultSPD", "is_ev_rej == 0 and trigger_hasbit_HighMultSPD", \
                 "trigger_hasbit_HighMultSPD", "is_ev_rej == 0 and trigger_hasbit_HighMultSPD", "is_ev_rej == 0 and trigger_hasbit_HighMultSPD", "is_ev_rej == 0 and trigger_hasbit_HighMultSPD"]
elif trigger_type == "HMV0":
    query_all = ["trigger_hasbit_HighMultV0", "is_ev_rej == 0 and trigger_hasbit_HighMultV0", "is_ev_rej == 0 and trigger_hasbit_HighMultV0", "is_ev_rej == 0 and trigger_hasbit_HighMultV0", \
                 "trigger_hasbit_HighMultV0", "is_ev_rej == 0 and trigger_hasbit_HighMultV0", "is_ev_rej == 0 and trigger_hasbit_HighMultV0", "is_ev_rej == 0 and trigger_hasbit_HighMultV0"]

databases = {}
for t, p in database_paths[CAND].items():
    _, databases[t] = read_database(p)
databases["runselection"] = read_database("data/database_run_list.yml")

file_name_evtorig = databases[trigger_type]["files_names"]["namefile_evtorig"] #AnalysisResultsEvtOrig.pkl.lz4
file_name_evt = databases[trigger_type]["files_names"]["namefile_evt"] #AnalysisResultsEvt.pkl.lz4
file_name_reco = databases[trigger_type]["files_names"]["namefile_reco"] #AnalysisResultsReco.pkl.lz4
file_name = [file_name_evtorig, file_name_evt, file_name_reco, file_name_reco, file_name_evtorig, file_name_evt, file_name_reco, file_name_reco]

select_children = databases[trigger_type]["multi"][CASE].get("select_children", None)

add_queries = {"MB": [None, None, None], "HMSPD": ["run_number < 0", "run_number < 0", None], "HMV0": ["run_number >= 256941", None, None]}
add_queries_runs = {"MB": [None, None, None], "HMSPD": [None, None, "HighMultSPD2018"], "HMV0": ["V0vspt_perc_v0m_2016", None, None]}
add_queries = add_queries[trigger_type]
add_queries_runs = add_queries_runs[trigger_type]

variables = ["n_tracklets", "n_tracklets", "n_tracklets", "n_tracklets", "n_tracklets_corr", "n_tracklets_corr", "n_tracklets_corr", "n_tracklets_corr"]
histo_name_suf = ["AllEv", "EvSel", "STDWithCand", "STDWithD", "AllEv", "EvSel", "STDWithCand", "STDWithD"]

file_out_name = f"input_efficiency_weights_{CASE}_{CAND}_{trigger_type}.root"
file_out = TFile.Open(file_out_name, "RECREATE")

for MODE in range(8):
    queries = [None]
    if CASE == "mc":
        queries = [None]
    histo_names = [f"h{histo_name_suf[MODE]}_{variables[MODE]}"]
    histo_params = [([variables[MODE]], (200, 0, 200))]

    histo_xtitles = ["n_{trkl}"]
    histo_ytitles = ["entries"]

    inv_mass_ref = INV_MASS_CAND[CAND]

    if CASE == "data" and (MODE == 3 or MODE == 7):
        query_all[MODE] = query_all[MODE] + " and abs(inv_mass - @inv_mass_ref) <= 0.02"

    if MODE == 3 or MODE == 7:
        cols = ["ev_id", "ev_id_ext", "run_number", "n_tracklets", "n_tracklets_corr", "inv_mass", "cand_type"]
    else:
        cols = ["ev_id", "ev_id_ext", "run_number", "n_tracklets", "n_tracklets_corr"]

    merge_on = [cols[:3]]

    index = 0
    for period, dir_applied, add_query, selruns in zip(databases[trigger_type]["multi"][CASE]["period"],
                                                       databases[trigger_type]["multi"][CASE]["pkl"],
                                                       add_queries, add_queries_runs):

        query_all_tmp = query_all[MODE]
        if CASE == "data" and query_all[MODE] and add_query:
            query_all_tmp = f"{query_all[MODE]} and {add_query}"

        select_children_temp = select_children
        if select_children:
            # Make sure we have "<child>/" instead if <child> only. Cause in the latter case
            # "child_1" might select further children like "child_11"
            select_children = [f"{child}/" for child in select_children[index]]
        listpath = list_folders(dir_applied, file_name[MODE], -1, select_children)
        select_children = select_children_temp
        files_all = createlist(dir_applied, listpath, file_name[MODE])

        if (MODE == 2 or MODE == 3 or MODE == 6 or MODE == 7):
            args = [((f_reco,), histo_params, cols, query_all_tmp, selruns, [[0],[]], only_one_evt, merge_on[0], queries, None) \
                    for f_reco in files_all]
        else:
            args = [((f_reco,), histo_params, cols, query_all_tmp, selruns, None, None, None, queries, None) \
                    for f_reco in files_all]

        histos = multi_proc(fill_from_pickles, args, None, 100, 30)

        histos_added = histos[0]
        for h_list in histos[1:]:
            for h_added, h in zip(histos_added, h_list):
                h_added.Add(h)

        histo_names_period = [f"{name}_{period}" for name in histo_names]
        for h_add, name, xtitle, ytitle in zip(histos_added, histo_names_period, histo_xtitles, histo_ytitles):
            h_add.SetName(name)
            h_add.SetTitle(name)
            h_add.GetXaxis().SetTitle(xtitle)
            h_add.GetYaxis().SetTitle(ytitle)

            file_out.WriteTObject(h_add)
        index = index + 1
file_out.Close()

