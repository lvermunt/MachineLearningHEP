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


import sys
from glob import glob
import multiprocessing as mp
import argparse
import pickle

import pandas as pd
import yaml
from lz4 import frame # pylint: disable=unused-import

from root_numpy import fill_hist # pylint: disable=import-error

from ROOT import TFile, TH1F, TH2F # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.utilities import openfile
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.io import parse_yaml
from machine_learning_hep.do_variations import modify_dictionary


# Needed here for multiprocessing
INV_MASS = [None]
INV_MASS_WINDOW = [None]


def only_one_evt(df_in, dupl_cols):
    return df_in.drop_duplicates(dupl_cols)

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

def summary_histograms_and_write(file_out, histos, histo_names,
                                 histo_xtitles, histo_ytitles):

    histos_added = histos[0]
    for h_list in histos[1:]:
        for h_added, h in zip(histos_added, h_list):
            h_added.Add(h)

    for h_add, name, xtitle, ytitle \
            in zip(histos_added, histo_names, histo_xtitles, histo_ytitles):
        h_add.SetName(name)
        h_add.SetTitle(name)
        h_add.GetXaxis().SetTitle(xtitle)
        h_add.GetYaxis().SetTitle(ytitle)

        file_out.WriteTObject(h_add)


def derive(periods, in_top_dirs, gen_file_name, required_columns, use_mass_window, # pylint: disable=too-many-arguments, too-many-branches
           distribution_column, distribution_x_range, file_name_mlwp_map, file_out_name,
           queryrunlists, queries_periods=None, query_all=None, queries_slices=None):

    """

    make n_tracklets distributions for all events

    """
    queryrunlists = [None] * len(periods) if not queryrunlists else queryrunlists
    queries_periods = [None] * len(periods) if not queries_periods else queries_periods

    # Prepare histogram parameters
    queries_slices = [None] if not queries_slices else queries_slices
    histo_names = [f"{distribution_column}_{i}" for i in range(len(queries_slices))]

    histo_params = [([distribution_column], distribution_x_range) for _ in histo_names]

    histo_xtitles = [distribution_column] * len(histo_params)
    histo_ytitles = ["entries"] * len(histo_params)

    file_out = TFile.Open(file_out_name, "RECREATE")

    merge_on = [required_columns[:3]]

    for period, dir_applied, query_period, queryrunlist in zip(periods, in_top_dirs, queries_periods, queryrunlists): # pylint: disable=too-many-nested-blocks

        query_tmp = None
        if query_all:
            query_tmp = query_all
            if query_period:
                query_tmp += f" and {query_period}"
        elif query_period:
            query_tmp = query_period

        if use_mass_window:
            if query_tmp:
                query_tmp += " and abs(inv_mass - @INV_MASS[0]) <= @INV_MASS_WINDOW[0]"
            else:
                query_tmp = "abs(inv_mass - @INV_MASS) <= @INV_MASS_WINDOW"

        files_all = glob(f"{dir_applied}/**/{gen_file_name}", recursive=True)

        if not file_name_mlwp_map:
            args = [((f_reco,), histo_params, required_columns, \
                    query_tmp, queryrunlist, only_one_evt, merge_on[0], queries_slices, None) \
                    for f_reco in files_all]

        else:
            print(file_name_mlwp_map)
            args = []
            for file_name in files_all:
                found = False
                query_tmp_file = query_tmp
                for key, value in file_name_mlwp_map.items():
                    if key in file_name:
                        if query_tmp_file:
                            query_tmp_file += f" and {value}"
                        else:
                            query_tmp_file = value
                        found = True
                        break
                if not found:
                    print(f"ERROR: {file_name}")
                    sys.exit(0)
                args.append(((file_name,), histo_params, required_columns, \
                        query_tmp_file, queryrunlist, only_one_evt, merge_on[0], queries_slices, None))

        histos = multi_proc(fill_from_pickles, args, None, 100, 30)

        histo_names_period = [f"{name}_{period}" for name in histo_names]
        summary_histograms_and_write(file_out, histos, histo_names_period,
                                     histo_xtitles, histo_ytitles)

    file_out.Close()




def make_distributions(args, inv_mass, inv_mass_window): # pylint: disable=too-many-statements

    config = parse_yaml(args.config)

    database_path = config["database"]
    data_or_mc = config["data_or_mc"]
    analysis_name = config["analysis"]
    distribution = config["distribution"]
    distribution_x_range = config["x_range"]
    out_file = config["out_file"]
    # whether or not to slice and derive weights in these slices
    period_cuts = config.get("period_cuts", None)
    slice_cuts = config.get("slice_cuts", None)
    required_columns = config.get("required_columns", None)
    query_all = config.get("query_all", None)
    use_ml_selection = config.get("use_ml_selection", True)
    use_mass_window = config.get("use_mass_window", True)

    # Now open database
    _, database = read_database(database_path)

    analysis_config = database["analysis"][analysis_name]
    inv_mass[0] = database["mass"]

    inv_mass_window[0] = config.get("mass_window", 0.02)

    # required column names
    column_names = ["ev_id", "ev_id_ext", "run_number"]
    column_names.append(distribution)

    # Add column names required by the user
    if required_columns:
        for rcn in required_columns:
            if rcn not in column_names:
                column_names.append(rcn)

    periods = database["multi"][data_or_mc]["period"]

    # is this ML or STD?
    is_ml = database["doml"]

    # No cuts for specific input file
    file_names_cut_map = None

    # Set where to read data from and set overall selection query
    column_names.append("inv_mass")
    trigger_sel = analysis_config["triggersel"]["data"]
    in_top_dirs = database["mlapplication"][data_or_mc]["pkl_skimmed_dec"]
    if trigger_sel:
        if query_all:
            query_all += f" and {trigger_sel}"
        else:
            query_all = trigger_sel

    in_file_name_gen = database["files_names"]["namefile_reco"]
    in_file_name_gen = in_file_name_gen[:in_file_name_gen.find(".")]
    multiclass_labels = database["ml"].get("multiclass_labels", ["", ""])
    queryrunlists = analysis_config[data_or_mc].get("runselection", None)

    if is_ml:
        pkl_extension = ""
        if use_ml_selection:
            model_name = database["mlapplication"]["modelname"]
            ml_sel_column = f"y_test_prob{model_name}"
            ml_sel_column0 = f"y_test_prob{model_name}{multiclass_labels[0]}"
            ml_sel_column1 = f"y_test_prob{model_name}{multiclass_labels[1]}"
            ml_sel_pt = database["mlapplication"]["probcutoptimal"]
            pt_bins_low = database["sel_skim_binmin"]
            pt_bins_up = database["sel_skim_binmax"]
            in_file_names = [f"{in_file_name_gen}{ptl}_{ptu}" \
                    for ptl, ptu in zip(pt_bins_low, pt_bins_up)]
            if not isinstance(ml_sel_pt[0], list):
                file_names_cut_map = {ifn: f"{ml_sel_column} > {cut}" \
                        for ifn, cut in zip(in_file_names, ml_sel_pt)}
                column_names.append(ml_sel_column)
            else:
                file_names_cut_map = {ifn: f"{ml_sel_column0} <= {cut[0]} and {ml_sel_column1} >= {cut[1]}" \
                        for ifn, cut in zip(in_file_names, ml_sel_pt)}
                #if data_or_mc == "data":
                #    file_names_cut_map = {ifn: f"{ml_sel_column0} <= {cut[0]} and {ml_sel_column1} >= {cut[1]}" \
                #            for ifn, cut in zip(in_file_names, ml_sel_pt)}
                #else:
                #    file_names_cut_map = {ifn: f"{ml_sel_column0} <= 1.0 and {ml_sel_column1} >= 0.0" \
                #            for ifn, cut in zip(in_file_names, ml_sel_pt)}
                column_names.append(ml_sel_column0)
                column_names.append(ml_sel_column1)
    else:
        pkl_extension = "_std"

    in_file_name_gen = in_file_name_gen + "*"

    # Now make the directory path right
    in_top_dirs = [f"{itd}{pkl_extension}" for itd in in_top_dirs]

    derive(periods, in_top_dirs, in_file_name_gen, column_names, use_mass_window,
           distribution, distribution_x_range, file_names_cut_map, out_file, queryrunlists,
           period_cuts, query_all, slice_cuts)


def make_weights(args, *ignore): # pylint: disable=unused-argument
    file_data = TFile.Open(args.data, "READ")
    file_mc = TFile.Open(args.mc, "READ")

    keys_data = file_data.GetListOfKeys()
    keys_mc = file_mc.GetListOfKeys()

    out_file_name = f"weights_{args.data}"
    out_file = TFile.Open(out_file_name, "RECREATE")

    def get_mc_histo(histos, period):
        for h in histos:
            if period in h.GetName():
                return h
        sys.exit(1)
        return None

    mc_histos = [k.ReadObj() for k in keys_mc]
    data_histos = [k.ReadObj() for k in keys_data]

    # norm all
    for h in mc_histos:
        if h.GetEntries():
            h.Scale(1. / h.Integral())
    for h in data_histos:
        if h.GetEntries():
            h.Scale(1. / h.Integral())

    for dh in data_histos:
        name = dh.GetName()
        per_pos = name.rfind("_")

        period = name[per_pos:]
        mc_histo = get_mc_histo(mc_histos, period)

        dh.Divide(dh, mc_histo, 1., 1.)
        out_file.cd()
        dh.Write(f"{dh.GetName()}_weights")

    out_file.Close()
    file_data.Close()
    file_mc.Close()


#############
# FUNCTIONS #
#############

def _callback(err):
    print(err)

def multi_proc(function, argument_list, kw_argument_list, maxperchunk, max_n_procs=10):

    chunks_args = [argument_list[x:x+maxperchunk] \
            for x in range(0, len(argument_list), maxperchunk)]
    if not kw_argument_list:
        kw_argument_list = [{} for _ in argument_list]
    chunks_kwargs = [kw_argument_list[x:x+maxperchunk] \
            for x in range(0, len(kw_argument_list), maxperchunk)]
    res_all = []
    for chunk_args, chunk_kwargs in zip(chunks_args, chunks_kwargs):
        print("Processing new chunck size=", maxperchunk)
        pool = mp.Pool(max_n_procs)
        res = [pool.apply_async(function, args=args, kwds=kwds, error_callback=_callback) \
                for args, kwds in zip(chunk_args, chunk_kwargs)]
        pool.close()
        pool.join()
        res_all.extend(res)


    res_list = None
    try:
        res_list = [r.get() for r in res_all]
    except Exception as e: # pylint: disable=broad-except
        print("EXCEPTION")
        print(e)
    return res_list


def fill_from_pickles(file_paths, histo_params, cols=None, query=None, queryrunlist=None, skim_func=None,
                      skim_func_args=None, queries=None, merge_on=None):

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
        runsel2018HMSPD = [287658, 287657, 287656, 287654, 287578, 287575, 287524, 287521, 287518, 287517, 287516, 287513, 287486, 287484, 287481, 287480, 287451, 287413, 287389, 287388, 287387, 287385, 287381, 287380, 287360, 287356, 287355, 287353, 287349, 287347, 287346, 287344, 287343, 287325, 287324, 287323, 287283, 287254, 287251, 287250, 287249, 287248, 287209, 287208, 287204, 287203, 287202, 287201, 287185, 287155, 287137, 287077, 287072, 287071, 287066, 287064, 287063, 287021, 287000, 287977, 287975, 287941, 287923, 287915, 287913, 287912, 287911, 287885, 287884, 287877, 287876, 287784, 287783, 288804, 288806, 288943, 289165, 289166, 289167, 289169, 289172, 289175, 289176, 289177, 289198, 289199, 289200, 289201, 289971, 289966, 289965, 289943, 289941, 289940, 289935, 289931, 289928, 289884, 289880, 289879, 289857, 289856, 289855, 289854, 289852, 289849, 289830, 289818, 289817, 289816, 289815, 289814, 289811, 289808, 289775, 289757, 289732, 289731, 289729, 289724, 289723, 289721, 289666, 289664, 289660, 289659, 289658, 289657, 289634, 289632, 289625, 289582, 289577, 289576, 289574, 289547, 289521, 289494, 289493, 289468, 289466, 289465, 289463, 289462, 289444, 289426, 289374, 289373, 289370, 289369, 289368, 289367, 289366, 289365, 289356, 289355, 289354, 289353, 289309, 289308, 289306, 289303, 289300, 289281, 289280, 289278, 289277, 289276, 289275, 289254, 289253, 289249, 289247, 289243, 289242, 289241, 289240, 289666, 289664, 289660, 289659, 289658, 289657, 289634, 289632, 289625, 289582, 289577, 289576, 289574, 292839, 292836, 292834, 292832, 292831, 292811, 292810, 292809, 292804, 292803, 292754, 292750, 292748, 292747, 292744, 292739, 292737, 292704, 292701, 292698, 292696, 292695, 292693, 292586, 292584, 292563, 292560, 292559, 292557, 292554, 292553, 292526, 292524, 292523, 292521, 292500, 292497, 292496, 292495, 292461, 292460, 292457, 292456, 292434, 292432, 292430, 292429, 292428, 292406, 292405, 292398, 292397, 292298, 292273, 292265, 292242, 292241, 292240, 292218, 292192, 292168, 292167, 292166, 292164, 292163, 292162, 292161, 292160, 292140, 292115, 292114, 292109, 292108, 292107, 292106, 292081, 292080, 292077, 292075, 292067, 292062, 292061, 292060, 292040, 292012, 291982, 291977, 291976, 291953, 291948, 291946, 291945, 291944, 291943, 291942, 291803, 291796, 291795, 291769, 291768, 291766, 291762, 291760, 291756, 291755, 291729, 291706, 291698, 291697, 291690, 291665, 291661, 291657, 291626, 291624, 291622, 291618, 291615, 291614, 291590, 291485, 291484, 291482, 291481, 291457, 291456, 291453, 291451, 291447, 291424, 291420, 291417, 291416, 291402, 291400, 291399, 291397, 291377, 291375, 291373, 291363, 291362, 291361, 291360, 291286, 291285, 291284, 291282, 291266, 291265, 291263, 291262, 291257, 291240, 291209, 291188, 291143, 291116, 291111, 291110, 291101, 291100, 291093, 291069, 291066, 291065, 291041, 291037, 291035, 291006, 291005, 291004, 291003, 291002, 290980, 290979, 290976, 290975, 290974, 290948, 290944, 290943, 290941, 290935, 290932, 290895, 290894, 290888, 290887, 290886, 290862, 290860, 290853, 290848, 290846, 290843, 290841, 290790, 290787, 290766, 290689, 290687, 290665, 290660, 290645, 290632, 290627, 290615, 290614, 290613, 290612, 290590, 290588, 290553, 290550, 290549, 290544, 290540, 290539, 290538, 290501, 290500, 290499, 290469, 290467, 290459, 290458, 290456, 290427, 290426, 290425, 290423, 290412, 290411, 290404, 290401, 290399, 290376, 290375, 290374, 290350, 290327, 290323, 291373, 293898, 293896, 293893, 293891, 293886, 293856, 293831, 293830, 293829, 293809, 293807, 293806, 293805, 293802, 293776, 293774, 293773, 293770, 293741, 293740, 293698, 293696, 293695, 293692, 293691, 293588, 293587, 293583, 293582, 293579, 293578, 293573, 293571, 293570, 293475, 293496, 293494, 293474, 293424, 293413, 293392, 293386, 293368, 294925, 294916, 294884, 294883, 294880, 294875, 294852, 294818, 294817, 294816, 294815, 294813, 294809, 294805, 294775, 294774, 294772, 294769, 294749, 294747, 294746, 294745, 294744, 294742, 294741, 294722, 294718, 294715, 294710, 294703, 294653, 294636, 294633, 294632, 294593, 294591, 294590, 294587, 294586, 294563, 294562, 294558, 294556, 294553, 294531, 294530, 294529, 294527, 294526, 294525, 294524, 294310, 294308, 294307, 294305, 294242, 294241, 294212, 294210, 294208, 294205, 294201, 294200, 294199, 294156, 294155, 294154, 294152, 294131, 294013, 294012, 294011, 294010, 294009]
    #    runlistunique = df.run_number.unique()
    #    for run in runlistunique:
    #        if run not in runsel2018HMSPD:
    #            print(run)
        if queryrunlist == "HighMultSPD2018":
            templen = len(df)
            df = selectdfrunlist(df, runsel2018HMSPD, "run_number")
            if(templen - len(df) > 0):
                print("Rejected cands with run selection: ", templen - len(df))
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






def main():
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest="command", help="[distr, weights]")

    distr_parser = sub_parsers.add_parser("distr")
    distr_parser.add_argument("config", help="configuration to derive distributions")
    distr_parser.set_defaults(func=make_distributions)

    weights_parser = sub_parsers.add_parser("weights")
    weights_parser.add_argument("data", help="ROOT file with data distributions")
    weights_parser.add_argument("mc", help="ROOT file with MC distribution")
    weights_parser.set_defaults(func=make_weights)

    args_parsed = parser.parse_args()

    args_parsed.func(args_parsed, INV_MASS, INV_MASS_WINDOW)


if __name__ == "__main__":
    main()
