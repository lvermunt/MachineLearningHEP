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
utilities for bit selection, fiducial acceptance, pid, single topological variable selections and normalization
"""

import operator
from functools import reduce

import numba
import numpy as np
import pandas as pd
from ROOT import TH1F  # pylint: disable=import-error, no-name-in-module


def selectdfquery(dfr, selection):
    """
    Query on dataframe
    """
    if selection is not None:
        dfr = dfr.query(selection)
    return dfr

def selectdfrunlist(dfr, runlist, runvar):
    """
    Select smaller runlist on dataframe
    """
    if runlist is not None:
        runlist_np = np.asarray(runlist)
        array_run_np = np.asarray(dfr[runvar].values)
        issel = select_runs(runlist_np, array_run_np)
        dfr = dfr[issel]
    return dfr

def seldf_singlevar(dataframe, var, minval, maxval):
    """
    Make projection on variable using [X,Y), e.g. pT or multiplicity
    """
    dataframe = dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] < maxval)]
    return dataframe

def seldf_singlevar_inclusive(dataframe, var, minval, maxval):
    """
    Make projection on variable using [X,Y), e.g. pT or multiplicity
    """
    dataframe = dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] <= maxval)]
    return dataframe

def split_df_sigbkg(dataframe_, var_signal_):
    """
    Split dataframe in signal and background dataframes
    """
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_

def selectbiton(array_cand_type, mask):
    return [((cand_type & mask) == mask) for cand_type in array_cand_type]

def selectbitoff(array_cand_type, mask):
    return [((cand_type & mask) == 0) for cand_type in array_cand_type]

def tag_bit_df(dfin, namebitmap, activatedbit):
    bitson = activatedbit[0]
    bitsoff = activatedbit[1]
    array_cand_type = dfin.loc[:, namebitmap].values.astype("int")
    res_on = pd.Series([True]*len(array_cand_type))
    res_off = pd.Series([True]*len(array_cand_type))
    res = pd.Series()

    if bitson:
        mask = reduce(operator.or_, ((1 << bit) for bit in bitson), 0)
        bitmapon = selectbiton(array_cand_type, mask)
        res_on = pd.Series(bitmapon)
    if bitsoff:
        mask = reduce(operator.or_, ((1 << bit) for bit in bitsoff), 0)
        bitmapoff = selectbitoff(array_cand_type, mask)
        res_off = pd.Series(bitmapoff)
    res = res_on & res_off
    return res

def filter_bit_df(dfin, namebitmap, activatedbit):
    res = tag_bit_df(dfin, namebitmap, activatedbit)
    df_sel = dfin[res.values]
    return df_sel

def selectcandidateml(array_prob, probcut):
    array_is_sel = []
    for prob in array_prob:
        if prob > probcut:
            array_is_sel.append(True)
        else:
            array_is_sel.append(False)
    return array_is_sel

@numba.njit
def select_runs(good_runlist, array_run):
    array_run_sel = np.zeros(len(array_run), np.bool_)
    for i, candrun in np.ndenumerate(array_run):
        for _, goodrun in np.ndenumerate(good_runlist):
            if candrun == goodrun:
                array_run_sel[i] = True
                break
    return array_run_sel

def selectfidacc(array_pt, array_y):
    array_is_sel = []
    for icand, pt in enumerate(array_pt):
        if pt > 5:
            if abs(array_y[icand]) < 0.8:
                array_is_sel.append(True)
            else:
                array_is_sel.append(False)
        else:
            yfid = -0.2/15 * pt**2 + 1.9/15 * pt + 0.5
            if abs(array_y[icand]) < yfid:
                array_is_sel.append(True)
            else:
                array_is_sel.append(False)
    return array_is_sel

# pylint: disable=too-many-arguments
def selectpid_dstokkpi(array_nsigma_tpc_pi_0, array_nsigma_tpc_k_0, \
    array_nsigma_tof_pi_0, array_nsigma_tof_k_0, \
        array_nsigma_tpc_k_1, array_nsigma_tof_k_1, \
            array_nsigma_tpc_pi_2, array_nsigma_tpc_k_2, \
                array_nsigma_tof_pi_2, array_nsigma_tof_k_2, nsigmacut):

    array_is_pid_sel = []

    for icand, _ in enumerate(array_nsigma_tpc_pi_0):
        is_track_0_sel = array_nsigma_tpc_pi_0[icand] < nsigmacut \
            or array_nsigma_tof_pi_0[icand] < nsigmacut \
                or array_nsigma_tpc_k_0[icand] < nsigmacut \
                    or array_nsigma_tof_k_0[icand] < nsigmacut
        #second track must be a kaon
        is_track_1_sel = array_nsigma_tpc_k_1[icand] < nsigmacut \
            or array_nsigma_tof_k_1[icand] < nsigmacut
        is_track_2_sel = array_nsigma_tpc_pi_2[icand] < nsigmacut \
            or array_nsigma_tof_pi_2[icand] < nsigmacut \
                or array_nsigma_tpc_k_2[icand] < nsigmacut \
                    or array_nsigma_tof_k_2[icand] < nsigmacut
        if is_track_0_sel and is_track_1_sel and is_track_2_sel:
            array_is_pid_sel.append(True)
        else:
            array_is_pid_sel.append(False)
    return array_is_pid_sel

def selectpid_dzerotokpi(array_nsigma_tpc_pi_0, array_nsigma_tpc_k_0, \
    array_nsigma_tof_pi_0, array_nsigma_tof_k_0, \
        array_nsigma_tpc_pi_1, array_nsigma_tpc_k_1, \
            array_nsigma_tof_pi_1, array_nsigma_tof_k_1, nsigmacut):

    array_is_pid_sel = []

    for icand, _ in enumerate(array_nsigma_tpc_pi_0):
        is_track_0_sel = array_nsigma_tpc_pi_0[icand] < nsigmacut \
            or array_nsigma_tof_pi_0[icand] < nsigmacut \
                or array_nsigma_tpc_k_0[icand] < nsigmacut \
                    or array_nsigma_tof_k_0[icand] < nsigmacut
        is_track_1_sel = array_nsigma_tpc_pi_1[icand] < nsigmacut \
            or array_nsigma_tof_pi_1[icand] < nsigmacut \
                or array_nsigma_tpc_k_1[icand] < nsigmacut \
                    or array_nsigma_tof_k_1[icand] < nsigmacut
        if is_track_0_sel and is_track_1_sel:
            array_is_pid_sel.append(True)
        else:
            array_is_pid_sel.append(False)
    return array_is_pid_sel

def selectpid_lctov0bachelor(array_nsigma_tpc, array_nsigma_tof, nsigmacut):
    #nsigma for desired species (i.e. p in case of pK0s or pi in case of piL)
    array_is_pid_sel = []

    for icand, _ in enumerate(array_nsigma_tpc):
        is_track_sel = array_nsigma_tpc[icand] < nsigmacut or \
            array_nsigma_tof[icand] < nsigmacut
        if is_track_sel:
            array_is_pid_sel.append(True)
        else:
            array_is_pid_sel.append(False)
    return array_is_pid_sel

def selectcand_lincut(array_cut_var, minvalue, maxvalue, isabs):
    array_is_sel = []
    for icand, _ in enumerate(array_cut_var):
        if isabs:
            value = abs(array_cut_var[icand])
        else:
            value = array_cut_var[icand]
        if minvalue < value < maxvalue:
            array_is_sel.append(True)
        else:
            array_is_sel.append(False)
    return array_is_sel

def getnormforselevt(df_evt):
    #accepted events
    df_acc_ev = df_evt.query('is_ev_rej==0')
    #rejected events because of trigger / physics selection / centrality
    df_to_keep = filter_bit_df(df_evt, 'is_ev_rej', [[], [0, 5, 6, 10, 11]])

    #events with reco vtx after previous selection
    df_bit_recovtx = filter_bit_df(df_to_keep, 'is_ev_rej', [[], [1, 2, 7, 12]])
    #events with reco zvtx > 10 cm after previous selection
    df_bit_zvtx_gr10 = filter_bit_df(df_to_keep, 'is_ev_rej', [[3], [1, 2, 7, 12]])

    n_no_reco_vtx = len(df_to_keep.index)-len(df_bit_recovtx.index)
    n_zvtx_gr10 = len(df_bit_zvtx_gr10.index)
    n_ev_sel = len(df_acc_ev.index)

    return (n_ev_sel+n_no_reco_vtx) - n_no_reco_vtx*n_zvtx_gr10 / (n_ev_sel+n_zvtx_gr10)

def gethistonormforselevt(df_evt, dfevtevtsel, label):
    hSelMult = TH1F('sel_' + label, 'sel_' + label, 1, -0.5, 0.5)
    hNoVtxMult = TH1F('novtx_' + label, 'novtx_' + label, 1, -0.5, 0.5)
    hVtxOutMult = TH1F('vtxout_' + label, 'vtxout_' + label, 1, -0.5, 0.5)

    df_to_keep = filter_bit_df(df_evt, 'is_ev_rej', [[], [0, 5, 6, 10, 11]])
    # events with reco vtx after previous selection
    tag_vtx = tag_bit_df(df_to_keep, 'is_ev_rej', [[], [1, 2, 7, 12]])
    df_no_vtx = df_to_keep[~tag_vtx.values]
    # events with reco zvtx > 10 cm after previous selection
    df_bit_zvtx_gr10 = filter_bit_df(df_to_keep, 'is_ev_rej', [[3], [1, 2, 7, 12]])

    hSelMult.SetBinContent(1, len(dfevtevtsel))
    hNoVtxMult.SetBinContent(1, len(df_no_vtx))
    hVtxOutMult.SetBinContent(1, len(df_bit_zvtx_gr10))
    return hSelMult, hNoVtxMult, hVtxOutMult
