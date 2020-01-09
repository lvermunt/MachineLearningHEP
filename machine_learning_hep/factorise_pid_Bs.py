from io import BytesIO
import numba
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from root_numpy import fill_hist
from ROOT import TFile, TH1F

@numba.njit
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

@numba.njit
def selectpid_Bspion(array_nsigma_tpc, array_nsigma_tof, nsigmacut):
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

def plot_dataframe_hist(df, plotname, bins, range, title):
    fig = plt.figure(figsize=(9, 8))
    df.plot.hist(alpha=0.8, bins=bins, range=range)
    plt.title(title);

    plt.subplots_adjust(wspace=0.5)
    savepng = './FactorisePID/%s_pt_cand14_99.png' %(plotname)
    plt.savefig(savepng)
    img_import = BytesIO()
    plt.savefig(img_import, format='png')
    img_import.seek(0)

    savepickle = './FactorisePID/%s_pt_cand14_99.pickle' %(plotname)
    with open(savepickle, 'wb') as fid:
        pickle.dump(fig, fid)

#namedata = "/data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1/ITS2_19h1b2_full/330_20191229-1514/ITS2Upgr_19_bkg_mltot/AnalysisResultsReco_pt_cand14_99.pkl"
#namesig = "/data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1/ITS2_19h1b2_full/331_20191229-2130/ITS2Upgr_19_sig_mltot/AnalysisResultsReco_pt_cand14_99.pkl"
namesig = "/data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1/ITS2_19h1b2_full/331_20191229-2130/skpkldecmerged/AnalysisResultsReco14_99_0.30.pkl"
namedata = "/data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1/ITS2_19h1b2_full/330_20191229-1514/skpkldecmerged/AnalysisResultsReco14_99_0.30.pkl"
#fileout_name = "./FactorisePID/PID_nsigma_distributions_pt_cand14_99.root"
fileout_name = "./FactorisePID/PID_nsigma_distributions_Applied_pt_cand14_99.root"
masspeakmin = 5.371-3*0.04206
masspeakmax = 5.371+3*0.04206
nsigmacut = 3

hs_TPC_Pi_0_bf = TH1F("hs_TPC_Pi_0_bf", "h_TPC_Pi_0", 200, -10, 10)
hs_TPC_K_0_bf = TH1F("hs_TPC_K_0_bf", "h_TPC_K_0", 200, -10, 10)
hs_TOF_Pi_0_bf = TH1F("hs_TOF_Pi_0_bf", "h_TOF_Pi_0", 200, -10, 10)
hs_TOF_K_0_bf = TH1F("hs_TOF_K_0_bf", "h_TOF_K_0", 200, -10, 10)
hs_TPC_K_1_bf = TH1F("hs_TPC_K_1_bf", "h_TPC_K_1", 200, -10, 10)
hs_TOF_K_1_bf = TH1F("hs_TOF_K_1_bf", "h_TOF_K_1", 200, -10, 10)
hs_TPC_Pi_2_bf = TH1F("hs_TPC_Pi_2_bf", "h_TPC_Pi_2", 200, -10, 10)
hs_TPC_K_2_bf = TH1F("hs_TPC_K_2_bf", "h_TPC_K_2", 200, -10, 10)
hs_TOF_Pi_2_bf = TH1F("hs_TOF_Pi_2_bf", "h_TOF_Pi_2", 200, -10, 10)
hs_TOF_K_2_bf = TH1F("hs_TOF_K_2_bf", "h_TOF_K_2", 200, -10, 10)
hs_TPC_Pi_3_bf = TH1F("hs_TPC_Pi_3_bf", "h_TPC_Pi_3", 200, -10, 10)
hs_TOF_Pi_3_bf = TH1F("hs_TOF_Pi_3_bf", "h_TOF_Pi_3", 200, -10, 10)

hs_TPC_Pi_0_af = TH1F("hs_TPC_Pi_0_af", "h_TPC_Pi_0", 200, -10, 10)
hs_TPC_K_0_af = TH1F("hs_TPC_K_0_af", "h_TPC_K_0", 200, -10, 10)
hs_TOF_Pi_0_af = TH1F("hs_TOF_Pi_0_af", "h_TOF_Pi_0", 200, -10, 10)
hs_TOF_K_0_af = TH1F("hs_TOF_K_0_af", "h_TOF_K_0", 200, -10, 10)
hs_TPC_K_1_af = TH1F("hs_TPC_K_1_af", "h_TPC_K_1", 200, -10, 10)
hs_TOF_K_1_af = TH1F("hs_TOF_K_1_af", "h_TOF_K_1", 200, -10, 10)
hs_TPC_Pi_2_af = TH1F("hs_TPC_Pi_2_af", "h_TPC_Pi_2", 200, -10, 10)
hs_TPC_K_2_af = TH1F("hs_TPC_K_2_af", "h_TPC_K_2", 200, -10, 10)
hs_TOF_Pi_2_af = TH1F("hs_TOF_Pi_2_af", "h_TOF_Pi_2", 200, -10, 10)
hs_TOF_K_2_af = TH1F("hs_TOF_K_2_af", "h_TOF_K_2", 200, -10, 10)
hs_TPC_Pi_3_af = TH1F("hs_TPC_Pi_3_af", "h_TPC_Pi_3", 200, -10, 10)
hs_TOF_Pi_3_af = TH1F("hs_TOF_Pi_3_af", "h_TOF_Pi_3", 200, -10, 10)

hb_TPC_Pi_0_bf = TH1F("hb_TPC_Pi_0_bf", "h_TPC_Pi_0", 200, -10, 10)
hb_TPC_K_0_bf = TH1F("hb_TPC_K_0_bf", "h_TPC_K_0", 200, -10, 10)
hb_TOF_Pi_0_bf = TH1F("hb_TOF_Pi_0_bf", "h_TOF_Pi_0", 200, -10, 10)
hb_TOF_K_0_bf = TH1F("hb_TOF_K_0_bf", "h_TOF_K_0", 200, -10, 10)
hb_TPC_K_1_bf = TH1F("hb_TPC_K_1_bf", "h_TPC_K_1", 200, -10, 10)
hb_TOF_K_1_bf = TH1F("hb_TOF_K_1_bf", "h_TOF_K_1", 200, -10, 10)
hb_TPC_Pi_2_bf = TH1F("hb_TPC_Pi_2_bf", "h_TPC_Pi_2", 200, -10, 10)
hb_TPC_K_2_bf = TH1F("hb_TPC_K_2_bf", "h_TPC_K_2", 200, -10, 10)
hb_TOF_Pi_2_bf = TH1F("hb_TOF_Pi_2_bf", "h_TOF_Pi_2", 200, -10, 10)
hb_TOF_K_2_bf = TH1F("hb_TOF_K_2_bf", "h_TOF_K_2", 200, -10, 10)
hb_TPC_Pi_3_bf = TH1F("hb_TPC_Pi_3_bf", "h_TPC_Pi_3", 200, -10, 10)
hb_TOF_Pi_3_bf = TH1F("hb_TOF_Pi_3_bf", "h_TOF_Pi_3", 200, -10, 10)

hb_TPC_Pi_0_af = TH1F("hb_TPC_Pi_0_af", "h_TPC_Pi_0", 200, -10, 10)
hb_TPC_K_0_af = TH1F("hb_TPC_K_0_af", "h_TPC_K_0", 200, -10, 10)
hb_TOF_Pi_0_af = TH1F("hb_TOF_Pi_0_af", "h_TOF_Pi_0", 200, -10, 10)
hb_TOF_K_0_af = TH1F("hb_TOF_K_0_af", "h_TOF_K_0", 200, -10, 10)
hb_TPC_K_1_af = TH1F("hb_TPC_K_1_af", "h_TPC_K_1", 200, -10, 10)
hb_TOF_K_1_af = TH1F("hb_TOF_K_1_af", "h_TOF_K_1", 200, -10, 10)
hb_TPC_Pi_2_af = TH1F("hb_TPC_Pi_2_af", "h_TPC_Pi_2", 200, -10, 10)
hb_TPC_K_2_af = TH1F("hb_TPC_K_2_af", "h_TPC_K_2", 200, -10, 10)
hb_TOF_Pi_2_af = TH1F("hb_TOF_Pi_2_af", "h_TOF_Pi_2", 200, -10, 10)
hb_TOF_K_2_af = TH1F("hb_TOF_K_2_af", "h_TOF_K_2", 200, -10, 10)
hb_TPC_Pi_3_af = TH1F("hb_TPC_Pi_3_af", "h_TPC_Pi_3", 200, -10, 10)
hb_TOF_Pi_3_af = TH1F("hb_TOF_Pi_3_af", "h_TOF_Pi_3", 200, -10, 10)

df_bkg = pickle.load(open(namedata,'rb'))
df_sig = pickle.load(open(namesig,'rb'))

df_bkg = df_bkg.loc[df_bkg["ismcbkg"] == 1]

plot_dataframe_hist(df_sig['inv_mass'], "mass_sig_710_before", 100, [5.0, 5.7], "inv_mass")
plot_dataframe_hist(df_bkg['inv_mass'], "mass_bkg_710_before", 100, [5.0, 5.7], "inv_mass")
plot_dataframe_hist(df_sig['nsigTPC_Pi_0'], "nsigTPC_Pi_0_sig_710_before", 100, [-5.0, 5.0], "nsigTPC_Pi_0")
plot_dataframe_hist(df_bkg['nsigTPC_Pi_0'], "nsigTPC_Pi_0_bkg_710_before", 100, [-5.0, 5.0], "nsigTPC_Pi_0")
plot_dataframe_hist(df_sig['nsigTPC_K_1'], "nsigTPC_K_1_sig_710_before", 100, [-5.0, 5.0], "nsigTPC_K_1")
plot_dataframe_hist(df_bkg['nsigTPC_K_1'], "nsigTPC_K_1_bkg_710_before", 100, [-5.0, 5.0], "nsigTPC_K_1")
plot_dataframe_hist(df_sig['nsigTPC_Pi_3'], "nsigTPC_Pi_3_sig_710_before", 100, [-5.0, 5.0], "nsigTPC_Pi_3")
plot_dataframe_hist(df_bkg['nsigTPC_Pi_3'], "nsigTPC_Pi_3_bkg_710_before", 100, [-5.0, 5.0], "nsigTPC_Pi_3")

df_sig = df_sig.query("inv_mass > @masspeakmin and inv_mass < @masspeakmax")
df_bkg = df_bkg.query("inv_mass > @masspeakmin and inv_mass < @masspeakmax")

plot_dataframe_hist(df_sig['inv_mass'], "mass_sig_710_before_cut", 100, [5.0, 5.7], "inv_mass")
plot_dataframe_hist(df_bkg['inv_mass'], "mass_bkg_710_before_cut", 100, [5.0, 5.7], "inv_mass")
plot_dataframe_hist(df_sig['nsigTPC_Pi_0'], "nsigTPC_Pi_0_sig_710_before_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_0")
plot_dataframe_hist(df_bkg['nsigTPC_Pi_0'], "nsigTPC_Pi_0_bkg_710_before_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_0")
plot_dataframe_hist(df_sig['nsigTPC_K_1'], "nsigTPC_K_1_sig_710_before_cut", 100, [-5.0, 5.0], "nsigTPC_K_1")
plot_dataframe_hist(df_bkg['nsigTPC_K_1'], "nsigTPC_K_1_bkg_710_before_cut", 100, [-5.0, 5.0], "nsigTPC_K_1")
plot_dataframe_hist(df_sig['nsigTPC_Pi_3'], "nsigTPC_Pi_3_sig_710_before_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_3")
plot_dataframe_hist(df_bkg['nsigTPC_Pi_3'], "nsigTPC_Pi_3_bkg_710_before_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_3")

fill_hist(hs_TPC_Pi_0_bf, df_sig.nsigTPC_Pi_0)
fill_hist(hs_TPC_K_0_bf, df_sig.nsigTPC_K_0)
fill_hist(hs_TOF_Pi_0_bf, df_sig.nsigTOF_Pi_0)
fill_hist(hs_TOF_K_0_bf, df_sig.nsigTOF_K_0)
fill_hist(hs_TPC_K_1_bf, df_sig.nsigTPC_K_1)
fill_hist(hs_TOF_K_1_bf, df_sig.nsigTOF_K_1)
fill_hist(hs_TPC_Pi_2_bf, df_sig.nsigTPC_Pi_2)
fill_hist(hs_TPC_K_2_bf, df_sig.nsigTPC_K_2)
fill_hist(hs_TOF_Pi_2_bf, df_sig.nsigTOF_Pi_2)
fill_hist(hs_TOF_K_2_bf, df_sig.nsigTOF_K_2)
fill_hist(hs_TPC_Pi_3_bf, df_sig.nsigTPC_Pi_3)
fill_hist(hs_TOF_Pi_3_bf, df_sig.nsigTOF_Pi_3)
fill_hist(hb_TPC_Pi_0_bf, df_bkg.nsigTPC_Pi_0)
fill_hist(hb_TPC_K_0_bf, df_bkg.nsigTPC_K_0)
fill_hist(hb_TOF_Pi_0_bf, df_bkg.nsigTOF_Pi_0)
fill_hist(hb_TOF_K_0_bf, df_bkg.nsigTOF_K_0)
fill_hist(hb_TPC_K_1_bf, df_bkg.nsigTPC_K_1)
fill_hist(hb_TOF_K_1_bf, df_bkg.nsigTOF_K_1)
fill_hist(hb_TPC_Pi_2_bf, df_bkg.nsigTPC_Pi_2)
fill_hist(hb_TPC_K_2_bf, df_bkg.nsigTPC_K_2)
fill_hist(hb_TOF_Pi_2_bf, df_bkg.nsigTOF_Pi_2)
fill_hist(hb_TOF_K_2_bf, df_bkg.nsigTOF_K_2)
fill_hist(hb_TPC_Pi_3_bf, df_bkg.nsigTPC_Pi_3)
fill_hist(hb_TOF_Pi_3_bf, df_bkg.nsigTOF_Pi_3)

sig_entr_before = len(df_sig)
bkg_entr_before = len(df_bkg)

bkg_array_nsigma_tpc_pi_0 = np.abs(df_bkg['nsigTPC_Pi_0'].to_numpy())
bkg_array_nsigma_tpc_k_0 = np.abs(df_bkg['nsigTPC_K_0'].to_numpy())
bkg_array_nsigma_tof_pi_0 = np.abs(df_bkg['nsigTOF_Pi_0'].to_numpy())
bkg_array_nsigma_tof_k_0 = np.abs(df_bkg['nsigTOF_K_0'].to_numpy())
bkg_array_nsigma_tpc_k_1 = np.abs(df_bkg['nsigTPC_K_1'].to_numpy())
bkg_array_nsigma_tof_k_1 = np.abs(df_bkg['nsigTOF_K_1'].to_numpy())
bkg_array_nsigma_tpc_pi_2 = np.abs(df_bkg['nsigTPC_Pi_2'].to_numpy())
bkg_array_nsigma_tpc_k_2 = np.abs(df_bkg['nsigTPC_K_2'].to_numpy())
bkg_array_nsigma_tof_pi_2 = np.abs(df_bkg['nsigTOF_Pi_2'].to_numpy())
bkg_array_nsigma_tof_k_2 = np.abs(df_bkg['nsigTOF_K_2'].to_numpy())
bkg_array_nsigma_tpc_pi_3 = np.abs(df_bkg['nsigTPC_Pi_3'].to_numpy())
bkg_array_nsigma_tof_pi_3 = np.abs(df_bkg['nsigTOF_Pi_3'].to_numpy())

sig_array_nsigma_tpc_pi_0 = np.abs(df_sig['nsigTPC_Pi_0'].to_numpy())
sig_array_nsigma_tpc_k_0 = np.abs(df_sig['nsigTPC_K_0'].to_numpy())
sig_array_nsigma_tof_pi_0 = np.abs(df_sig['nsigTOF_Pi_0'].to_numpy())
sig_array_nsigma_tof_k_0 = np.abs(df_sig['nsigTOF_K_0'].to_numpy())
sig_array_nsigma_tpc_k_1 = np.abs(df_sig['nsigTPC_K_1'].to_numpy())
sig_array_nsigma_tof_k_1 = np.abs(df_sig['nsigTOF_K_1'].to_numpy())
sig_array_nsigma_tpc_pi_2 = np.abs(df_sig['nsigTPC_Pi_2'].to_numpy())
sig_array_nsigma_tpc_k_2 = np.abs(df_sig['nsigTPC_K_2'].to_numpy())
sig_array_nsigma_tof_pi_2 = np.abs(df_sig['nsigTOF_Pi_2'].to_numpy())
sig_array_nsigma_tof_k_2 = np.abs(df_sig['nsigTOF_K_2'].to_numpy())
sig_array_nsigma_tpc_pi_3 = np.abs(df_sig['nsigTPC_Pi_3'].to_numpy())
sig_array_nsigma_tof_pi_3 = np.abs(df_sig['nsigTOF_Pi_3'].to_numpy())

bkg_isDssel = selectpid_dstokkpi(bkg_array_nsigma_tpc_pi_0, bkg_array_nsigma_tpc_k_0, \
                  bkg_array_nsigma_tof_pi_0, bkg_array_nsigma_tof_k_0, \
                      bkg_array_nsigma_tpc_k_1, bkg_array_nsigma_tof_k_1, \
                          bkg_array_nsigma_tpc_pi_2, bkg_array_nsigma_tpc_k_2, \
                              bkg_array_nsigma_tof_pi_2, bkg_array_nsigma_tof_k_2, nsigmacut)
sig_isDssel = selectpid_dstokkpi(sig_array_nsigma_tpc_pi_0, sig_array_nsigma_tpc_k_0, \
                  sig_array_nsigma_tof_pi_0, sig_array_nsigma_tof_k_0, \
                      sig_array_nsigma_tpc_k_1, sig_array_nsigma_tof_k_1, \
                          sig_array_nsigma_tpc_pi_2, sig_array_nsigma_tpc_k_2, \
                              sig_array_nsigma_tof_pi_2, sig_array_nsigma_tof_k_2, nsigmacut)
bkg_isPisel = selectpid_Bspion(bkg_array_nsigma_tpc_pi_3, bkg_array_nsigma_tof_pi_3, nsigmacut)
sig_isPisel = selectpid_Bspion(sig_array_nsigma_tpc_pi_3, sig_array_nsigma_tof_pi_3, nsigmacut)

sig_entr_after_Ds = np.sum(sig_isDssel)
bkg_entr_after_Ds = np.sum(bkg_isDssel)
sig_entr_after_pi = np.sum(sig_isPisel)
bkg_entr_after_pi = np.sum(bkg_isPisel)
sig_isDPisel = np.multiply(sig_isDssel, sig_isPisel)
bkg_isDPisel = np.multiply(bkg_isDssel, bkg_isPisel)
sig_entr_after_Dspi = np.sum(sig_isDPisel)
bkg_entr_after_Dspi = np.sum(bkg_isDPisel)

print("Signal.     Before:", sig_entr_before, "After Ds:", sig_entr_after_Ds, "After pi:", sig_entr_after_pi, "After Dspi:", sig_entr_after_Dspi)
print("Background. Before:", bkg_entr_before, "After Ds:", bkg_entr_after_Ds, "After pi:", bkg_entr_after_pi, "After Dspi:", bkg_entr_after_Dspi)

df_sig = df_sig[sig_isDPisel]
df_bkg = df_bkg[bkg_isDPisel]
print("Check length. Signal:", len(df_sig), "Background:", len(df_bkg))

plot_dataframe_hist(df_sig['inv_mass'], "mass_sig_710_after_cut", 100, [5.0, 5.7], "inv_mass")
plot_dataframe_hist(df_bkg['inv_mass'], "mass_bkg_710_after_cut", 100, [5.0, 5.7], "inv_mass")
plot_dataframe_hist(df_sig['nsigTPC_Pi_0'], "nsigTPC_Pi_0_sig_710_after_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_0")
plot_dataframe_hist(df_bkg['nsigTPC_Pi_0'], "nsigTPC_Pi_0_bkg_710_after_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_0")
plot_dataframe_hist(df_sig['nsigTPC_K_1'], "nsigTPC_K_1_sig_710_after_cut", 100, [-5.0, 5.0], "nsigTPC_K_1")
plot_dataframe_hist(df_bkg['nsigTPC_K_1'], "nsigTPC_K_1_bkg_710_after_cut", 100, [-5.0, 5.0], "nsigTPC_K_1")
plot_dataframe_hist(df_sig['nsigTPC_Pi_3'], "nsigTPC_Pi_3_sig_710_after_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_3")
plot_dataframe_hist(df_bkg['nsigTPC_Pi_3'], "nsigTPC_Pi_3_bkg_710_after_cut", 100, [-5.0, 5.0], "nsigTPC_Pi_3")

fill_hist(hs_TPC_Pi_0_af, df_sig.nsigTPC_Pi_0)
fill_hist(hs_TPC_K_0_af, df_sig.nsigTPC_K_0)
fill_hist(hs_TOF_Pi_0_af, df_sig.nsigTOF_Pi_0)
fill_hist(hs_TOF_K_0_af, df_sig.nsigTOF_K_0)
fill_hist(hs_TPC_K_1_af, df_sig.nsigTPC_K_1)
fill_hist(hs_TOF_K_1_af, df_sig.nsigTOF_K_1)
fill_hist(hs_TPC_Pi_2_af, df_sig.nsigTPC_Pi_2)
fill_hist(hs_TPC_K_2_af, df_sig.nsigTPC_K_2)
fill_hist(hs_TOF_Pi_2_af, df_sig.nsigTOF_Pi_2)
fill_hist(hs_TOF_K_2_af, df_sig.nsigTOF_K_2)
fill_hist(hs_TPC_Pi_3_af, df_sig.nsigTPC_Pi_3)
fill_hist(hs_TOF_Pi_3_af, df_sig.nsigTOF_Pi_3)
fill_hist(hb_TPC_Pi_0_af, df_bkg.nsigTPC_Pi_0)
fill_hist(hb_TPC_K_0_af, df_bkg.nsigTPC_K_0)
fill_hist(hb_TOF_Pi_0_af, df_bkg.nsigTOF_Pi_0)
fill_hist(hb_TOF_K_0_af, df_bkg.nsigTOF_K_0)
fill_hist(hb_TPC_K_1_af, df_bkg.nsigTPC_K_1)
fill_hist(hb_TOF_K_1_af, df_bkg.nsigTOF_K_1)
fill_hist(hb_TPC_Pi_2_af, df_bkg.nsigTPC_Pi_2)
fill_hist(hb_TPC_K_2_af, df_bkg.nsigTPC_K_2)
fill_hist(hb_TOF_Pi_2_af, df_bkg.nsigTOF_Pi_2)
fill_hist(hb_TOF_K_2_af, df_bkg.nsigTOF_K_2)
fill_hist(hb_TPC_Pi_3_af, df_bkg.nsigTPC_Pi_3)
fill_hist(hb_TOF_Pi_3_af, df_bkg.nsigTOF_Pi_3)

myfile = TFile(fileout_name, "RECREATE")
myfile.cd()
hs_TPC_Pi_0_bf.Write()
hs_TPC_K_0_bf.Write()
hs_TOF_Pi_0_bf.Write()
hs_TOF_K_0_bf.Write()
hs_TPC_K_1_bf.Write()
hs_TOF_K_1_bf.Write()
hs_TPC_Pi_2_bf.Write()
hs_TPC_K_2_bf.Write()
hs_TOF_Pi_2_bf.Write()
hs_TOF_K_2_bf.Write()
hs_TPC_Pi_3_bf.Write()
hs_TOF_Pi_3_bf.Write()
hs_TPC_Pi_0_af.Write()
hs_TPC_K_0_af.Write()
hs_TOF_Pi_0_af.Write()
hs_TOF_K_0_af.Write()
hs_TPC_K_1_af.Write()
hs_TOF_K_1_af.Write()
hs_TPC_Pi_2_af.Write()
hs_TPC_K_2_af.Write()
hs_TOF_Pi_2_af.Write()
hs_TOF_K_2_af.Write()
hs_TPC_Pi_3_af.Write()
hs_TOF_Pi_3_af.Write()
hb_TPC_Pi_0_bf.Write()
hb_TPC_K_0_bf.Write()
hb_TOF_Pi_0_bf.Write()
hb_TOF_K_0_bf.Write()
hb_TPC_K_1_bf.Write()
hb_TOF_K_1_bf.Write()
hb_TPC_Pi_2_bf.Write()
hb_TPC_K_2_bf.Write()
hb_TOF_Pi_2_bf.Write()
hb_TOF_K_2_bf.Write()
hb_TPC_Pi_3_bf.Write()
hb_TOF_Pi_3_bf.Write()
hb_TPC_Pi_0_af.Write()
hb_TPC_K_0_af.Write()
hb_TOF_Pi_0_af.Write()
hb_TOF_K_0_af.Write()
hb_TPC_K_1_af.Write()
hb_TOF_K_1_af.Write()
hb_TPC_Pi_2_af.Write()
hb_TPC_K_2_af.Write()
hb_TOF_Pi_2_af.Write()
hb_TOF_K_2_af.Write()
hb_TPC_Pi_3_af.Write()
hb_TOF_Pi_3_af.Write()
myfile.Close()

