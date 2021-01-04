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
import math
import array
import pickle
import os
import numpy as np
from root_numpy import fill_hist  # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.bitwise import tag_bit_df
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities import seldf_singlevar, openfile
from machine_learning_hep.selectionutils import gethistonormforselevt
from machine_learning_hep.processer import Processer

class ProcesserDhadrons(Processer): # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                         d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                         p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                         p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                         d_results, typean, runlisttrigger, d_mcreweights)

        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.s_presel_gen_eff = datap["analysis"][self.typean]['presel_gen_eff']


        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.lpt_mergebinmin = datap["analysis"][self.typean].get("sel_anmerge_binmin",
                                                                  "sel_an_binmin")
        self.lpt_mergebinmax = datap["analysis"][self.typean].get("sel_anmerge_binmax",
                                                                  "sel_an_binmax")
        self.p_nptmergebins = len(self.lpt_mergebinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.runlistrigger = runlisttrigger

        # Event re-weighting MC
        self.apply_pt_weights_eff = datap["analysis"][self.typean].get("apply_pt_weights", False)
        self.v_var_pt_weights_eff = datap["analysis"][self.typean].get("var_pt_weights", "pt_cand")
        self.event_weighting_mc = datap["analysis"][self.typean].get("event_weighting_mc", {})
        self.event_weighting_mc = self.event_weighting_mc.get(self.period, {})

    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        neventsorig = len(dfevtorig)
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        neventsaftertrigger = len(dfevtorig)
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, \
                             self.run_param[self.runlistrigger], "run_number")
        neventsafterrunsel = len(dfevtorig)
        dfevtevtsel = dfevtorig.query(self.s_evtsel)
        neventsafterevtsel = len(dfevtevtsel)

        #validation plot for event selection
        histonorm = TH1F("histonorm", "histonorm", 10, 0, 10)
        histonorm.SetBinContent(1, neventsorig)
        histonorm.GetXaxis().SetBinLabel(1, "tot events")
        histonorm.SetBinContent(2, neventsaftertrigger)
        histonorm.GetXaxis().SetBinLabel(2, "tot events after trigger")
        histonorm.SetBinContent(3, neventsafterrunsel)
        histonorm.GetXaxis().SetBinLabel(3, "tot events after run sel")
        histonorm.SetBinContent(4, neventsafterevtsel)
        histonorm.GetXaxis().SetBinLabel(4, "tot events after evt sel")
        histonorm.Write()

        myfile.cd()
        labeltrigger = "hbit%s" % (self.triggerbit)
        hsel, hnovtxmult, hvtxoutmult = gethistonormforselevt(dfevtorig, dfevtevtsel, \
                                                              labeltrigger)
        hsel.Write()
        hnovtxmult.Write()
        hvtxoutmult.Write()

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]

            if self.appliedongrid is False:
                df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            else:
                df = pickle.load(openfile(self.mptfiles_recosk[bin_id][index], "rb"))

            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger is not None:
                df = df.query(self.s_trigger)
            if self.runlistrigger is not None:
                df = selectdfrunlist(df, \
                    self.run_param[self.runlistrigger], "run_number")

            if self.doml is True or self.appliedongrid is True:
                df = df.query(self.l_selml[bin_id])
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            if self.do_custom_analysis_cuts:
                df = self.apply_cuts_ptbin(df, ipt)

            if self.mltype == "MultiClassification":
                suffix = "%s%d_%d_%.2f%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt][0],
                          self.lpt_probcutfin[ipt][1])
            else:
                suffix = "%s%d_%d_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt])

            h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                             self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            fill_hist(h_invmass, df.inv_mass)
            myfile.cd()
            h_invmass.Write()

            if self.mcordata == "mc":
                df[self.v_ismcrefl] = np.array(tag_bit_df(df, self.v_bitvar,
                                                          self.b_mcrefl), dtype=int)
                df_sig = df[df[self.v_ismcsignal] == 1]
                df_bkg = df[df[self.v_ismcbkg] == 1]
                df_refl = df[df[self.v_ismcrefl] == 1]
                h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_bkg = TH1F("hmass_bkg" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_refl = TH1F("hmass_refl" + suffix, "", self.p_num_bins,
                                      self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass_sig, df_sig.inv_mass)
                fill_hist(h_invmass_bkg, df_bkg.inv_mass)
                fill_hist(h_invmass_refl, df_refl.inv_mass)
                myfile.cd()
                h_invmass_sig.Write()
                h_invmass_bkg.Write()
                h_invmass_refl.Write()

    # pylint: disable=too-many-branches
    def process_histomass_merge_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        neventsorig = len(dfevtorig)
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        neventsaftertrigger = len(dfevtorig)
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, \
                             self.run_param[self.runlistrigger], "run_number")
        neventsafterrunsel = len(dfevtorig)
        dfevtevtsel = dfevtorig.query(self.s_evtsel)
        neventsafterevtsel = len(dfevtevtsel)

        #validation plot for event selection
        histonorm = TH1F("histonorm", "histonorm", 10, 0, 10)
        histonorm.SetBinContent(1, neventsorig)
        histonorm.GetXaxis().SetBinLabel(1, "tot events")
        histonorm.SetBinContent(2, neventsaftertrigger)
        histonorm.GetXaxis().SetBinLabel(2, "tot events after trigger")
        histonorm.SetBinContent(3, neventsafterrunsel)
        histonorm.GetXaxis().SetBinLabel(3, "tot events after run sel")
        histonorm.SetBinContent(4, neventsafterevtsel)
        histonorm.GetXaxis().SetBinLabel(4, "tot events after evt sel")
        histonorm.Write()

        myfile.cd()
        labeltrigger = "hbit%s" % (self.triggerbit)
        hsel, hnovtxmult, hvtxoutmult = gethistonormforselevt(dfevtorig, dfevtevtsel, \
                                                              labeltrigger)
        hsel.Write()
        hnovtxmult.Write()
        hvtxoutmult.Write()

        for iptm in range(self.p_nptfinbins):
            if self.mltype == "MultiClassification":
                suffix = "%s%d_%d_%.2f%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[iptm],
                          self.lpt_finbinmax[iptm], self.lpt_probcutfin[iptm][0],
                          self.lpt_probcutfin[iptm][1])
            else:
                suffix = "%s%d_%d_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[iptm],
                          self.lpt_finbinmax[iptm], self.lpt_probcutfin[iptm])

            h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                             self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            h_invmass_bkg = TH1F("hmass_bkg" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            h_invmass_refl = TH1F("hmass_refl" + suffix, "", self.p_num_bins,
                                  self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])

            for ipt in range(self.p_nptmergebins):
                bin_id = self.bin_matching[ipt]

                #if self.lpt_mergebinmin[ipt] != self.lpt_finbinmin[iptm] and \
                #  self.lpt_mergebinmax[ipt] != self.lpt_finbinmax[iptm]:
                #    continue

                if self.lpt_mergebinmin[ipt] < self.lpt_finbinmin[iptm] or \
                  self.lpt_mergebinmax[ipt] > self.lpt_finbinmax[iptm]:
                    continue

                if self.appliedongrid is False:
                    df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
                else:
                    df = pickle.load(openfile(self.mptfiles_recosk[bin_id][index], "rb"))

                if self.s_evtsel is not None:
                    df = df.query(self.s_evtsel)
                if self.s_trigger is not None:
                    df = df.query(self.s_trigger)
                if self.runlistrigger is not None:
                    df = selectdfrunlist(df, \
                        self.run_param[self.runlistrigger], "run_number")

                if self.doml is True or self.appliedongrid is True:
                    df = df.query(self.l_selml[bin_id])

                df = seldf_singlevar(df, self.v_var_binning, \
                                     self.lpt_mergebinmin[ipt], self.lpt_mergebinmax[ipt])

                if self.do_custom_analysis_cuts:
                    df = self.apply_cuts_ptbin(df, iptm)

                fill_hist(h_invmass, df.inv_mass)
                myfile.cd()

                if self.mcordata == "mc":
                    df[self.v_ismcrefl] = np.array(tag_bit_df(df, self.v_bitvar,
                                                              self.b_mcrefl), dtype=int)
                    df_sig = df[df[self.v_ismcsignal] == 1]
                    df_bkg = df[df[self.v_ismcbkg] == 1]
                    df_refl = df[df[self.v_ismcrefl] == 1]
                    fill_hist(h_invmass_sig, df_sig.inv_mass)
                    fill_hist(h_invmass_bkg, df_bkg.inv_mass)
                    fill_hist(h_invmass_refl, df_refl.inv_mass)
                    myfile.cd()

            h_invmass.Write()
            if self.mcordata == "mc":
                h_invmass_sig.Write()
                h_invmass_bkg.Write()
                h_invmass_refl.Write()

    # pylint: disable=too-many-branches
    def process_histomass_scan_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        neventsorig = len(dfevtorig)
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        neventsaftertrigger = len(dfevtorig)
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, \
                             self.run_param[self.runlistrigger], "run_number")
        neventsafterrunsel = len(dfevtorig)
        dfevtevtsel = dfevtorig.query(self.s_evtsel)
        neventsafterevtsel = len(dfevtevtsel)

        #validation plot for event selection
        histonorm = TH1F("histonorm", "histonorm", 10, 0, 10)
        histonorm.SetBinContent(1, neventsorig)
        histonorm.GetXaxis().SetBinLabel(1, "tot events")
        histonorm.SetBinContent(2, neventsaftertrigger)
        histonorm.GetXaxis().SetBinLabel(2, "tot events after trigger")
        histonorm.SetBinContent(3, neventsafterrunsel)
        histonorm.GetXaxis().SetBinLabel(3, "tot events after run sel")
        histonorm.SetBinContent(4, neventsafterevtsel)
        histonorm.GetXaxis().SetBinLabel(4, "tot events after evt sel")
        histonorm.Write()

        myfile.cd()
        labeltrigger = "hbit%s" % (self.triggerbit)
        hsel, hnovtxmult, hvtxoutmult = gethistonormforselevt(dfevtorig, dfevtevtsel, \
                                                              labeltrigger)
        hsel.Write()
        hnovtxmult.Write()
        hvtxoutmult.Write()

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]

            if self.appliedongrid is False:
                df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            else:
                df = pickle.load(openfile(self.mptfiles_recosk[bin_id][index], "rb"))

            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger is not None:
                df = df.query(self.s_trigger)
            if self.runlistrigger is not None:
                df = selectdfrunlist(df, \
                    self.run_param[self.runlistrigger], "run_number")

            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            if self.do_custom_analysis_cuts:
                df = self.apply_cuts_ptbin(df, ipt)

            for iscan in range(self.p_nmlscanbins):
                if self.doml is True or self.appliedongrid is True:
                    df = df.query(self.l_selml_scan[iscan])

                if self.mltype == "MultiClassification":
                    suffix = "%s%d_%d_%.2f%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], self.lpt_probcut_scan[iscan][0],
                              self.lpt_probcut_scan[iscan][1])
                else:
                    suffix = "%s%d_%d_%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], self.lpt_probcut_scan[iscan])

                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass, df.inv_mass)
                myfile.cd()
                h_invmass.Write()

                if self.mcordata == "mc":
                    if not self.v_ismcrefl in df.columns:
                        df[self.v_ismcrefl] = np.array(tag_bit_df(df, self.v_bitvar,
                                                                  self.b_mcrefl), dtype=int)
                    df_sig = df[df[self.v_ismcsignal] == 1]
                    df_bkg = df[df[self.v_ismcbkg] == 1]
                    df_refl = df[df[self.v_ismcrefl] == 1]
                    h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                         self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    h_invmass_bkg = TH1F("hmass_bkg" + suffix, "", self.p_num_bins,
                                         self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    h_invmass_refl = TH1F("hmass_refl" + suffix, "", self.p_num_bins,
                                          self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    fill_hist(h_invmass_sig, df_sig.inv_mass)
                    fill_hist(h_invmass_bkg, df_bkg.inv_mass)
                    fill_hist(h_invmass_refl, df_refl.inv_mass)
                    myfile.cd()
                    h_invmass_sig.Write()
                    h_invmass_bkg.Write()
                    h_invmass_refl.Write()


    def get_reweighted_count(self, dfsel):
        """Apply event weights
        Args:
            dfsel: pandas.DataFrame
                dataframe with column to apply weights for
        Returns:
            float: nominal value,
            float: error
        """

        def no_weights(df_):
            val = len(df_)
            return val, math.sqrt(val)

        event_weighting_mc = {}
        if self.event_weighting_mc:
            # Check is there is a dictionary with desired info
            event_weighting_mc = self.event_weighting_mc[0]

        # If there were explicit info in the analysis database, assume that all fields exist
        # If incomplete, there will be a mix-up between these values and default values
        filepath = event_weighting_mc.get("filepath", os.path.join(self.d_mcreweights,
                                                                   self.n_mcreweights))
        if not os.path.exists(filepath):
            print(f"Could not find filepath {filepath} for MC event weighting." \
                    "Compute unweighted values...")
            return no_weights(dfsel)

        weight_file = TFile.Open(filepath, "read")
        histo_name = event_weighting_mc.get("histo_name", "Weights0")
        weights = weight_file.Get(histo_name)

        if not weights:
            print(f"Could not find histogram {histo_name} for MC event weighting." \
                    "Compute unweighted values...")
            return no_weights(dfsel)

        weight_according_to = event_weighting_mc.get("according_to", self.v_var_pt_weights_eff)

        w = [weights.GetBinContent(weights.FindBin(v)) for v in
             dfsel[weight_according_to]]
        val = sum(w)
        err = math.sqrt(sum(map(lambda i: i * i, w)))
        #print('reweighting sum: {:.1f} +- {:.1f} -> {:.1f} +- {:.1f} (zeroes: {})' \
        #      .format(len(dfsel), math.sqrt(len(dfsel)), val, err, w.count(0.)))
        return val, err


    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):

        out_file = TFile.Open(self.l_histoeff[index], "recreate")

        n_bins = len(self.lpt_finbinmin)
        analysis_bin_lims_temp = self.lpt_finbinmin.copy()
        analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
        analysis_bin_lims = array.array('f', analysis_bin_lims_temp)

        def make_histo(name, title,
                       bins=n_bins,
                       binning=analysis_bin_lims):
            histo = TH1F(name, title, bins, binning)
            return histo

        h_gen_pr = make_histo("h_gen_pr",
                              "Prompt Generated in acceptance |y|<0.5")
        h_presel_pr = make_histo("h_presel_pr",
                                 "Prompt Reco in acc |#eta|<0.8 and sel")
        h_sel_pr = make_histo("h_sel_pr",
                              "Prompt Reco and sel in acc |#eta|<0.8 and sel")

        h_gen_fd = make_histo("h_gen_fd",
                              "FD Generated in acceptance |y|<0.5")
        h_presel_fd = make_histo("h_presel_fd",
                                 "FD Reco in acc |#eta|<0.8 and sel")
        h_sel_fd = make_histo("h_sel_fd",
                              "FD Reco and sel in acc |#eta|<0.8 and sel")

        bincounter = 0
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df_mc_reco = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            if self.runlistrigger is not None:
                df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger], "run_number")

            if self.do_custom_analysis_cuts:
                df_mc_reco = self.apply_cuts_ptbin(df_mc_reco, ipt)

            df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            if self.runlistrigger is not None:
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                         self.run_param[self.runlistrigger], "run_number")
            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
            df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
            df_reco_sel_pr = None
            if self.doml is True:
                df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[bin_id])
            else:
                df_reco_sel_pr = df_reco_presel_pr.copy()
            df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
            df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
            df_reco_sel_fd = None
            if self.doml is True:
                df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])
            else:
                df_reco_sel_fd = df_reco_presel_fd.copy()

            def set_content(df_to_use, histogram,
                            b_c=bincounter):
                if self.apply_pt_weights_eff is True:
                    val, err = self.get_reweighted_count(df_to_use)
                else:
                    val = len(df_to_use)
                    err = math.sqrt(val)
                histogram.SetBinContent(b_c + 1, val)
                histogram.SetBinError(b_c + 1, err)

            set_content(df_gen_sel_pr, h_gen_pr)
            set_content(df_reco_presel_pr, h_presel_pr)
            set_content(df_reco_sel_pr, h_sel_pr)

            set_content(df_gen_sel_fd, h_gen_fd)
            set_content(df_reco_presel_fd, h_presel_fd)
            set_content(df_reco_sel_fd, h_sel_fd)

            bincounter = bincounter + 1

        out_file.cd()
        h_gen_pr.Write()
        h_presel_pr.Write()
        h_sel_pr.Write()
        h_gen_fd.Write()
        h_presel_fd.Write()
        h_sel_fd.Write()


    # pylint: disable=line-too-long
    def process_efficiency_merge_single(self, index):
        return
