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
main script for doing ml optimisation
"""
import os
import sys
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from hipe4ml import plot_utils
from hipe4ml.model_handler import ModelHandler

from machine_learning_hep.logger import get_logger
from machine_learning_hep.utilities import openfile
from machine_learning_hep.utilities_selection import seldf_singlevar
from machine_learning_hep.utilities_selection import selectdfquery
from machine_learning_hep.utilities_selection import split_df_sigbkg


# pylint: disable=too-many-instance-attributes, too-many-statements
class Optimiserhipe4ml:
    # Class Attribute
    species = "optimiser_hipe4ml"

    def __init__(self, data_param, binmin, binmax, training_var, hyper_pars):

        self.logger = get_logger()

        self.v_bin = data_param["var_binning"]
        # directory
        dirmcml = data_param["multi"]["mc"]["pkl_skimmed_merge_for_ml_all"]
        dirdataml = data_param["multi"]["data"]["pkl_skimmed_merge_for_ml_all"]
        # directory
        self.dirmlout = data_param["ml"]["mlout"]
        self.dirmlplot = data_param["ml"]["mlplot"]
        # ml file names
        self.n_reco = data_param["files_names"]["namefile_reco"]
        self.n_reco = self.n_reco.replace(".pkl", "_%s%d_%d.pkl" % (self.v_bin, binmin, binmax))
        self.n_gen = data_param["files_names"]["namefile_gen"]
        self.n_gen = self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % (self.v_bin, binmin, binmax))
        # ml files
        self.f_gen_mc = os.path.join(dirmcml, self.n_gen)
        self.f_reco_mc = os.path.join(dirmcml, self.n_reco)
        self.f_reco_data = os.path.join(dirdataml, self.n_reco)
        # variables
        self.v_train = training_var
        self.v_sig = data_param["variables"]["var_signal"]
        # parameters
        self.p_nbkg = data_param["ml"]["nbkg"]
        self.p_nsig = data_param["ml"]["nsig"]
        self.p_tagsig = data_param["ml"]["sampletagforsignal"]
        self.p_tagbkg = data_param["ml"]["sampletagforbkg"]
        self.p_binmin = binmin
        self.p_binmax = binmax
        self.rnd_shuffle = data_param["ml"]["rnd_shuffle"]
        self.rnd_splt = data_param["ml"]["rnd_splt"]
        self.test_frac = data_param["ml"]["test_frac"]

        self.p_evtsel = data_param["ml"]["evtsel"]
        self.p_triggersel_mc = data_param["ml"]["triggersel"]["mc"]
        self.p_triggersel_data = data_param["ml"]["triggersel"]["data"]

        # dataframes
        self.df_mc = None
        self.df_mcgen = None
        self.df_data = None
        self.df_sig = None
        self.df_bkg = None
        self.df_ml = None
        self.df_mltest = None
        self.df_mltrain = None
        self.df_sigtrain = None
        self.df_sigtest = None
        self.df_bkgtrain = None
        self.df_xtrain = None
        self.df_ytrain = None
        self.df_xtest = None
        self.df_ytest = None
        self.traintestdata = None
        self.ypredtrain_hipe4ml = None
        self.ypredtest_hipe4ml = None
        # selections
        self.s_selbkgml = data_param["ml"]["sel_bkgml"]
        self.s_selsigml = data_param["ml"]["sel_sigml"]
        self.p_presel_gen_eff = data_param["ml"]["opt"]["presel_gen_eff"]

        self.preparesample()

        self.p_hipe4ml_model = None
        self.v_hipe4ml_pars = hyper_pars
        self.load_hipe4mlmodel()

        self.bayesoptconfig_hipe4ml = data_param["hipe4ml"]["hyper_par_opt"]["bayes_opt_config"]
        self.average_method_hipe4ml = data_param["hipe4ml"]["roc_auc_average"]
        self.nfold_hipe4ml = data_param["hipe4ml"]["hyper_par_opt"]["nfolds"]
        self.init_points = data_param["hipe4ml"]["hyper_par_opt"]["initpoints"]
        self.n_iter_hipe4ml = data_param["hipe4ml"]["hyper_par_opt"]["niter"]
        self.njobs_hipe4ml = data_param["hipe4ml"]["hyper_par_opt"]["njobs"]
        self.roc_method_hipe4ml = data_param["hipe4ml"]["roc_auc_approach"]
        self.raw_output_hipe4ml = data_param["hipe4ml"]["raw_output"]
        self.train_test_log_hipe4ml = data_param["hipe4ml"]["train_test_log"]

        self.logger.info("Using the following training variables: %s", training_var)

    def preparesample(self):
        self.logger.info("Prepare Sample for hipe4ml")
        self.df_data = pickle.load(openfile(self.f_reco_data, "rb"))
        self.df_mc = pickle.load(openfile(self.f_reco_mc, "rb"))
        self.df_mcgen = pickle.load(openfile(self.f_gen_mc, "rb"))
        self.df_data = selectdfquery(self.df_data, self.p_evtsel)
        self.df_mc = selectdfquery(self.df_mc, self.p_evtsel)
        self.df_mcgen = selectdfquery(self.df_mcgen, self.p_evtsel)

        self.df_data = selectdfquery(self.df_data, self.p_triggersel_data)
        self.df_mc = selectdfquery(self.df_mc, self.p_triggersel_mc)
        self.df_mcgen = selectdfquery(self.df_mcgen, self.p_triggersel_mc)

        self.df_mcgen = self.df_mcgen.query(self.p_presel_gen_eff)
        arraydf = [self.df_data, self.df_mc]
        self.df_mc = seldf_singlevar(self.df_mc, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_mcgen = seldf_singlevar(self.df_mcgen, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_data = seldf_singlevar(self.df_data, self.v_bin, self.p_binmin, self.p_binmax)

        self.df_sig, self.df_bkg = arraydf[self.p_tagsig], arraydf[self.p_tagbkg]
        self.df_sig = seldf_singlevar(self.df_sig, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_bkg = seldf_singlevar(self.df_bkg, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_sig = self.df_sig.query(self.s_selsigml)
        self.df_bkg = self.df_bkg.query(self.s_selbkgml)
        self.df_bkg["ismcsignal"] = 0
        self.df_bkg["ismcprompt"] = 0
        self.df_bkg["ismcfd"] = 0
        self.df_bkg["ismcbkg"] = 0

        if self.p_nsig > len(self.df_sig):
            self.logger.warning("There are not enough signal events")
        if self.p_nbkg > len(self.df_bkg):
            self.logger.warning("There are not enough background events")

        self.p_nsig = min(len(self.df_sig), self.p_nsig)
        self.p_nbkg = min(len(self.df_bkg), self.p_nbkg)

        self.logger.info("Used number of signal events is %d", self.p_nsig)
        self.logger.info("Used number of background events is %d", self.p_nbkg)

        self.df_ml = pd.DataFrame()
        self.df_sig = shuffle(self.df_sig, random_state=self.rnd_shuffle)
        self.df_bkg = shuffle(self.df_bkg, random_state=self.rnd_shuffle)
        self.df_sig = self.df_sig[:self.p_nsig]
        self.df_bkg = self.df_bkg[:self.p_nbkg]
        self.df_sig[self.v_sig] = 1
        self.df_bkg[self.v_sig] = 0
        self.df_ml = pd.concat([self.df_sig, self.df_bkg])
        self.df_mltrain, self.df_mltest = train_test_split(self.df_ml, test_size=self.test_frac,
                                                           random_state=self.rnd_splt)
        self.df_mltrain = self.df_mltrain.reset_index(drop=True)
        self.df_mltest = self.df_mltest.reset_index(drop=True)
        self.df_sigtrain, self.df_bkgtrain = split_df_sigbkg(self.df_mltrain, self.v_sig)
        self.df_sigtest, self.df_bkgtest = split_df_sigbkg(self.df_mltest, self.v_sig)
        self.logger.info("Total number of candidates: train %d and test %d", len(self.df_mltrain),
                         len(self.df_mltest))
        self.logger.info("Number of signal candidates: train %d and test %d",
                         len(self.df_sigtrain), len(self.df_sigtest))
        self.logger.info("Number of bkg candidates: %d and test %d", len(self.df_bkgtrain),
                         len(self.df_bkgtest))

        self.df_xtrain = self.df_mltrain[self.v_train]
        self.df_ytrain = self.df_mltrain[self.v_sig]
        self.df_xtest = self.df_mltest[self.v_train]
        self.df_ytest = self.df_mltest[self.v_sig]
        self.traintestdata = [self.df_xtrain, self.df_ytrain, self.df_xtest, self.df_ytest]
        # self.traintestdata = [self.df_mltrain, self.df_ytrain, self.df_mltest, self.df_ytest]

    def load_hipe4mlmodel(self):
        self.logger.info("Loading hipe4ml model")
        model_xgboost = xgb.XGBClassifier()
        self.p_hipe4ml_model = ModelHandler(model_xgboost, self.v_train, self.v_hipe4ml_pars)

    def set_hipe4ml_modelpar(self):
        self.logger.info("Setting hipe4ml hyperparameters")
        self.p_hipe4ml_model.set_model_params(self.v_hipe4ml_pars)

    def do_hipe4mlhyperparopti(self):
        self.logger.info("Optimising hipe4ml hyperparameters (Bayesian)")

        if not (self.average_method_hipe4ml in ['macro', 'weighted'] and
                self.roc_method_hipe4ml in ['ovo', 'ovr']):
            self.logger.fatal("Selected ROC configuration is not valid!")

        if self.average_method_hipe4ml == 'weighted':
            metric = f'roc_auc_{self.roc_method_hipe4ml}_{self.average_method_hipe4ml}'
        else:
            metric = f'roc_auc_{self.roc_method_hipe4ml}'

        hypparsfile = f'{self.dirmlout}/HyperParOpt_pT_{self.p_binmin}_{self.p_binmax}.txt'
        outfilehyppars = open(hypparsfile, 'wt')
        sys.stdout = outfilehyppars
        self.p_hipe4ml_model.optimize_params_bayes(self.traintestdata, self.bayesoptconfig_hipe4ml,
                                                   metric, self.nfold_hipe4ml, self.init_points,
                                                   self.n_iter_hipe4ml, self.njobs_hipe4ml)
        outfilehyppars.close()
        sys.stdout = sys.__stdout__
        self.logger.info("Performing hyper-parameters optimisation: Done!")

    def do_hipe4mltrain(self):
        self.logger.info("Training + testing hipe4ml model")
        t0 = time.time()

        self.p_hipe4ml_model.train_test_model(self.traintestdata, self.average_method_hipe4ml,
                                              self.roc_method_hipe4ml)
        self.ypredtrain_hipe4ml = self.p_hipe4ml_model.predict(self.traintestdata[0],
                                                               self.raw_output_hipe4ml)
        self.ypredtest_hipe4ml = self.p_hipe4ml_model.predict(self.traintestdata[2],
                                                              self.raw_output_hipe4ml)

        modelfile = f'{self.dirmlout}/ModelHandler_pT_{self.p_binmin}_{self.p_binmax}.pickle'
        self.p_hipe4ml_model.dump_model_handler(modelfile)

        self.logger.info("Training + testing hipe4ml: Done!")
        self.logger.info("Time elapsed = %.3f", time.time() - t0)

    def do_hipe4mlplot(self):
        self.logger.info("Plotting hipe4ml model")

        leglabels = ["Background", "Prompt signal"]
        outputlabels = ["Bkg", "SigPrompt"]

        # _____________________________________________
        plot_utils.plot_distr([self.df_bkgtrain, self.df_sigtrain], self.v_train, 100, leglabels)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        figname = f'{self.dirmlplot}/DistributionsAll_pT_{self.p_binmin}_{self.p_binmax}.pdf'
        plt.savefig(figname)
        plt.close('all')
        # _____________________________________________
        corrmatrixfig = plot_utils.plot_corr([self.df_bkgtrain, self.df_sigtrain],
                                             self.v_train, leglabels)
        for figg, labb in zip(corrmatrixfig, outputlabels):
            plt.figure(figg.number)
            plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
            figname = f'{self.dirmlplot}/CorrMatrix{labb}_pT_{self.p_binmin}_{self.p_binmax}.pdf'
            figg.savefig(figname)
        # _____________________________________________
        plt.rcParams["figure.figsize"] = (10, 7)
        mloutputfig = plot_utils.plot_output_train_test(self.p_hipe4ml_model, self.traintestdata,
                                                        80, self.raw_output_hipe4ml,
                                                        leglabels, self.train_test_log_hipe4ml,
                                                        density=True)
        figname = f'{self.dirmlplot}/MLOutputDistr_pT_{self.p_binmin}_{self.p_binmax}.pdf'
        mloutputfig.savefig(figname)
        # _____________________________________________
        plt.rcParams["figure.figsize"] = (10, 9)
        roccurvefig = plot_utils.plot_roc(self.traintestdata[3], self.ypredtest_hipe4ml,
                                          None, leglabels, self.average_method_hipe4ml,
                                          self.roc_method_hipe4ml)
        figname = f'{self.dirmlplot}/ROCCurveAll_pT_{self.p_binmin}_{self.p_binmax}.pdf'
        roccurvefig.savefig(figname)
        # _____________________________________________
        plt.rcParams["figure.figsize"] = (10, 9)
        roccurvettfig = plot_utils.plot_roc_train_test(self.traintestdata[3],
                                                       self.ypredtest_hipe4ml,
                                                       self.traintestdata[1],
                                                       self.ypredtrain_hipe4ml, None,
                                                       leglabels, self.average_method_hipe4ml,
                                                       self.roc_method_hipe4ml)
        figname = f'{self.dirmlplot}/ROCCurveTrainTest_pT_{self.p_binmin}_{self.p_binmax}.pdf'
        roccurvettfig.savefig(figname)
        # _____________________________________________
        precisionrecallfig = plot_utils.plot_precision_recall(self.traintestdata[3],
                                                              self.ypredtest_hipe4ml,
                                                              leglabels)
        figname = f'{self.dirmlplot}/PrecisionRecallAll_pT_{self.p_binmin}_{self.p_binmax}.pdf'
        precisionrecallfig.savefig(figname)
        # _____________________________________________
        plt.rcParams["figure.figsize"] = (12, 7)
        featuresimportancefig = plot_utils.plot_feature_imp(self.traintestdata[2][self.v_train],
                                                            self.traintestdata[3],
                                                            self.p_hipe4ml_model,
                                                            leglabels)
        figname = (f'{self.dirmlplot}/FeatureImportanceAll_'
                   f'pT_{self.p_binmin}_{self.p_binmax}.pdf')
        featuresimportancefig.savefig(figname)
