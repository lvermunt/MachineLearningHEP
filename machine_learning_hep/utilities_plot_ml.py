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
Methods for correlation and variable plots
"""
import itertools
import pickle
from collections import deque
from io import BytesIO
from os.path import join as osjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split

from machine_learning_hep.utilities import parse_yaml
from machine_learning_hep.logger import get_logger
from machine_learning_hep.utilities import openfile


def importanceplotall(mylistvariables_, names_, trainedmodels_, suffix_, folder):

    if len(names_) == 1:
        plt.figure(figsize=(18, 15))
    else:
        plt.figure(figsize=(25, 15))

    i = 1
    for name, model in zip(names_, trainedmodels_):
        if "SVC" in name:
            continue
        if "Logistic" in name:
            continue
        if "Keras" in name:
            continue
        if len(names_) > 1:
            ax1 = plt.subplot(2, (len(names_)+1)/2, i)
        else:
            ax1 = plt.subplot(1, 1, i)
        #plt.subplots_adjust(left=0.3, right=0.9)
        feature_importances_ = model.feature_importances_
        y_pos = np.arange(len(mylistvariables_))
        ax1.barh(y_pos, feature_importances_, align='center', color='green')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(mylistvariables_, fontsize=17)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Importance', fontsize=17)
        ax1.set_title('Importance features '+name, fontsize=17)
        ax1.xaxis.set_tick_params(labelsize=17)
        plt.xlim(0, 0.7)
        i += 1
    plt.subplots_adjust(wspace=0.5)
    plotname = folder +'/importanceplotall%s.png' % suffix_
    plt.savefig(plotname)
    img_import = BytesIO()
    plt.savefig(img_import, format='png')
    img_import.seek(0)
    return img_import


def decisionboundaries(names_, trainedmodels_, suffix_, x_train_, y_train_, folder):
    mylistvariables_ = x_train_.columns.tolist()
    dictionary_train = x_train_.to_dict(orient='records')
    vec = DictVectorizer()
    x_train_array_ = vec.fit_transform(dictionary_train).toarray()

    figure = plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)
    height = .10
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = x_train_array_[:, 0].min() - .5, x_train_array_[:, 0].max() + .5
    y_min, y_max = x_train_array_[:, 1].min() - .5, x_train_array_[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, height), np.arange(y_min, y_max, height))

    i = 1
    for name, model in zip(names_, trainedmodels_):
        if hasattr(model, "decision_function"):
            z_contour = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            z_contour = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        ax = plt.subplot(2, (len(names_)+1)/2, i)

        z_contour = z_contour.reshape(xx.shape)
        ax.contourf(xx, yy, z_contour, cmap=cm, alpha=.8)
        # Plot also the training points
        ax.scatter(x_train_array_[:, 0], x_train_array_[:, 1],
                   c=y_train_, cmap=cm_bright, edgecolors='k', alpha=0.3)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        score = model.score(x_train_, y_train_)
        ax.text(xx.max() - .3, yy.min() + .3, ('accuracy=%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right', verticalalignment='center')
        ax.set_title(name, fontsize=17)
        ax.set_ylabel(mylistvariables_[1], fontsize=17)
        ax.set_xlabel(mylistvariables_[0], fontsize=17)
        figure.subplots_adjust(hspace=.5)
        i += 1
    plotname = folder +'/decisionboundaries%s.png' % suffix_
    plt.savefig(plotname)
    img_boundary = BytesIO()
    plt.savefig(img_boundary, format='png')
    img_boundary.seek(0)
    return img_boundary


def plot_cross_validation_mse(names_, classifiers_, x_train, y_train, cv_, ncores, suffix_, folder):
    df_scores = pd.DataFrame()
    for name, clf in zip(names_, classifiers_):
        if "Keras" in name:
            ncores = 1
        kfold = StratifiedKFold(n_splits=cv_, shuffle=True, random_state=1)
        scores = cross_val_score(clf, x_train, y_train, cv=kfold,
                                 scoring="neg_mean_squared_error", n_jobs=ncores)
        #scores = cross_val_score(clf, x_train, y_train, cv=cv_,
        #                         scoring="neg_mean_squared_error", n_jobs=ncores)
        tree_rmse_scores = np.sqrt(-scores)
        df_scores[name] = tree_rmse_scores

    figure1 = plt.figure(figsize=(20, 15))
    i = 1
    for name in names_:
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        ax.set_xlim([0, (df_scores[name].mean()*2)])
        plt.hist(df_scores[name].values, color="blue")
        mystring = '$\\mu=%8.2f, \\sigma=%8.2f$' % (df_scores[name].mean(), df_scores[name].std())
        plt.text(0.2, 4., mystring, fontsize=16)
        plt.title(name, fontsize=16)
        plt.xlabel("scores RMSE", fontsize=16)
        plt.ylim(0, 5)
        plt.ylabel("Entries", fontsize=16)
        figure1.subplots_adjust(hspace=.5)
        i += 1
    plotname = folder +'/scoresRME%s.png' % suffix_
    plt.savefig(plotname)
    img_scoresRME = BytesIO()
    plt.savefig(img_scoresRME, format='png')
    img_scoresRME.seek(0)
    return img_scoresRME


def plotdistributiontarget(names_, testset, myvariablesy, suffix_, folder):
    figure1 = plt.figure(figsize=(20, 15))
    i = 1
    for name in names_:
        _ = plt.subplot(2, (len(names_)+1)/2, i)
        plt.hist(testset[myvariablesy].values, color="blue", bins=100, label="true value")
        plt.hist(
            testset['y_test_prediction'+name].values,
            color="red", bins=100, label="predicted value")
        plt.title(name, fontsize=16)
        plt.xlabel(myvariablesy, fontsize=16)
        plt.ylabel("Entries", fontsize=16)
        figure1.subplots_adjust(hspace=.5)
        i += 1
    plt.legend(loc="center right")
    plotname = folder +'/distributionregression%s.png' % suffix_
    plt.savefig(plotname)
    img_dist_reg = BytesIO()
    plt.savefig(img_dist_reg, format='png')
    img_dist_reg.seek(0)
    return img_dist_reg


def plotscattertarget(names_, testset, myvariablesy, suffix_, folder):
    _ = plt.figure(figsize=(20, 15))
    i = 1
    for name in names_:
        figure1 = plt.subplot(2, (len(names_)+1)/2, i)
        plt.scatter(
            testset[myvariablesy].values,
            testset['y_test_prediction'+name].values, color="blue")
        plt.title(name, fontsize=16)
        plt.xlabel(myvariablesy + "true", fontsize=20)
        plt.ylabel(myvariablesy + "predicted", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        figure1.subplots_adjust(hspace=.5)
        i += 1
    plotname = folder +'/scatterplotregression%s.png' % suffix_
    plt.savefig(plotname)
    img_scatt_reg = BytesIO()
    plt.savefig(img_scatt_reg, format='png')
    img_scatt_reg.seek(0)
    return img_scatt_reg


def confusion(names_, classifiers_, suffix_, x_train, y_train, cvgen, folder):
    figure1 = plt.figure(figsize=(25, 15))  # pylint: disable=unused-variable
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

    i = 1
    for name, clf in zip(names_, classifiers_):
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        y_train_pred = cross_val_predict(clf, x_train, y_train, cv=cvgen)
        conf_mx = confusion_matrix(y_train, y_train_pred)
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        np.fill_diagonal(norm_conf_mx, 0)
        df_cm = pd.DataFrame(norm_conf_mx, range(2), range(2))
        sns.set(font_scale=1.4)  # for label size
        ax.set_title(name+"tot diag=0")
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['signal', 'background'])
        ax.yaxis.set_ticklabels(['signal', 'background'])

        i += 1
    plotname = folder +'/confusion_matrix%s_Diag0.png' % suffix_
    plt.savefig(plotname)
    img_confmatrix_dg0 = BytesIO()
    plt.savefig(img_confmatrix_dg0, format='png')
    img_confmatrix_dg0.seek(0)

    figure2 = plt.figure(figsize=(20, 15))  # pylint: disable=unused-variable
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

    i = 1
    for name, clf in zip(names_, classifiers_):
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        y_train_pred = cross_val_predict(clf, x_train, y_train, cv=cvgen)
        conf_mx = confusion_matrix(y_train, y_train_pred)
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        df_cm = pd.DataFrame(norm_conf_mx, range(2), range(2))
        sns.set(font_scale=1.4)  # for label size
        ax.set_title(name)
        sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['signal', 'background'])
        ax.yaxis.set_ticklabels(['signal', 'background'])

        i += 1
    plotname = folder +'/confusion_matrix%s.png' % suffix_
    plt.savefig(plotname)
    img_confmatrix = BytesIO()
    plt.savefig(img_confmatrix, format='png')
    img_confmatrix.seek(0)
    return img_confmatrix_dg0, img_confmatrix


def precision_recall(names_, classifiers_, suffix_, x_train, y_train, cvgen, folder):

    if len(names_) == 1:
        figure1 = plt.figure(figsize=(20, 15))  # pylint: disable=unused-variable
    else:
        figure1 = plt.figure(figsize=(25, 15))  # pylint: disable=unused-variable
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

    i = 1
    for name, clf in zip(names_, classifiers_):
        if len(names_) > 1:
            plt.subplot(2, (len(names_)+1)/2, i)
        y_proba = cross_val_predict(clf, x_train, y_train, cv=cvgen, method="predict_proba")
        y_scores = y_proba[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision=TP/(TP+FP)", linewidth=5.0)
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall=TP/(TP+FN)", linewidth=5.0)
        plt.xlabel('Probability', fontsize=20)
        plt.ylabel('Precision or Recall', fontsize=20)
        plt.title('Precision, Recall '+name, fontsize=20)
        plt.legend(loc="best", prop={'size': 30})
        plt.ylim([0, 1])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        i += 1
    plotname = folder +'/precision_recall%s.png' % suffix_
    plt.savefig(plotname)
    img_precision_recall = BytesIO()
    plt.savefig(img_precision_recall, format='png')
    img_precision_recall.seek(0)

    figure2 = plt.figure(figsize=(20, 15))  # pylint: disable=unused-variable
    i = 1
    aucs = []

    for name, clf in zip(names_, classifiers_):
        y_proba = cross_val_predict(clf, x_train, y_train, cv=cvgen, method="predict_proba")
        y_scores = y_proba[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.xlabel('False Positive Rate or (1 - Specifity)', fontsize=20)
        plt.ylabel('True Positive Rate or (Sensitivity)', fontsize=20)
        plt.title('Receiver Operating Characteristic', fontsize=20)
        plt.plot(fpr, tpr, "b-", label='ROC %s (AUC = %0.2f)' %
                 (name, roc_auc), linewidth=5.0)
        plt.legend(loc="lower center", prop={'size': 30})
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        i += 1
    plotname = folder +'/ROCcurve%s.png' % suffix_
    plt.savefig(plotname)
    img_roc = BytesIO()
    plt.savefig(img_roc, format='png')
    img_roc.seek(0)

    return img_precision_recall, img_roc


def plot_learning_curves(names_, classifiers_, suffix_, folder, x_data, y_data, npoints):

    figure1 = plt.figure(figsize=(20, 15))
    i = 1
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    for name, clf in zip(names_, classifiers_):
        if len(names_) > 1:
            plt.subplot(2, (len(names_)+1)/2, i)
        train_errors, val_errors = [], []
        high = len(x_train)
        low = 100
        step_ = int((high-low)/npoints)
        arrayvalues = np.arange(start=low, stop=high, step=step_)
        for m in arrayvalues:
            clf.fit(x_train[:m], y_train[:m])
            y_train_predict = clf.predict(x_train[:m])
            y_val_predict = clf.predict(x_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
        plt.plot(arrayvalues, np.sqrt(train_errors), "r-+", linewidth=5, label="training")
        plt.plot(arrayvalues, np.sqrt(val_errors), "b-", linewidth=5, label="testing")
        plt.ylim([0, np.amax(np.sqrt(val_errors))*2])
        plt.title("Learning curve "+name, fontsize=20)
        plt.xlabel("Training set size", fontsize=20)
        plt.ylabel("RMSE", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if len(names_) > 1:
            figure1.subplots_adjust(hspace=.5)
        plt.legend(loc="best", prop={'size': 30})
        i += 1
    plotname = folder +'/learning_curve%s.png' % suffix_
    plt.savefig(plotname)
    img_learn = BytesIO()
    plt.savefig(img_learn, format='png')
    img_learn.seek(0)
    return img_learn


def plot_overtraining(names, classifiers, suffix, folder, x_train, y_train, x_val, y_val,
                      bins=50):
    for name, clf in zip(names, classifiers):
        fig = plt.figure(figsize=(10, 8))
        decisions = []
        for x, y in ((x_train, y_train), (x_val, y_val)):
            d1 = clf.predict_proba(x[y > 0.5])[:, 1]
            d2 = clf.predict_proba(x[y < 0.5])[:, 1]
            decisions += [d1, d2]

        plt.hist(decisions[0], color='r', alpha=0.5, range=[0, 1], bins=bins,
                 histtype='stepfilled', density=True, label='S, train')
        plt.hist(decisions[1], color='b', alpha=0.5, range=[0, 1], bins=bins,
                 histtype='stepfilled', density=True, label='B, train')
        hist, bins = np.histogram(decisions[2], bins=bins, range=[0, 1], density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S, test')
        hist, bins = np.histogram(decisions[3], bins=bins, range=[0, 1], density=True)
        scale = len(decisions[3]) / sum(hist)
        err = np.sqrt(hist * scale) / scale
        plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B, test')
        plt.xlabel("Model output", fontsize=15)
        plt.ylabel("Arbitrary units", fontsize=15)
        plt.legend(loc="best", frameon=False, fontsize=15)
        plt.yscale("log")

        plot_name = f'{folder}/ModelOutDistr_{name}_{suffix}.png'
        fig.savefig(plot_name)
        plot_name = plot_name.replace('png', 'pickle')
        with open(plot_name, 'wb') as out:
            pickle.dump(fig, out)


def roc_train_test(names, classifiers, x_train, y_train, x_test, y_test, suffix, folder):
    fig = plt.figure(figsize=(20, 15))

    for name, clf in zip(names, classifiers):
        y_train_pred = clf.predict_proba(x_train)[:, 1]
        y_test_pred = clf.predict_proba(x_test)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_test = auc(fpr_test, tpr_test)
        train_line = plt.plot(fpr_train, tpr_train, lw=3, alpha=0.4,
                              label=f'ROC {name} - Train set (AUC = {roc_auc_train:.4f})')
        plt.plot(fpr_test, tpr_test, lw=3, alpha=0.8, c=train_line[0].get_color(),
                 label=f'ROC {name} - Test set (AUC = {roc_auc_test:.4f})')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend(loc='lower right', prop={'size': 18})
    plot_name = f'{folder}/ROCtraintest{suffix}.png'
    fig.savefig(plot_name)
    plot_name = plot_name.replace('png', 'pickle')
    with open(plot_name, 'wb') as out:
        pickle.dump(fig, out)


def vardistplot(dataframe_sig_, dataframe_bkg_, mylistvariables_, output_,
                binmin, binmax):
    figure = plt.figure(figsize=(20, 15)) # pylint: disable=unused-variable
    i = 1
    for var in mylistvariables_:
        ax = plt.subplot(3, int(len(mylistvariables_)/3+1), i)
        plt.xlabel(var, fontsize=11)
        plt.ylabel("entries", fontsize=11)
        plt.yscale('log')
        kwargs = dict(alpha=0.3, density=True, bins=100)
        plt.hist(dataframe_sig_[var], facecolor='b', label='signal', **kwargs)
        plt.hist(dataframe_bkg_[var], facecolor='g', label='background', **kwargs)
        ax.legend()
        i = i+1
    plotname = output_+'/variablesDistribution_nVar%d_%d%d.png' % \
                            (len(mylistvariables_), binmin, binmax)
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO


def vardistplot_probscan(dataframe_, mylistvariables_, modelname_, thresharray_, # pylint: disable=too-many-statements
                         output_, suffix_, opt=1, plot_options_=None):

    plot_type_name = "prob_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    fig = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [fig.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

    # Sort the thresharray_
    thresharray_.sort()
    # Re-use skimmed dataframe
    df_skimmed = None
    variables_selected = mylistvariables_ + [f"y_test_prob{modelname_}"]

    xrange_min = []
    xrange_max = []
    ref_hists = []
    for thresh_index, threshold in enumerate(thresharray_):
        selml = f"y_test_prob{modelname_}>{threshold}"
        if df_skimmed is None:
            df_skimmed = dataframe_.query(selml)[variables_selected]
        else:
            df_skimmed = df_skimmed.query(selml)

        if thresh_index == 0 and len(df_skimmed[mylistvariables_[0]]) == 0:
            logger.warning("Dataframe is empty, skipping probscan")
            break

        for i, var in enumerate(mylistvariables_):

            # Extract minimum and maximum for x-axis, this is only done once
            # for each variable
            if thresh_index == 0:
                axes[i].set_xlabel(var, fontsize=30)
                ylabel = "entries"
                if opt == 1:
                    ylabel += f"/entries(prob{thresharray_[0]})"
                axes[i].set_ylabel(ylabel, fontsize=30)
                axes[i].tick_params(labelsize=20)
                if var in plot_options and "xlim" in plot_options[var]:
                    xrange_min.append(plot_options[var]["xlim"][0])
                    xrange_max.append(plot_options[var]["xlim"][1])
                else:
                    values0 = df_skimmed[var]
                    xrange_min.append(values0.min())
                    xrange_max.append(values0.max())

            n = len(df_skimmed[var])
            lbl = f'prob > {threshold} n = {n}'
            clr = color[thresh_index%len(color)]
            values = df_skimmed[var]
            his, bina = np.histogram(values, range=(xrange_min[i], xrange_max[i]), bins=100)
            if thresh_index == 0:
                ref_hists.append(his)
            width = np.diff(bina)
            center = (bina[:-1] + bina[1:]) / 2

            if opt == 0:
                axes[i].set_yscale('log')
            elif opt == 1:
                his = np.divide(his, ref_hists[i])
                axes[i].set_ylim(0.001, 1.1)

            if np.any(his):
                axes[i].bar(center, his, align='center', width=width, facecolor=clr, label=lbl)
                axes[i].legend(fontsize=10)
    plotname = osjoin(output_, f"variables_distribution_{suffix_}_ratio{opt}.png")
    plt.savefig(plotname, bbox_inches='tight')


def efficiency_cutscan(dataframe_, mylistvariables_, modelname_, threshold, # pylint: disable=too-many-statements
                       output_, suffix_, plot_options_=None):

    plot_type_name = "eff_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})
    selml = "y_test_prob%s>%s" % (modelname_, threshold)
    dataframe_ = dataframe_.query(selml)

    fig = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [fig.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

    for i, var_tuple in enumerate(mylistvariables_):
        var = var_tuple[0]
        vardir = var_tuple[1]
        cen = var_tuple[2]

        axes[i].set_xlabel(var, fontsize=30)
        axes[i].set_ylabel("entries (normalised)", fontsize=30)
        axes[i].tick_params(labelsize=20)
        axes[i].set_yscale('log')
        axes[i].set_ylim(0.1, 1.5)
        values = dataframe_[var].values
        if "abs" in  vardir:
            values = np.array([abs(v - cen) for v in values])
        nbinscan = 100
        minv, maxv = values.min(), values.max()
        if var in plot_options and "xlim" in plot_options[var]:
            minv = plot_options[var]["xlim"][0]
            maxv = plot_options[var]["xlim"][1]
        else:
            minv = values.min()
            maxv = values.max()
        _, bina = np.histogram(values, range=(minv, maxv), bins=nbinscan)
        widthbin = (maxv - minv) / float(nbinscan)
        width = np.diff(bina)
        center = (bina[:-1] + bina[1:]) / 2
        den = len(values)
        ratios = deque()
        if "lt" in vardir:
            for ibin in range(nbinscan):
                values = values[values > minv+widthbin*ibin]
                num = len(values)
                eff = float(num)/float(den)
                ratios.append(eff)
        elif "st" in vardir:
            for ibin in range(nbinscan, 0, -1):
                values = values[values < minv+widthbin*ibin]
                num = len(values)
                eff = float(num)/float(den)
                ratios.appendleft(eff)
        lbl = f'prob > {threshold}'
        axes[i].bar(center, ratios, align='center', width=width, label=lbl)
        axes[i].legend(fontsize=30)
    plotname = osjoin(output_, f"variables_effscan_prob{threshold}_{suffix_}.png")
    plt.savefig(plotname, bbox_inches='tight')
    plt.savefig(plotname, bbox_inches='tight')


def picklesize_cutscan(dataframe_, mylistvariables_, output_, suffix_, plot_options_=None): # pylint: disable=too-many-statements

    plot_type_name = "picklesize_cut_scan"
    plot_options = {}
    if isinstance(plot_options_, dict):
        plot_options = plot_options_.get(plot_type_name, {})

    fig = plt.figure(figsize=(60, 25))
    gs = GridSpec(3, int(len(mylistvariables_)/3+1))
    axes = [fig.add_subplot(gs[i]) for i in range(len(mylistvariables_))]

    df_reference_pkl_size = len(pickle.dumps(dataframe_, protocol=4))
    df_reference_size = dataframe_.shape[0] * dataframe_.shape[1]

    for i, var_tuple in enumerate(mylistvariables_):
        var = var_tuple[0]
        vardir = var_tuple[1]
        cen = var_tuple[2]

        axes[i].set_xlabel(var, fontsize=30)
        axes[i].set_ylabel("rel. pickle size after cut", fontsize=30)
        axes[i].tick_params(labelsize=20)
        axes[i].set_yscale('log')
        axes[i].set_ylim(0.005, 1.5)
        values = dataframe_[var].values
        if "abs" in  vardir:
            values = np.array([abs(v - cen) for v in values])
        nbinscan = 100
        if var in plot_options and "xlim" in plot_options[var]:
            minv = plot_options[var]["xlim"][0]
            maxv = plot_options[var]["xlim"][1]
        else:
            minv = values.min()
            maxv = values.max()
        _, bina = np.histogram(values, range=(minv, maxv), bins=nbinscan)
        widthbin = (maxv - minv) / float(nbinscan)
        width = np.diff(bina)
        center = (bina[:-1] + bina[1:]) / 2
        ratios_df_pkl_size = deque()
        ratios_df_size = deque()
        df_skimmed = dataframe_
        if "lt" in vardir:
            for ibin in range(nbinscan):
                df_skimmed = df_skimmed.iloc[values > minv+widthbin*ibin]
                values = values[values > minv+widthbin*ibin]
                num = len(pickle.dumps(df_skimmed, protocol=4))
                eff = float(num)/float(df_reference_pkl_size)
                ratios_df_pkl_size.append(eff)
                num = df_skimmed.shape[0] * df_skimmed.shape[1]
                eff = float(num)/float(df_reference_size)
                ratios_df_size.append(eff)
        elif "st" in vardir:
            for ibin in range(nbinscan, 0, -1):
                df_skimmed = df_skimmed.iloc[values < minv+widthbin*ibin]
                values = values[values < minv+widthbin*ibin]
                num = len(pickle.dumps(df_skimmed, protocol=4))
                eff = float(num)/float(df_reference_pkl_size)
                ratios_df_pkl_size.appendleft(eff)
                num = df_skimmed.shape[0] * df_skimmed.shape[1]
                eff = float(num)/float(df_reference_size)
                ratios_df_size.appendleft(eff)
        axes[i].bar(center, ratios_df_pkl_size, align='center', width=width, label="rel. pkl size",
                    alpha=0.5)
        axes[i].bar(center, ratios_df_size, align='center', width=width, label="rel. df length",
                    alpha=0.5)
        axes[i].legend(fontsize=30)
    plotname = osjoin(output_, f"variables_cutscan_picklesize_{suffix_}.png")
    plt.savefig(plotname, bbox_inches='tight')


def scatterplot(dataframe_sig_, dataframe_bkg_, mylistvariablesx_,
                mylistvariablesy_, output_, binmin, binmax):
    figurecorr = plt.figure(figsize=(30, 20)) # pylint: disable=unused-variable
    i = 1
    for j, _ in enumerate(mylistvariablesx_):
        axcorr = plt.subplot(3, int(len(mylistvariablesx_)/3+1), i)
        plt.xlabel(mylistvariablesx_[j], fontsize=11)
        plt.ylabel(mylistvariablesy_[j], fontsize=11)
        plt.scatter(
            dataframe_bkg_[mylistvariablesx_[j]], dataframe_bkg_[mylistvariablesy_[j]],
            alpha=0.4, c="g", label="background")
        plt.scatter(
            dataframe_sig_[mylistvariablesx_[j]], dataframe_sig_[mylistvariablesy_[j]],
            alpha=0.4, c="b", label="signal")
        plt.title(
            'Pearson sgn: %s' %
            dataframe_sig_.corr().loc[mylistvariablesx_[j]][mylistvariablesy_[j]].round(2)+
            ',  Pearson bkg: %s' %
            dataframe_bkg_.corr().loc[mylistvariablesx_[j]][mylistvariablesy_[j]].round(2))
        axcorr.legend()
        i = i+1
    plotname = output_+'/variablesScatterPlot%d%d.png' % (binmin, binmax)
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO


def correlationmatrix(dataframe, mylistvariables, label, output, binmin, binmax):
    corr = dataframe[mylistvariables].corr()
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(label, fontsize=11)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1, vmax=1,
                square=True, ax=ax)
    nVar = len(mylistvariables)
    plotname = f'{output}/CorrMatrix_{label}_nVar{nVar}_{binmin:.1f}_{binmax:.1f}.png'
    plt.savefig(plotname, bbox_inches='tight')
    imagebytesIO = BytesIO()
    plt.savefig(imagebytesIO, format='png')
    imagebytesIO.seek(0)
    return imagebytesIO


# pylint: disable=too-many-locals, too-many-statements
def perform_plot_gridsearch(names, out_dirs):
    '''
    Function for grid scores plotting (working with scikit 0.20)
    '''
    logger = get_logger()

    for name, out_dir in zip(names, out_dirs):

        # Read written results
        gps = parse_yaml(osjoin(out_dir, "parameters.yaml"))
        score_obj = pickle.load(openfile(osjoin(out_dir, "results.pkl"), "rb"))

        param_keys = [f"param_{key}" for key in gps["params"].keys()]
        if not param_keys:
            logger.warning("Add at least 1 parameter (even just 1 value)")
            continue

        # Re-arrange scoring such that the refitted one is always on top
        score_names = gps["scoring"]
        refit_score = gps["refit"]
        del score_names[score_names.index(refit_score)]
        score_names.insert(0, refit_score)

        # Extract scores
        x_labels = []
        y_values = {}
        y_errors = {}

        for sn in score_names:
            y_values[sn] = {"train": [], "test": []}
            y_errors[sn] = {"train": [], "test": []}

        # Get indices of values to put on x-axis and identify parameter combination
        values_indices = [range(len(values)) for values in gps["params"].values()]

        y_axis_mins = {sn: 9999 for sn in score_names}
        y_axis_maxs = {sn: -9999 for sn in score_names}
        for indices, case in zip(itertools.product(*values_indices),
                                 itertools.product(*list(gps["params"].values()))):
            df_case = score_obj.copy()
            for i_case, i_key in zip(case, param_keys):
                df_case = df_case.loc[df_case[i_key] == df_case[i_key].dtype.type(i_case)]

            x_labels.append(",".join([str(i) for i in indices]))
            # As we just nailed it down to one value
            for sn in score_names:
                for tt in ("train", "test"):
                    y_values[sn][tt].append(df_case[f"mean_{tt}_{sn}"].values[0])
                    y_errors[sn][tt].append(df_case[f"std_{tt}_{sn}"].values[0])
                    y_axis_mins[sn] = min(y_axis_mins[sn], y_values[sn][tt][-1])
                    y_axis_maxs[sn] = max(y_axis_maxs[sn], y_values[sn][tt][-1])

        # Prepare text for parameters
        text_parameters = "\n".join([f"{key}: {values}" for key, values in gps["params"].items()])

        # To determine fontsizes later
        figsize = (35, 18 * len(score_names))
        fig, axes = plt.subplots(len(score_names), 1, sharex=True, gridspec_kw={"hspace": 0.05},
                                 figsize=figsize)
        ax_plot = dict(zip(score_names, axes))

        # The axes to put the parameter list
        ax_main = axes[-1]
        # The axes with the title being on top
        ax_top = axes[0]

        points_per_inch = 72
        markerstyles = ["o", "+"]
        markersize = 20

        for sn in score_names:
            ax = ax_plot[sn]
            ax_min = y_axis_mins[sn] - (y_axis_maxs[sn] - y_axis_mins[sn]) / 10.
            ax_max = y_axis_maxs[sn] + (y_axis_maxs[sn] - y_axis_mins[sn]) / 10.
            ax.set_ylim(ax_min, ax_max)
            ax.set_ylabel(f"mean {sn}", fontsize=20)
            ax.get_yaxis().set_tick_params(labelsize=20)

            for j, tt in enumerate(("train", "test")):
                markerstyle = markerstyles[j % len(markerstyles)]

                ax.errorbar(range(len(x_labels)), y_values[sn][tt], yerr=y_errors[sn][tt],
                            ls="", marker=markerstyle, markersize=markersize, label=f"{sn} ({tt})")

                # Add values to points
                ylim = ax.get_ylim()
                plot_labels_offset = (ylim[1] - ylim[0]) / 40
                for x, y in enumerate(y_values[sn][tt]):
                    ax.text(x, y - plot_labels_offset, f"{y:.4f}", fontsize=20)

        ax_main.set_xlabel("parameter indices", fontsize=20)
        ax_top.set_title(f"Grid search {name}", fontsize=30)
        ax_main.get_xaxis().set_tick_params(labelsize=20)
        ax_main.set_xticks(range(len(x_labels)))
        ax_main.set_xticklabels(x_labels, rotation=45)

        text_point_size = int(4 * fig.dpi / points_per_inch * figsize[1] / len(gps["params"]))
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()

        xlow = xlim[0] + (xlim[1] - xlim[0]) / 100
        ylow = ylim[0] + (ylim[1] - ylim[0]) / 3
        ax_main.text(xlow, ylow, text_parameters, fontsize=text_point_size)

        for ax in ax_plot.values():
            ax.legend(loc="center right", fontsize=20)
        plotname = osjoin(out_dir, "GridSearchResults.png")
        plt.savefig(plotname)
        plt.close(fig)


def plotvariance_pca(pca_object, output_):
    figure = plt.figure(figsize=(15, 10)) # pylint: disable=unused-variable
    plt.plot(np.cumsum(pca_object.explained_variance_ratio_))
    plt.plot([0, 10], [0.95, 0.95])
    plt.xlabel('number of components', fontsize=16)
    plt.ylabel('cumulative explained variance', fontsize=16)
    plt.title('Explained variance', fontsize=16)
    plt.ylim([0, 1])
    plotname = output_+'/PCAvariance.png'
    plt.savefig(plotname, bbox_inches='tight')
    img_pca = BytesIO()
    plt.savefig(img_pca, format='png')
    img_pca.seek(0)
