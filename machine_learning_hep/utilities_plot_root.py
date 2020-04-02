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
Script containing all helper functions related to plotting with ROOT
"""
import math
# pylint: disable=too-many-lines
from array import array

from root_numpy import fill_hist  # pylint: disable=import-error, no-name-in-module

# pylint: disable=import-error, no-name-in-module
from ROOT import TH2F, TFile, TH1, TH3F
from ROOT import TPad, TCanvas, TLegend, kBlack, kGreen, kRed, kBlue, kWhite
from ROOT import gStyle, gROOT

from machine_learning_hep.logger import get_logger


def buildarray(listnumber):
    arraynumber = array('d', listnumber)
    return arraynumber

def makefill3dhist(df_, titlehist, arrayx, arrayy, arrayz, nvar1, nvar2, nvar3):
    """
    Create a TH3F histogram and fill it with three variables from a dataframe.
    """
    lenx = len(arrayx) - 1
    leny = len(arrayy) - 1
    lenz = len(arrayz) - 1

    histo = TH3F(titlehist, titlehist, lenx, arrayx, leny, arrayy, lenz, arrayz)
    histo.Sumw2()
    df_rd = df_[[nvar1, nvar2, nvar3]]
    arr3 = df_rd.values
    fill_hist(histo, arr3)
    return histo

def build2dhisto(titlehist, arrayx, arrayy):
    """
    Create a TH2 histogram from two axis arrays.
    """
    lenx = len(arrayx) - 1
    leny = len(arrayy) - 1

    histo = TH2F(titlehist, titlehist, lenx, arrayx, leny, arrayy)
    histo.Sumw2()
    return histo

def fill2dhist(df_, histo, nvar1, nvar2):
    """
    Fill a TH2 histogram with two variables from a dataframe.
    """
    df_rd = df_[[nvar1, nvar2]]
    arr2 = df_rd.values
    fill_hist(histo, arr2)
    return histo

def load_root_style_simple():
    """
    Set basic ROOT style for histograms
    """
    gStyle.SetOptStat(0)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)

def load_root_style():
    """
    Set more advanced ROOT style for histograms
    """
    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetTitleOffset(1.15, "y")
    gStyle.SetTitleFont(42, "xy")
    gStyle.SetLabelFont(42, "xy")
    gStyle.SetTitleSize(0.042, "xy")
    gStyle.SetLabelSize(0.035, "xy")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

def scatterplotroot(dfevt, nvar1, nvar2, nbins1, min1, max1, nbins2, min2, max2):
    """
    Make TH2F scatterplot between two variables from dataframe
    """
    hmult1_mult2 = TH2F(nvar1 + nvar2, nvar1 + nvar2, nbins1, min1, max1, nbins2, min2, max2)
    dfevt_rd = dfevt[[nvar1, nvar2]]
    arr2 = dfevt_rd.values
    fill_hist(hmult1_mult2, arr2)
    return hmult1_mult2

def find_axes_limits(histos, use_log_y=False):
    """
    Finds common axes limits for list of histograms provided
    """
    # That might be considered to be a hack since it now only has a chance to work
    # reasonably well if there is at least one histogram.
    max_y = min([h.GetMinimum() for h in histos if isinstance(h, TH1)])
    min_y = min([h.GetMaximum() for h in histos if isinstance(h, TH1)])
    if not min_y > 0. and use_log_y:
        min_y = 10.e-9

    max_x = max([h.GetXaxis().GetXmax() for h in histos])
    min_x = max([h.GetXaxis().GetXmin() for h in histos])

    for h in histos:
        if not isinstance(h, TH1):
            # That might be considered to be a hack since it now only has a chance to work
            # reasonably well if there is at least one histogram.
            continue
        min_x = min(min_x, h.GetXaxis().GetXmin())
        max_x = max(max_x, h.GetXaxis().GetXmax())
        min_y_tmp = h.GetBinContent(h.GetMinimumBin())
        if min_y_tmp > 0. and use_log_y or not use_log_y:
            min_y = min(min_y, h.GetBinContent(h.GetMinimumBin()))
        max_y = max(max_y, h.GetBinContent(h.GetMaximumBin()))

    return min_x, max_x, min_y, max_y

def style_histograms(histos, linestyles=None, markerstyles=None, colors=None, linewidths=None,
                     fillstyles=None, fillcolors=None):
    """
    Loops over given line- and markerstyles as well as colors applying them to the given list
    of histograms. The list of histograms might be larger than the styles provided. In that case
    the styles start again
    """
    if linestyles is None:
        linestyles = [1, 1, 1, 1]
    if markerstyles is None:
        markerstyles = [2, 4, 5, 32]
    if colors is None:
        colors = [kBlack, kRed, kGreen + 2, kBlue]
    if linewidths is None:
        linewidths = [1]
    if fillstyles is None:
        fillstyles = [0]
    if fillcolors is None:
        fillcolors = [kWhite]

    for i, h in enumerate(histos):
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineStyle(linestyles[i % len(linestyles)])
        h.SetMarkerStyle(markerstyles[i % len(markerstyles)])
        h.SetMarkerColor(colors[i % len(colors)])
        h.SetLineWidth(linewidths[i % len(linewidths)])
        h.SetFillStyle(fillstyles[i % len(fillstyles)])
        h.SetFillColor(fillcolors[i % len(fillcolors)])
        h.GetXaxis().SetTitleSize(0.02)
        h.GetXaxis().SetTitleSize(0.02)
        h.GetYaxis().SetTitleSize(0.02)

def divide_all_by_first(histos):
    """
    Divides all histograms in the list by the first one in the list and returns the
    divided histograms in the same order
    """

    histos_ratio = []
    for h in histos:
        histos_ratio.append(h.Clone(f"{h.GetName()}_ratio"))
        histos_ratio[-1].Divide(histos[0])

    return histos_ratio

def divide_by_eachother(histos1, histos2, scale=None, rebin2=None):
    """
    Divides all histos1 by histos2 and returns the
    divided histograms in the same order
    """

    if len(histos1) != len(histos2):
        get_logger().fatal("Number of histograms mismatch, %i vs. %i", \
                            len(histos1), len(histos2))

    histos_ratio = []
    for i, _ in enumerate(histos1):

        if rebin2 is not None:
            rebin = array('d', rebin2)
            histos1[i] = histos1[i].Rebin(len(rebin2)-1, f"{histos1[i].GetName()}_rebin", rebin)
            histos2[i] = histos2[i].Rebin(len(rebin2)-1, f"{histos2[i].GetName()}_rebin", rebin)

        if scale is not None:
            histos1[i].Scale(1./scale[0])
            histos2[i].Scale(1./scale[1])

        histos_ratio.append(histos1[i].Clone(f"{histos1[i].GetName()}_ratio"))
        histos_ratio[-1].Divide(histos2[i])

    return histos_ratio

def divide_by_eachother_barlow(histos1, histos2, scale=None, rebin2=None):
    """
    Divides all histos1 by histos2 using Barlow for stat. unc. and returns the
    divided histograms in the same order
    """

    if len(histos1) != len(histos2):
        get_logger().fatal("Number of histograms mismatch, %i vs. %i", \
                            len(histos1), len(histos2))

    histos_ratio = []
    for i, _ in enumerate(histos1):

        if rebin2 is not None:
            rebin = array('d', rebin2)
            histos1[i] = histos1[i].Rebin(len(rebin2)-1, f"{histos1[i].GetName()}_rebin", rebin)
            histos2[i] = histos2[i].Rebin(len(rebin2)-1, f"{histos2[i].GetName()}_rebin", rebin)

        if scale is not None:
            histos1[i].Scale(1./scale[0])
            histos2[i].Scale(1./scale[1])

        stat1 = []
        stat2 = []
        for j in range(histos1[i].GetNbinsX()):
            stat1.append(histos1[i].GetBinError(j+1) / histos1[i].GetBinContent(j+1))
            stat2.append(histos2[i].GetBinError(j+1) / histos2[i].GetBinContent(j+1))

        histos_ratio.append(histos1[i].Clone(f"{histos1[i].GetName()}_ratio"))
        histos_ratio[-1].Divide(histos2[i])

        for j in range(histos_ratio[-1].GetNbinsX()):
            statunc = math.sqrt(abs(stat1[j] * stat1[j] - stat2[j] * stat2[j]))
            histos_ratio[-1].SetBinError(j+1, histos_ratio[-1].GetBinContent(j+1) * statunc)

    return histos_ratio

def divide_all_by_first_multovermb(histos):
    """
    Divides all histograms in the list by the first one in the list and returns the
    divided histograms in the same order
    """

    histos_ratio = []
    err = []
    for h in histos:
        histos_ratio.append(h.Clone(f"{h.GetName()}_ratio"))

        stat = []
        for j in range(h.GetNbinsX()):
            stat.append(h.GetBinError(j+1) / h.GetBinContent(j+1))
        err.append(stat)
        histos_ratio[-1].Divide(histos[0])

        for j in range(h.GetNbinsX()):
            statunc = math.sqrt(abs(err[-1][j] * err[-1][j] - err[0][j] * err[0][j]))
            histos_ratio[-1].SetBinError(j+1, histos_ratio[-1].GetBinContent(j+1) * statunc)

    return histos_ratio

def put_in_pad(pad, use_log_y, histos, title="", x_label="", y_label="", yrange=None, **kwargs):
    """
    Providing a TPad this plots all given histograms in that pad adjusting the X- and Y-ranges
    accordingly.
    """

    draw_options = kwargs.get("draw_options", None)

    min_x, max_x, min_y, max_y = find_axes_limits(histos, use_log_y)
    pad.SetLogy(use_log_y)
    pad.cd()
    scale_frame_y = (0.01, 100.) if use_log_y else (0.7, 1.2)
    if yrange is None:
        yrange = [min_y * scale_frame_y[0], max_y * scale_frame_y[1]]
    frame = pad.DrawFrame(min_x, yrange[0], max_x, yrange[1],
                          f"{title};{x_label};{y_label}")
    frame.GetYaxis().SetTitleOffset(1.2)
    pad.SetTicks()
    if draw_options is None:
        draw_options = ["" for _ in histos]
    for h, o in zip(histos, draw_options):
        h.Draw(f"same {o}")

#pylint: disable=too-many-statements
def plot_histograms(histos, use_log_y=False, ratio_=False, legend_titles=None, title="", x_label="",
                    y_label_up="", y_label_ratio="", save_path="./plot.eps", **kwargs):
    """
    Throws all given histograms into one canvas. If desired, a ratio plot will be added.
    """
    gStyle.SetOptStat(0)
    justratioplot = False
    yrange = None
    if isinstance(ratio_, list):
        ratio = ratio_[0]
        justratioplot = ratio_[1]
        yrange = ratio_[2]
    else:
        justratioplot = ratio_
        ratio = ratio_

    linestyles = kwargs.get("linestyles", None)
    markerstyles = kwargs.get("markerstyles", None)
    colors = kwargs.get("colors", None)
    draw_options = kwargs.get("draw_options", None)
    linewidths = kwargs.get("linewidths", None)
    fillstyles = kwargs.get("fillstyles", None)
    fillcolors = kwargs.get("fillcolors", None)
    canvas_name = kwargs.get("canvas_name", "Canvas")
    style_histograms(histos, linestyles, markerstyles, colors, linewidths, fillstyles, fillcolors)

    canvas = TCanvas('canvas', canvas_name, 800, 800)
    pad_up_start = 0.4 if ratio else 0.

    pad_up = TPad("pad_up", "", 0., pad_up_start, 1., 1.)
    if ratio:
        pad_up.SetBottomMargin(0.)
    pad_up.Draw()

    x_label_up_tmp = x_label if not ratio else ""
    put_in_pad(pad_up, use_log_y, histos, title, x_label_up_tmp, y_label_up,
               yrange, draw_options=draw_options)

    pad_up.cd()
    legend = None
    if legend_titles is not None:
        if justratioplot:
            legend = TLegend(.2, .65, .6, .85)
        else:
            legend = TLegend(.45, .65, .85, .85)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.02)
        for h, l in zip(histos, legend_titles):
            if l is not None:
                legend.AddEntry(h, l)
        legend.Draw()

    canvas.cd()
    pad_ratio = None
    histos_ratio = None

    if ratio and justratioplot is False:
        histos_ratio = divide_all_by_first(histos)
        pad_ratio = TPad("pad_ratio", "", 0., 0.05, 1., pad_up_start)
        pad_ratio.SetTopMargin(0.)
        pad_ratio.SetBottomMargin(0.3)
        pad_ratio.Draw()

        put_in_pad(pad_ratio, False, histos_ratio, "", x_label, y_label_ratio)

    canvas.SaveAs(save_path)

    index = save_path.rfind(".")

    # Save also everything into a ROOT file
    root_save_path = save_path[:index] + ".root"
    root_file = TFile.Open(root_save_path, "RECREATE")
    for h in histos:
        h.Write()
    canvas.Write()
    root_file.Close()

    canvas.Close()

def save_histograms(histos, save_path="./plot.root"):
    """
    Save everything into a ROOT file for offline plotting
    """
    index = save_path.rfind(".")

    # Save also everything into a ROOT file
    root_save_path = save_path[:index] + ".root"
    root_file = TFile.Open(root_save_path, "RECREATE")
    for h in histos:
        h.Write()
    root_file.Close()
