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
Script containing all helper functions related to systematics

Script also contains the "class Errors", used for systematic uncertainties (to
replace AliHFSystErr from AliPhysics).
"""
# pylint: disable=too-many-lines
from array import array

import numpy as np
# pylint: disable=import-error, no-name-in-module
from ROOT import TGraphAsymmErrors

from utilities import parse_yaml, dump_yaml_from_dict
from logger import get_logger


# pylint: disable=too-many-branches
def calc_systematic_multovermb(errnum_list, errden_list, n_bins, justfd=-99):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Ds(mult) / Ds(MB). Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list.errors.values())[0]) or \
     n_bins != len(list(errden_list.errors.values())[0]):
        get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                            n_bins, len(list(errnum_list.errors.values())[0]), \
                            len(list(errden_list.errors.values())[0]))

    listimpl = ["yield", "cut", "pid", "feeddown_mult", "feeddown_mult_spectra", "trigger", \
                "multiplicity_interval", "multiplicity_weights", "track", "ptshape", \
                "feeddown_NB", "sigmav0", "branching_ratio"]

    j = 0
    for (_, errnum), (_, errden) in zip(errnum_list.errors.items(), errden_list.errors.items()):
        for i in range(n_bins):

            if errnum_list.names[j] not in listimpl:
                get_logger().fatal("Unknown systematic name: %s", errnum_list.names[j])
            if errnum_list.names[j] != errden_list.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum.names[j], errden.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list.names[j] == "yield" and justfd is not True:
                    #Partially correlated, take largest
                    tot_list[i][nb] += max(errnum[i][nb], errden[i][nb]) \
                                        * max(errnum[i][nb], errden[i][nb])
                elif errnum_list.names[j] == "cut" and justfd is not True:
                    #Partially correlated, take largest
                    tot_list[i][nb] += max(errnum[i][nb], errden[i][nb]) \
                                        * max(errnum[i][nb], errden[i][nb])
                elif errnum_list.names[j] == "pid" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "feeddown_mult" and justfd is not False:
                    #Assign directly from multiplicity case, no syst for MB
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "feeddown_mult_spectra" and justfd is not False:
                    #Ratio here, skip spectra syst
                    pass
                elif errnum_list.names[j] == "trigger" and justfd is not True:
                    #Assign directly from multiplicity case, no syst for MB
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "multiplicity_interval" and justfd is not True:
                    #FD: estimated using 7TeV strategy directly for ratio
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "multiplicity_weights" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "track" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "ptshape" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "feeddown_NB" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "sigmav0" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list.names[j] == "branching_ratio" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def calc_systematic_mesonratio(errnum_list, errden_list, n_bins, justfd=-99):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Ds(MB or mult) / D0(MB or mult).
    Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list.errors.values())[0]) or \
     n_bins != len(list(errden_list.errors.values())[0]):
        get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                            n_bins, len(list(errnum_list.errors.values())[0]), \
                            len(list(errden_list.errors.values())[0]))

    listimpl = ["yield", "cut", "pid", "feeddown_mult", "feeddown_mult_spectra", "trigger", \
                "multiplicity_interval", "multiplicity_weights", "track", "ptshape", \
                "feeddown_NB", "sigmav0", "branching_ratio"]

    j = 0
    for (_, errnum), (_, errden) in zip(errnum_list.errors.items(), errden_list.errors.items()):
        for i in range(n_bins):

            if errnum_list.names[j] not in listimpl:
                get_logger().fatal("Unknown systematic name: %s", errnum_list.names[j])
            if errnum_list.names[j] != errden_list.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum_list.names[j], errden_list.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list.names[j] == "yield" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "cut" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "pid" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "feeddown_mult_spectra" and justfd is not False:
                    #Fully correlated
                    ynum = errnum_list.errors["feeddown_NB"][i][4]
                    yden = errden_list.errors["feeddown_NB"][i][4]
                    #Relative uncertainties stored, make absolute
                    ynuml = ynum - ynum * errnum[i][2]
                    ydenl = yden - yden * errden[i][2]
                    ynumh = ynum + ynum * errnum[i][3]
                    ydenh = yden + yden * errden[i][3]
                    rat = [ynuml / ydenl, ynum / yden, ynumh / ydenh]
                    minsys = min(rat)
                    maxsys = max(rat)
                    if nb == 2:
                        tot_list[i][nb] += (rat[1] - minsys) * (rat[1] - minsys) / (rat[1] * rat[1])
                    if nb == 3:
                        tot_list[i][nb] += (maxsys - rat[1]) * (maxsys - rat[1]) / (rat[1] * rat[1])
                elif errnum_list.names[j] == "feeddown_mult" and justfd is not False:
                    #Spectra here, skip ratio systematic
                    pass
                elif errnum_list.names[j] == "trigger" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "feeddown_NB" and justfd is not False:
                    #Fully correlated under assumption central Fc value stays within Nb syst
                    ynum = errnum[i][4]
                    yden = errden[i][4]
                    #Absolute uncertainties stored
                    ynuml = ynum - errnum[i][2]
                    ydenl = yden - errden[i][2]
                    ynumh = ynum + errnum[i][3]
                    ydenh = yden + errden[i][3]
                    rat = [ynuml / ydenl, ynum / yden, ynumh / ydenh]
                    minsys = min(rat)
                    maxsys = max(rat)
                    if nb == 2:
                        tot_list[i][nb] += (rat[1] - minsys) * (rat[1] - minsys) / (rat[1] * rat[1])
                    if nb == 3:
                        tot_list[i][nb] += (maxsys - rat[1]) * (maxsys - rat[1]) / (rat[1] * rat[1])
                elif errnum_list.names[j] == "multiplicity_weights" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "track" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "ptshape" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "multiplicity_interval" and justfd is not True:
                    #NB: Assuming ratio: 3prongs over 2prongs here! 2prong part cancels
                    #We use 1/3 of systematic of numerator
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] / 9
                elif errnum_list.names[j] == "sigmav0" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list.names[j] == "branching_ratio" and justfd is not True:
                    #Uncorrelated, but usually not plotted in boxes, so pass
                    pass
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

def calc_systematic_mesondoubleratio(errnum_list1, errnum_list2, errden_list1, \
                                     errden_list2, n_bins, dropbins=None, justfd=-99):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Lc/D0_mult-i / Lc/D0_mult-j.
    Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list1.errors.values())[0]) or \
     n_bins != len(list(errden_list1.errors.values())[0]):
        if dropbins is None:
            get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                                n_bins, len(list(errnum_list1.errors.values())[0]), \
                                len(list(errden_list1.errors.values())[0]))

    listimpl = ["yield", "cut", "pid", "feeddown_mult", "feeddown_mult_spectra", "trigger", \
                "multiplicity_interval", "multiplicity_weights", "track", "ptshape", \
                "feeddown_NB", "sigmav0", "branching_ratio"]

    j = 0
    for (_, errnum1), (_, errnum2), (_, errden1), (_, errden2) in zip(errnum_list1.errors.items(), \
                                                                      errnum_list2.errors.items(), \
                                                                      errden_list1.errors.items(), \
                                                                      errden_list2.errors.items()):
        for i in range(n_bins):

            inum = i
            iden = i
            if dropbins is not None:
                inum = dropbins[0][i]
                iden = dropbins[1][i]

            if errnum_list1.names[j] not in listimpl:
                get_logger().fatal("Unknown systematic name: %s", errnum_list1.names[j])
            if errnum_list1.names[j] != errden_list2.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum_list1.names[j], errden_list2.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list1.names[j] == "yield" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum1[inum][nb] * errnum1[inum][nb] + \
                                       errnum2[inum][nb] * errnum2[inum][nb] + \
                                       errden1[iden][nb] * errden1[iden][nb] + \
                                       errden2[iden][nb] * errden2[iden][nb]
                elif errnum_list1.names[j] == "cut" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum1[inum][nb] * errnum1[inum][nb] + \
                                       errnum2[inum][nb] * errnum2[inum][nb] + \
                                       errden1[iden][nb] * errden1[iden][nb] + \
                                       errden2[iden][nb] * errden2[iden][nb]
                elif errnum_list1.names[j] == "pid" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "feeddown_mult_spectra" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "feeddown_mult" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "trigger" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "feeddown_NB" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "multiplicity_weights" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "track" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "ptshape" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum1[inum][nb] * errnum1[inum][nb] + \
                                       errnum2[inum][nb] * errnum2[inum][nb] + \
                                       errden1[iden][nb] * errden1[iden][nb] + \
                                       errden2[iden][nb] * errden2[iden][nb]
                elif errnum_list1.names[j] == "multiplicity_interval" and justfd is not True:
                    #NB: Assuming ratio: 3prongs over 2prongs here! 2prong part cancels
                    #We use 1/3 of systematic of numerator
                    tot_list[i][nb] += errden1[iden][nb] * errden1[iden][nb] / 9
                elif errnum_list1.names[j] == "sigmav0" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list1.names[j] == "branching_ratio" and justfd is not True:
                    #Uncorrelated, but usually not plotted in boxes, so pass
                    pass
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

# pylint: disable=too-many-nested-blocks
class Errors:
    """
    Errors corresponding to one histogram
    Relative errors are assumed
    """
    def __init__(self, n_bins):
        # A dictionary of lists, lists will contain 4-tuples
        self.errors = {}
        # Number of errors per bin
        self.n_bins = n_bins
        # Names of systematic in order as they appear in self.errors
        self.names = {}
        # The logger...
        self.logger = get_logger()

    @staticmethod
    def make_symm_y_errors(*args):
        return [[0, 0, a, a] for a in args]

    @staticmethod
    def make_asymm_y_errors(*args):
        if len(args) % 2 != 0:
            get_logger().fatal("Need an even number ==> ((low, up) * n_central) of errors")
        return [[0, 0, args[i], args[i+1]] for i in range(0, len(args), 2)]


    @staticmethod
    def make_root_asymm(histo_central, error_list, **kwargs):
        """
        This takes a list of 4-tuples and a central histogram assumed to have number of bins
        corresponding to length of error_list
        """
        n_bins = histo_central.GetNbinsX()
        if n_bins != len(error_list):
            get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i",
                               n_bins, len(error_list))
        rel_x = kwargs.get("rel_x", True)
        rel_y = kwargs.get("rel_y", True)
        const_x_err = kwargs.get("const_x_err", None)
        const_y_err = kwargs.get("const_y_err", None)

        x_low = None
        x_up = None
        y_low = None
        y_up = None
        # Make x up and down
        if const_x_err is not None:
            x_up = array("d", [const_x_err] * n_bins)
            x_low = array("d", [const_x_err] * n_bins)
        elif rel_x is True:
            x_up = array("d", [err[1] * histo_central.GetBinCenter(b + 1) \
                    for b, err in enumerate(error_list)])
            x_low = array("d", [err[0] * histo_central.GetBinCenter(b + 1) \
                    for b, err in enumerate(error_list)])
        else:
            x_up = array("d", [err[1] for err in error_list])
            x_low = array("d", [err[0] for err in error_list])

        # Make y up and down
        if const_y_err is not None:
            y_up = array("d", [const_y_err] * n_bins)
            y_low = array("d", [const_y_err] * n_bins)
        elif rel_y is True:
            y_up = array("d", [err[3] * histo_central.GetBinContent(b + 1) \
                    for b, err in enumerate(error_list)])
            y_low = array("d", [err[2] * histo_central.GetBinContent(b + 1) \
                    for b, err in enumerate(error_list)])
        else:
            y_up = array("d", [err[3] for err in error_list])
            y_low = array("d", [err[2] for err in error_list])

        bin_centers = array("d", [histo_central.GetBinCenter(b + 1) for b in range(n_bins)])
        bin_contents = array("d", [histo_central.GetBinContent(b + 1) for b in range(n_bins)])

        return TGraphAsymmErrors(n_bins, bin_centers, bin_contents, x_low, x_up, y_low, y_up)

    @staticmethod
    def make_root_asymm_dummy(histo_central):
        n_bins = histo_central.GetNbinsX()
        bin_centers = array("d", [histo_central.GetBinCenter(b + 1) for b in range(n_bins)])
        bin_contents = array("d", [histo_central.GetBinContent(b + 1) for b in range(n_bins)])
        y_up = array("d", [0.] * n_bins)
        y_low = array("d", [0.] * n_bins)
        x_up = array("d", [0.] * n_bins)
        x_low = array("d", [0.] * n_bins)

        return TGraphAsymmErrors(n_bins, bin_centers, bin_contents, x_low, x_up, y_low, y_up)

    def add_errors(self, name, err_list):
        """
        err_list assumed to be a list of 4-tuples
        """
        if name in self.errors:
            self.logger.fatal("Error %s already registered", name)
        if len(err_list) != self.n_bins:
            self.logger.fatal("%i errors required, you want to push %i", self.n_bins, len(err_list))

        self.errors[name] = err_list.copy()

    def read(self, yaml_errors, extra_errors=None):
        """
        Read everything from YAML
        """
        error_dict = parse_yaml(yaml_errors)
        for name, errors in error_dict.items():
            if name == "names":
                self.names = errors.copy()
            else:
                self.add_errors(name, errors)
        if extra_errors is not None:
            self.errors.update(extra_errors)
            for key in extra_errors:
                self.names.append(key)

    def write(self, yaml_path):
        """
        Write everything from YAML
        """
        dump_yaml_from_dict(self.errors, yaml_path)

    def define_correlations(self):
        """
        Not yet defined
        """
        self.logger.warning("Function \"define_correlations\' not yet defined")

    def divide(self):
        """
        Not yet defined
        """
        self.logger.warning("Function \"divide\" not yet defined")

    def get_total(self):
        """
        Returns a list of total errors
        For now only add in quadrature and take sqrt
        """
        tot_list = [[0., 0., 0., 0.] for _ in range(self.n_bins)]
        for _, errors in enumerate(self.errors.values()):
            for i in range(self.n_bins):
                for nb in range(len(tot_list[i])):
                    tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
        tot_list = np.sqrt(tot_list)
        return tot_list

    def get_total_for_spectra_plot(self, justfd=-99):
        """
        Returns a list of total errors
        For now only add in quadrature and take sqrt
        """
        tot_list = [[0., 0., 0., 0.] for _ in range(self.n_bins)]
        for j, errors in enumerate(self.errors.values()):
            for i in range(self.n_bins):
                for nb in range(len(tot_list[i])):
                    if self.names[j] != "branching_ratio" and self.names[j] != "sigmav0" \
                      and self.names[j] != "feeddown_mult":

                        if justfd == -99:
                            tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                        elif justfd is True:
                            if self.names[j] == "feeddown_NB" \
                              or self.names[j] == "feeddown_mult_spectra":
                                tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                        elif justfd is False:
                            if self.names[j] != "feeddown_NB" \
                              and self.names[j] != "feeddown_mult_spectra":
                                tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                        else:
                            get_logger().fatal("Option for spectra systematic not valid")

        tot_list = np.sqrt(tot_list)
        return tot_list
