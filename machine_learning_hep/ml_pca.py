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
Methods to: perform PCA analysis
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_pcadataframe_pca(dataframe, n_pca):
    data_values = dataframe.values
    pca = PCA(n_pca)
    principalComponent = pca.fit_transform(data_values)
    pca_name_list = []
    for i_pca in range(n_pca):
        pca_name_list.append("princ_comp_%d" % (i_pca+1))
    pca_dataframe = pd.DataFrame(data=principalComponent, columns=pca_name_list)
    return pca_dataframe, pca


def getdataframe_standardised(dataframe):
    listheaders = list(dataframe.columns.values)
    data_values = dataframe.values
    data_values_std = StandardScaler().fit_transform(data_values)
    dataframe_std = pd.DataFrame(data=data_values_std, columns=listheaders)
    return dataframe_std
