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
Script containing all helper functions
e.g. processing files, creating objects, calculating physical quantities.
"""
import bz2
import gzip
import lzma
import multiprocessing as mp
import os
import pickle
from array import array
from datetime import datetime

import lz4
import pandas as pd
import yaml
from ROOT import TNtuple, TFile  # pylint: disable=import-error,no-name-in-module

from machine_learning_hep.logger import get_logger


def parallelizer(function, argument_list, maxperchunk, max_n_procs=2):
    """
    A centralized version for quickly parallelizing basically identical to what can found in
    the Processer. It could also rely on this one.
    """
    chunks = [argument_list[x:x+maxperchunk] \
              for x in range(0, len(argument_list), maxperchunk)]
    for chunk in chunks:
        print("Processing new chunck size=", maxperchunk)
        pool = mp.Pool(max_n_procs)
        _ = [pool.apply_async(function, args=chunk[i]) for i in range(len(chunk))]
        pool.close()
        pool.join()

def openfile(filename, attr):
    """
    Open file with different compression types
    """
    if filename.lower().endswith('.bz2'):
        return bz2.BZ2File(filename, attr)
    if filename.lower().endswith('.xz'):
        return lzma.open(filename, attr)
    if filename.lower().endswith('.gz'):
        return gzip.open(filename, attr)
    if filename.lower().endswith('.lz4'):
        return lz4.frame.open(filename, attr)
    return open(filename, attr)

# pylint: disable=too-many-nested-blocks
def list_folders(main_dir, filenameinput, maxfiles):
    """
    List all files in a subdirectory structure
    """
    if not os.path.isdir(main_dir):
        print("the input directory =", main_dir, "doesnt exist")
    list_subdir0 = os.listdir(main_dir)
    listfolders = list()
    for subdir0 in list_subdir0:
        subdir0full = os.path.join(main_dir, subdir0)
        if os.path.isdir(subdir0full):
            list_subdir1 = os.listdir(subdir0full)
            for subdir1 in list_subdir1:
                subdir1full = os.path.join(subdir0full, subdir1)
                if os.path.isdir(subdir1full):
                    list_files_ = os.listdir(subdir1full)
                    for myfile in list_files_:
                        filefull = os.path.join(subdir1full, myfile)
                        if os.path.isfile(filefull) and \
                        myfile == filenameinput:
                            listfolders.append(os.path.join(subdir0, subdir1))
    if maxfiles is not -1:
        listfolders = listfolders[:maxfiles]
    return  listfolders

def create_folder_struc(maindir, listpath):
    """
    Reproduce the folder structure as input
    """
    for path in listpath:
        path = path.split("/")

        folder = os.path.join(maindir, path[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(folder, path[1])
        if not os.path.exists(folder):
            os.makedirs(folder)

def checkdirlist(dirlist):
    """
    Checks if list of folder already exist, to not overwrite by accident
    """
    exfolders = 0
    for _, mydir in enumerate(dirlist):
        if os.path.exists(mydir):
            print("rm -rf ", mydir)
            exfolders = exfolders - 1
    return exfolders

def checkdir(mydir):
    """
    Checks if folder already exist, to not overwrite by accident
    """
    exfolders = 0
    if os.path.exists(mydir):
        print("rm -rf ", mydir)
        exfolders = -1
    return exfolders

def checkmakedirlist(dirlist):
    """
    Makes directories from list using 'mkdir'
    """
    for _, mydir in enumerate(dirlist):
        print("creating folder ", mydir)
        os.makedirs(mydir)

def checkmakedir(mydir):
    """
    Makes directory using 'mkdir'
    """
    print("creating folder ", mydir)
    os.makedirs(mydir)

def appendfiletolist(mylist, namefile):
    """
    Append filename to list
    """
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder, mylist):
    """
    Append base foldername to paths in list
    """
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder, mylistfolder, namefile):
    """
    Appends base foldername + filename in list
    """
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles

def createstringselection(var, low, high):
    """
    Create string of main dataframe selection (e.g. pT)
    Used as suffix for storing ML plots
    """
    string_selection = "dfselection_"+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection

def merge_method(listfiles, namemerged):
    """
    Merge list of dataframes into one
    """
    dflist = []
    for myfilename in listfiles:
        myfile = openfile(myfilename, "rb")
        df = pickle.load(myfile)
        dflist.append(df)
    dftot = pd.concat(dflist)
    pickle.dump(dftot, openfile(namemerged, "wb"), protocol=4)

def parse_yaml(filepath):
    """
    Parse a YAML file and return dictionary
    Args:
        filepath: Path to the YAML file to be parsed.
    """
    if not os.path.isfile(filepath):
        get_logger().critical("YAML file %s does not exist.", filepath)
    with open(filepath) as f:
        return yaml.safe_load(f)

def dump_yaml_from_dict(to_yaml, path):
    path = os.path.expanduser(path)
    with open(path, "w") as stream:
        yaml.safe_dump(to_yaml, stream, default_flow_style=False, allow_unicode=False)

def print_dict(to_be_printed, indent=0, skip=None):
    for key, value in to_be_printed.items():
        if isinstance(skip, list) and key in skip:
            continue
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def make_file_path(directory, filename, extension, prefix=None, suffix=None):
    """
    Construct a common path+filename+suffix from args
    """
    if prefix is not None:
        filename = make_pre_suffix(prefix) + "_" + filename
    if suffix is not None:
        filename = filename + "_" + make_pre_suffix(suffix)
    extension = extension.replace(".", "")
    return os.path.join(directory, filename + "." + extension)

def make_pre_suffix(args):
    """
    Construct a common file suffix from args
    """
    try:
        _ = iter(args)
    except TypeError:
        args = [args]
    else:
        if isinstance(args, str):
            args = [args]
    return "_".join(args)

def make_message_notfound(name, location=None):
    """
    Return a formatted error message for not found or not properly loaded objects
    stating the name and optionally the location.
    """
    if location is not None:
        return "Error: Failed to get %s in %s" % (name, location)
    return "Error: Failed to get %s" % name

def get_timestamp_string():
    """
    Get timestamp, used for temporary files (like the 'hadd' ones)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def mergerootfiles(listfiles, mergedfile, tmp_dir):
    """
    Using ROOT's 'hadd' utility, to merge output rootfiles from analyses steps
    """
    def divide_chunks(list_to_split, chunk_size):
        for i in range(0, len(list_to_split), chunk_size):
            yield list_to_split[i:i + chunk_size]

    tmp_files = []
    if len(listfiles) > 500:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        for i, split_list in enumerate(divide_chunks(listfiles, 500)):
            tmp_files.append(os.path.join(tmp_dir, f"hadd_tmp_merged{i}.root"))
            outstring = " ".join(split_list)
            os.system("hadd -f -j 30 %s  %s " % (tmp_files[-1], outstring))
    else:
        tmp_files = listfiles

    outstring = " ".join(tmp_files)
    os.system("hadd -f -j 30 %s  %s " % (mergedfile, outstring))

def fill_ntuple(tupname, data, names):
    """
    Create and fill ROOT NTuple with the data sample.
      tupname : name of the NTuple
      data : data sample
      names : names of the NTuple variables
    """
    variables = ""
    for n in names:
        variables += "%s:" % n
    variables = variables[:-1]
    values = len(names)*[0.]
    avalues = array('f', values)
    nt = TNtuple(tupname, "", variables)
    for d in data:
        for i in range(len(names)):
            avalues[i] = d[i]
        nt.Fill(avalues)
    nt.Write()

def write_tree(filename, treename, dataframe):
    """
    Write TTree from dataframe with in filanem
    """
    listvar = list(dataframe)
    values = dataframe.values
    fout = TFile.Open(filename, "recreate")
    fout.cd()
    fill_ntuple(treename, values, listvar)

def make_latex_table(column_names, row_names, rows, caption=None, save_path="./table.tex"):
    """
    Store information in table format in tex file
    """
    caption = caption if caption is not None else "Caption"
    with open(save_path, "w") as f:
        f.write("\\documentclass{article}\n\n")
        f.write("\\usepackage[margin=0.7in]{geometry}\n")
        f.write("\\usepackage[parfill]{parskip}\n")
        f.write("\\usepackage{rotating}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{sidewaystable}\n")
        f.write("\\centering\n")
        # As many columns as we need
        columns = "|".join(["c"] * (len(column_names) + 1))
        f.write("\\begin{tabular}{" + columns + "}\n")
        f.write("\\hline\n")
        columns = "&".join([""] + column_names)
        columns = columns.replace("_", "\\_")
        f.write(columns + "\\\\\n")
        f.write("\\hline\\hline\n")
        for rn, row in zip(row_names, rows):
            row_string = "&".join([rn] + row)
            row_string = row_string.replace("_", "\\_")
            f.write(row_string + "\\\\\n")
        f.write("\\end{tabular}\n")
        caption = caption.replace("_", "\\_")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\end{sidewaystable}\n")
        f.write("\\end{document}\n")
