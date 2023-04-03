#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import pickle
import h5py as h5
import os
import argparse
import pyBigWig

'''
Script for preprocessing input epigenomic tracks (from .bigWig file)
and target Hi-C data (from diagonal.txt file).

Usage example: 

python /train_epiphany/prepare_data.py \
--epi_input_dir /data/leslie/yangr2/setd2/train_epiphany/GM12878_files \
--hic_input_dir /data/leslie/yangr2/setd2/train_epiphany/GM12878_hic/normcounts \
--target_dir /data/leslie/yangr2/setd2/train_epiphany/GM12878_processed \
--cell_type "GM12878" \
--file_name "ATAC-H3K36me3-H3K27ac-H3K27me3"
'''

parser = argparse.ArgumentParser(description="Set-up data preparations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epi_input_dir", help="Load prepared bigWig files.", type=str)
parser.add_argument("--hic_input_dir", help="Load prepared hic diagonal files.", type=str)
parser.add_argument("--target_dir", help="Output path. Path to store preprocessed data.", 
                    type=str)
parser.add_argument("--cell_type", help="Prepare dataset for this cell type.", type=str)
parser.add_argument("--hic_resolution", help="Resolution of the Hi-C contact map.", type=str)
parser.add_argument("--normalization_type", help="Which normalization to use as target,",
                    choices=["z-value", "obs-over-exp"])
parser.add_argument("--epi_order", help="Order of the epigenomic tracks for input.", nargs="*")
# ["ATAC", "H3K36me3", "H3K27ac", "H3K4me3", "H3K4me1"]
parser.add_argument("--epi_resolution", help="Resolution to extract the epigenomic tracks", 
                    type=int, nargs="?", const=100, default=100)
parser.add_argument("--file_name", help="Define processed file name to save as",
                    type=str, default="")
args = parser.parse_args()
config = vars(args)
print(config)

EPI_INPUT_DIR = config['epi_input_dir']
HIC_INPUT_DIR = config['hic_input_dir']
TARGET_DIR = config['target_dir']
EPI_ORDER = config['epi_order']
EPI_RESOLUTION = config['epi_resolution']
HIC_RESOLUTION = config['hic_resolution']
CELL_TYPE = [config['cell_type']]
NORM_TYPE = config['normalization_type']
FILE_NAME_PREFIX = config['file_name']

#########################
#     Load functions    #
#########################

def load_epitracks(input_dir = EPI_INPUT_DIR, chrom = "chr1", epi_order = EPI_ORDER, 
                 resolution = EPI_RESOLUTION):
    """Extract values from each .bw files. 
    Args:
        dir: directory to epigenomic tracks (bigWig files)
        chrom: chromosome to extract from (chr1, chr2, ...)
        resolution: in which resolution to extract from the tracks. Default is 100bp. 
    
    Returns:
        list of lists containing the 1D epigenmic tracks in certain order.
    """
    files = [i for i in os.listdir(input_dir) if "bw" in i]
    print(files)
    idx = [[i for i, s in enumerate(files) if chip in s][0] for chip in epi_order]
    files = [files[i] for i in idx]
    bw_list = []
    for file in files:
        # bwfile = os.path.join(dir, file)
        bwfile = f"{input_dir}/{file}"
        print(bwfile)
        bw = pyBigWig.open(bwfile)

        value_list = []
        for i in list(range(0, bw.chroms()[chrom] - resolution, resolution)):
            value_list.append(bw.stats(chrom, i, i + resolution)[0])

        value_list = [0 if v is None else v for v in value_list]
        bw_list.append(value_list)

    return bw_list

def make_chip(input_dir = EPI_INPUT_DIR, target_dir = TARGET_DIR, 
              cell_types = CELL_TYPE, name_prefix = FILE_NAME_PREFIX,
              resolution = EPI_RESOLUTION, epi_order = EPI_ORDER):
    chroms = ["chr" + str(i) for i in range(1, 23)][::-1]
    for cell in cell_types:
        chipseq_path = input_dir
        inputs = {}
        for chr in chroms:
            print("Loading", chr)
            bw_list = load_epitracks(chipseq_path, chrom=chr, 
                                     resolution = resolution,
                                     epi_order = epi_order)
            inputs[chr] = np.array(bw_list)
            print(np.array(bw_list).shape)
        save_path_X = f"{target_dir}/{cell}_{name_prefix}_X.pickle"
        with open(save_path_X, "wb") as handle:
            pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_labels(input_dir = HIC_INPUT_DIR, target_dir = TARGET_DIR,
                cell_types = CELL_TYPE, dtype = NORM_TYPE,
                name_prefix = FILE_NAME_PREFIX, res = HIC_RESOLUTION):
    chroms = ["chr" + str(i) for i in range(1, 23)]
    labels, inputs = {}, {}
    diag_list_dir = input_dir
    for cell in cell_types:
        for chr in chroms:
            print("Loading", chr)
            diag_list_path = f"{diag_list_dir}/HiC_{cell}_{chr}_{dtype}_{res}_diagonal.txt"
            with open(diag_list_path, "rb") as fp:
                diagonal_list = pickle.load(fp)
            diag_log_list = [(np.array(i)).tolist() for i in diagonal_list]
            labels[chr] = diag_log_list
    save_path_y = f"{target_dir}/{cell}_{name_prefix}_y.pickle"
    with open(save_path_y, "wb") as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


#########################
#     Script running    #
#########################

def main():
    args = parser.parse_args()
    config = vars(args)
    print(config)
    EPI_INPUT_DIR = config['epi_input_dir']
    HIC_INPUT_DIR = config['hic_input_dir']
    TARGET_DIR = config['target_dir']
    EPI_ORDER = config['epi_order']
    EPI_RESOLUTION = config['epi_resolution']
    HIC_RESOLUTION = config['hic_resolution']
    FILE_NAME_PREFIX = config['file_name']
    CELL_TYPE = [config['cell_type']]
    NORM_TYPE = config['normalization_type']
    make_chip(input_dir = EPI_INPUT_DIR, target_dir = TARGET_DIR, 
              cell_types = CELL_TYPE, dtype = NORM_TYPE,
              name_prefix = FILE_NAME_PREFIX, resolution = EPI_RESOLUTION,
              epi_order = EPI_ORDER)
    make_labels(input_dir = HIC_INPUT_DIR, target_dir = TARGET_DIR,
                cell_types = CELL_TYPE, dtype = NORM_TYPE,
                name_prefix = FILE_NAME_PREFIX, res = HIC_RESOLUTION)
    
if __name__ == '__main__':
    main()
