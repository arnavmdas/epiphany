#!/usr/bin/python
import numpy as np
import pickle
import math
import gzip
import argparse
from itertools import islice

'''Convert Hi-C data from HiC-DC+ normalized (txt.gz) format to diagonal.txt

Usage Example: 
[yangr2@lilac-ln02 script]$ screen
(base) bash-4.2$ bsub -n 2 -W 40:00 -R 'span[hosts=1] rusage[mem=64]' -Is /bin/bash
(base) bash-4.2$ source /home/yangr2/dnabert_environment/bin/activate
(dnabert_environment) (base) bash-4.2$ cd /data/leslie/yangr2/setd2/train_setd2/scripts/
(dnabert_environment) (base) bash-4.2$ python prepare_logFC_diagonal.py
'''


print("Converting data format from HiC-DC+ to diagonal.txt")


parser = argparse.ArgumentParser(description="Set-up data preparations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--norm_hic_file", help="Normalized Hi-C output file from HiC-DC+", type=str)
parser.add_argument("--chrom_len_file", help="Location of the chromosome length file", type=str)
parser.add_argument("--save_dir", help="Location to save the list", type=str)
parser.add_argument("--normalization_type", help="Which normalization to use as target,",
                    choices=["z-value", "obs-over-exp"])
parser.add_argument("--hic_resolution", help="Resolution of the Hi-C map being normalized", type=str)
parser.add_argument("--cell_type", help="Cell type of the Hi-C", type=str)


args = parser.parse_args()
config = vars(args)
print(config)
NORM_FILE = config['norm_hic_file']
CHROM_LEN_FILE = config['chrom_len_file']
SAVE_LOC = config['save_dir']
NORM_TYPE = config['normalization_type']
HIC_RESOLUTION = config['hic_resolution']
CELL_TYPE = config['cell_type']

#########################
#     Load functions    #
#########################

def file_conversion(chrom_len, norm_file = NORM_FILE,
                    dtype = NORM_TYPE, res = HIC_RESOLUTION, 
                    cell_type = CELL_TYPE, save_loc = SAVE_LOC):
    for chrom in ["chr" + range(1, 23)]:
        print("Now converting: ", norm_file, chrom)
        mat_dim = int(math.ceil(chrom_len[chrom]*1.0/res))
        contact_matrix = np.zeros((mat_dim, mat_dim))
        skip_row = 1
        with gzip.open(norm_file,'r') as fin:
            for line in islice(fin, skip_row, None):
                skip_row += 1
                line = line.decode('utf8')
                line_split = line.strip().split('\t')
                if line_split[0] == line_split[3] == chrom:
                    idx1 = line_split[1]
                    idx2 = line_split[4]
                    if dtype == "z-value":
                        value = line_split[10]
                    elif dtype == "obs-over-exp":
                        value = float(line_split[10] / (line_split[10] + 1e-10))
                    contact_matrix[int(idx1/res)][int(idx2/res)] = value
                elif line_split[0] == line_split[3] != chrom:
                    break
        diag_list = []
        for i in range(1000):
            diag_list.append(np.diagonal(contact_matrix, offset=i).tolist())
        with open(f"{save_loc}/HiC_{cell_type}_{chrom}_{dtype}_{res}_diagonal.txt", "wb") as fp:
            pickle.dump(diag_list, fp)

#########################
#     Script running    #
#########################

if __name__ == "__main__":
    args = parser.parse_args()
    config = vars(args)
    print(config)
    NORM_FILE = config['norm_hic_file']
    CHROM_LEN_FILE = config['chrom_len_file']
    SAVE_LOC = config['save_dir']
    NORM_TYPE = config['normalization_type']
    HIC_RESOLUTION = config['hic_resolution']
    CELL_TYPE = config['cell_type']
    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f"{CHROM_LEN_FILE}").readlines()}

    file_conversion(chrom_len = chrom_len, norm_file = NORM_FILE, dtype = NORM_TYPE,
                    cell_type = CELL_TYPE, res = HIC_RESOLUTION, save_loc = SAVE_LOC)

