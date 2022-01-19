from utils import *

import pandas as pd
import numpy as np
import torch.utils.data
import torch
import pickle
import h5py as h5
import os
from model import *


def make_chip():
    chroms = ["chr" + str(i) for i in range(1,23)][::-1]

    for cell in ['GM12878']:
        chipseq_path = os.path.join('./GM12878/')
        inputs = {}
        for chr in chroms:

            print("Loading ", chr)
            bw_list = load_chipseq(chipseq_path, chrom=chr, resolution=100)

            inputs[chr] = np.array(bw_list)
            print(np.array(bw_list).shape)


        save_path_X = cell + '_X.pickle'
        with open(save_path_X, 'wb') as handle:
            pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_labels():
    # Make pickle file for labels
    chroms =["chr" + str(i) for i in range(1,23)]
    labels = {}
    inputs = {}
    diag_list_dir = './GM12878' # CHANGE THIS

    for chr in chroms:

        print("Loading ", chr)
        diag_list_path = os.path.join(diag_list_dir, "Oct5_GM12878_" + chr + "_ICED_5kb_Akita_rawICE_diagonal.txt") #CHANGE THIS TO DESIRED FORMAT
        with open (diag_list_path, 'rb') as fp:
            diagonal_list = pickle.load(fp)

        diag_log_list = [(np.array(i)).tolist() for i in diagonal_list]
        labels[chr] = diag_log_list


    with open('GM12878_y.pickle', 'wb') as handle:
    	pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)		

def main():
    make_chip()
    make_labels()

 
if __name__ == '__main__':
    main()