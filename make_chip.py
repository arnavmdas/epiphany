from utils import *

import pandas as pd
import numpy as np
import torch.utils.data
import torch
import pickle
import h5py as h5
import os
from model import *


chroms = ["chr" + str(i) for i in range(1,23)][::-1]
print(chroms)

for cell in ['K562']:
    #mm10/CH12/Epigenetic/
    chipseq_path = os.path.join('/x0/arnav/Chip2HiC/K562/')
    inputs = {}
    for chr in chroms:

        print("Loading ", chr)
        bw_list = load_chipseq(chipseq_path, chrom=chr, resolution=100)

        inputs[chr] = np.array(bw_list)
        print(np.array(bw_list).shape)


    save_path_X = cell + '_X.pickle'
    with open(save_path_X, 'wb') as handle:
        pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)