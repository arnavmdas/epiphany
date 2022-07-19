'''
generate_perturbations_util.py 

Util functions for generating perturbations at certain genomic location

'''

##########################
#    Loading packages    #
##########################

# 1. Load packages

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import randn
from torch.nn import MSELoss
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import time
import pickle
from datetime import datetime
from torch.autograd import Variable
import gzip
import sys
import os 
from sklearn.decomposition import TruncatedSVD, PCA
torch.set_default_tensor_type(torch.DoubleTensor)
# !pip install pyBigWig
import pyBigWig
# !pip install hickle
# import hickle as hkl

# 2. Load data - part 2

# !wget https://s3.amazonaws.com/hicfiles.tc4ga.com/public/juicer/juicer_tools_1.22.01.jar
# !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.refGene.gtf.gz
# !gunzip /content/hg38.refGene.gtf.gz

chrom_list = ["chr"+str(i) for i in range(1,23)] #for human hg38
length_list = [248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,
               138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,
               83257441,80373285,58617616,64444167,46709983,50818468]
chrom_len_dict = dict(zip(chrom_list,length_list))

########################
#    Util functions    #
########################

#1. load epigenomic tracks for each chromosome
#(extract data from bigWig files to numerical vector)

def data_load(chrom,
              bwfile_dir,
              distances=100,
              cell_type="GM12878"):
    '''
    This function is used to generate epigenomic input tracks for the prediction 
    chrom: which chromosome to load 
    bwfile_dir: the folder where bigWig files for each epigenomic tracks are stored
    distances = 100: resolution of epigenomic data (epigenomic tracks are binned into 100 bp resolution)
    cell_type: find corresponding bigWig files for the specific cell type
    '''
    #a) get all epigenomic file names for a certain cell type, and put them in the correct order
    files = [j for j in [i for i in os.listdir(bwfile_dir) if "bigWig" in i] if cell_type in j]
    idx = [[i for i, s in enumerate(files) if chip in s][0] for chip in ['DNaseI','H3K27ac','H3K4me3','H3K27me3','CTCF']]
    files2 = [files[i] for i in idx]
    #b) extract data from these bigWig files
    bw_list = []
    for bwfile in files2:
        bw = pyBigWig.open(bwfile_dir + "/" + bwfile) # open bigWig files
        distance_gaps = list(range(0,bw.chroms()[chrom]-distances,distances)) # cut the chromosome into 100 bp bins
        value_list = [bw.stats(chrom,i,i+distances)[0] for i in distance_gaps] # extract values
        value_list = [0.0 if v is None else v for v in value_list] # replace NA with 0.0
        bw_list.append(value_list)
    del bw, value_list
    return bw_list # this is a list of list: [[values for DNaseI], [values for H3K27ac], ...]

def prediction(chrom,cell_type,input_tracks,
               bin0,bin1,
               perturb_type=None,
               track_idxs = [4],
               save_txt="/content/temp.txt.gz", remap = True):
    '''
	This function is used to generate perturbed/unperturbed region using pre-trained model. 
	chrom: which chromosome is of interest
	cell_type: cell type
	input_tracks: pre-loaded input tracks (output of `data_load` function)
	bin0, bin1: start and end location of the perturbation 
	perturb_type: [None,"mask", "deletion"]
	              - None: no perturbation is added onto the input signals. will generate original prediction for the region. 
				  - mask: signals in the perturbation area [bin0,bin1] in one or more epigenomic tracks got masked to the background
				  - deletion: signals in the perturbation area got deleted (deletion happens in all tracks)
	track_idxs: if perturb_type is "mask", signal on which track is masked 
				- 0 for DnaseI, 1 for H3K27ac, 2 for H3K4me3, 3 for H3K27me3, 4 for CTCF
				- can select one or more tracks, e.g. [4] or [0,4]
	save_txt: location for saving intermediate file 
	remap: whether to adjust the coordinates of deleted region. Only matters when perturb_type == "deletion".
	'''
    a = input_tracks
    window_size = 14000
    #a) Prepare corresponding input region
    start = int((bin0/100+bin1/100)/2 - (window_size+20000)/2)
    x0 = [i[start:start+(window_size+20000)] for i in a]
    peak_width = abs(bin1-bin0)
    if perturb_type == "mask":
        for track_idx in track_idxs:
            perturb_a = int((bin0 + bin1)/2) - int(peak_width/2)
            perturb_b = int((bin0 + bin1)/2) + int(peak_width/2)
            x0[track_idx][int(perturb_a/100-start):int(perturb_b/100-start)] = [np.mean(x0[track_idx]) for i in x0[0][int(perturb_a/100-start):int(perturb_b/100-start)]]
    elif perturb_type == "deletion":
        mid_point = int((bin0/100+bin1/100)/2)
        mid_before = mid_point - int(peak_width / 100)
        mid_after = mid_point + int(peak_width / 100)
        start = mid_before - int((window_size+20000)/2)
        end = mid_after + int((window_size+20000)/2)
        x_first_half = [i[start:mid_before] for i in a]
        x_second_half = [i[mid_after:end] for i in a]
        x0 = [i+j for i,j in zip(x_first_half,x_second_half)]
    torchtransform = transforms.Compose([transforms.ToTensor()])
    x = torchtransform(np.array(x0))[0]
    x0a = x

    #b) Make prediction 
    pred0 = net(x,seq_length = 400)
    
    #c) Prepare coordinates
    idx_start = int((bin0+bin1)/2-1000000)
    idx_end = int((bin0+bin1)/2+1000000)
    res = 5000
    col1_list, col2_list = [], []
    for i in range(int(idx_start/res),int(idx_end/res)): 
        first_position = 0
        col1 = [val for val in list(range(i+(first_position-100)+1,i+first_position+2)) for _ in (0,1)][::-1][1:-1] # for Mar9
        col2 = [val for val in list(range(i+first_position+1,i+first_position+100+1)) for _ in (0,1)] # only goes to 40100000.0	which should go to ~8000...
        col1_list.append(col1)
        col2_list.append(col2)
    col1_list = [j for i in col1_list for j in i]
    col2_list = [j for i in col2_list for j in i]
    col1_list = [int(i*res) for i in col1_list]
    col2_list = [int(i*res) for i in col2_list]

    # prepare data format to save into hic 
    value_list = pred0[0].detach().numpy()[0].flatten().tolist()
    col0 = [0] * len(col1_list)
    col_chr = [chrom] * len(col1_list)
    col1 = [1] * len(col1_list)
    if perturb_type == "deletion":
        if remap:
            mid_point = int((bin0+bin1)/2)
            distance_adjust = int(peak_width)
            col1_list = [i-distance_adjust if i < mid_point else i+distance_adjust for i in col1_list]
            col2_list = [i-distance_adjust if i < mid_point else i+distance_adjust for i in col2_list]
    # save into dataframe
    chr_for_HiC = pd.DataFrame(np.array((col0,col_chr,col1_list,col0,col0,col_chr,col2_list,col1,value_list)).T)
    chr_for_HiC.to_csv(save_txt,index=False,sep="\t",header=False)
    del pred0, value_list, col0, col_chr,col1_list,col2_list, chr_for_HiC
