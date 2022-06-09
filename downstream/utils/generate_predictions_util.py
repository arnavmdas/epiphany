'''
generate_predictions_util.py 

Util functions for generating predictions

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
# !pip install hickle
# import hickle as hkl
# !pip install pyBigWig
# import pyBigWig

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

def pred_chrom(chrom,
               chip_list,
               pred_location,
               window_size=14000,
               seq_length = 200,
               resolution_hic = 10000): 
    '''
    This function is used to generate submatrices of contact map 
    (100*200 sized matrix: 1Mb from the diagonal, 2Mb along the diagonal)
    chrom: which chromosome to predict
    chip_list: epigenomic track that loaded by data_load function
    pred_location: location and file name of where to save the predicted submatrices (they're stacked into a DataFrame)
    window_size: window size used in the model 
    seq_length: length of the submatrix along the diagonal 
                (seq_length=200 for 2Mb along the diagonal, 
                2Mb = seq_length * 100bp resolution for epigenomic tracks)
    resolution_hic: resolution of the Hi-C contact map (we choose 10kb as default)
    '''
    predlist = []
    n_chunk = round((len(chip_list[0])*100)/(resolution_hic*seq_length)) #how many submatrices in total to predict for each chromosome of selection
    torchtransform = transforms.Compose([transforms.ToTensor()])
    for k0 in range(n_chunk-2):
        k = k0 * seq_length * 100
        x0 = [i[k:k+seq_length*100+window_size] for i in chip_list] #find the correct input region to predict the submatrix
        x = torchtransform(np.array(x0))[0]
        net.eval()
        pred0 = net(x)[0]
        predlist.append(pred0[0].detach().numpy())
        del x0, x, pred0
    pred_df = pd.DataFrame(np.vstack(predlist))
    pred_df.to_csv(pred_location,index=False,sep="\t",header=False)

def pred_assemble(pred_location,
                  save_location,
                  window_size=14000,
                  resolution_hic=10000):
    '''
    This function is used to assemble all the predicted submatrices together 
    pred_location: the location and file name where the generated submatrices are saved at (the pred_location from pred_chrom function)
    save_location: the location and file name where assembled matrix to be saved at (please save into .txt format)
    window_size: the window size used in the model 
    resolution_hic: the resolution of Hi-C contact maps (default 10kb)
    '''
    chr_pred = pd.read_csv(pred_location,"\t",header=None)
    col1_list,col2_list = [],[]
    for i in range(chr_pred.shape[0]): #generate correct coordinates for each generated submatrix
        first_position = int((window_size / 2) / 100)
        col1 = [val for val in list(range(i+int(first_position-50)+1,i+first_position+2)) for _ in (0,1)][::-1][1:-1] # for Mar9
        col2 = [val for val in list(range(i+first_position+1,i+int(first_position+50+1))) for _ in (0,1)]
        col1_list.append(col1)
        col2_list.append(col2)
    col1_list = [j for i in col1_list for j in i]
    col2_list = [j for i in col2_list for j in i]
    col1_list = [int(i*resolution_hic) for i in col1_list]
    col2_list = [int(i*resolution_hic) for i in col2_list]
    chr_pred_flatten = [2**i-1 for i in np.array(chr_pred).flatten().tolist()] #need to exponentiate back (the prediction is under log2 scale)
    chr_coord = pd.DataFrame(np.array((col1_list,col2_list,chr_pred_flatten)).T)
    chr_coord.to_csv(save_loc,index=False,sep="\t",header=False) #save the assembled coordinate and counts 
    chr_coord.iloc[:,0] = [int(j) for j in chr_coord.iloc[:,0].to_list()]
    chr_coord.iloc[:,1] = [int(j) for j in chr_coord.iloc[:,1].to_list()]
    chr_coord.to_csv(str.replace(save_loc,".txt","_for_HiC.tsv.gz"),index=False,sep="\t",header=False) #prepare format to save into .hic format

def results_generation(chrom,
                       cell_type,
                       bwfile_dir,
                       submatrix_location,
                       assemble_matrix_location,
                       ground_truth_file,
                       ground_truth_location,
                       window_size=14000,
                       seq_length = 200,
                       resolution_hic = 10000):
    '''
    The overall function to generate predicted contact maps (1Mb distance band for the entire chromosome)
    chrom: which chromosome to generate
    cell_type: find the epigenomic bigWig files for the corresponding cell type
    bwfile_dir: the folder where bigWig files for each epigenomic tracks are stored
    submatrix_location: location for saving intermediate file (submatrices along the chromosome)
    assemble_matrix_location: location for saving intermediate file2 (assembled predicted submatrices along the chromosome)
    ground_truth_file: location of the ground truth contact matrices (saved as lists of lists in pickle format)
    ground_truth_location: location for saving subset ground truth with consistent coordinates with the predictions
    window_size: window size used in the model 
    seq_length: length of the submatrix along the diagonal
    resolution_hic: resolution of the Hi-C contact maps (default is 10kb)
    '''
    # 1. prepare chipseq data
    chip_list = data_load(chrom=chrom,bwfile_dir=bwfile_dir,cell_type=cell_type)

    # 2. generate predictions
    # a) generate submatrices (saved into stacked DataFrames)
    pred_chrom(chrom=chrom,chip_list=chip_list,pred_location = submatrix_location,
               window_size = window_size, seq_length = seq_length, resolution_hic = resolution_hic)
    # b) assemble generated submatrices into the an entire map for a chromosome
    pred_assemble(pred_location = submatrix_location, save_location = assemble_matrix_location, 
                  window_size = window_size, resolution_hic = resolution_hic)
    
    # 3. generate ground truth (subset ground truth matrix using the coordinates that we generated)
    # a) obtain coodinates for the ground truth maps
    with open(ground_truth_file, 'rb') as fp:
        diag_list = pickle.load(fp)
    diag_sublist = diag_list[:100] #subset 1Mb distance from the diagonal
    col1_list, col2_list = [], []
    for i in range(len(diag_sublist)):
        diag_vec = diag_sublist[i]
        col1 = [k*resolution_hic for k in range(len(diag_vec))]
        col2 = [(k+i)*resolution_hic for k in range(len(diag_vec))]
        col1_list.append(col1)
        col2_list.append(col2)
    diag_vec = [j for i in diag_sublist for j in i]
    col1_list = [j for i in col1_list for j in i]
    col2_list = [j for i in col2_list for j in i]
    diag_long = pd.DataFrame(np.array((col1_list,col2_list,diag_vec)).T)
    diag_long.columns = ["location1","location2","true_counts"]
    # b) load prediction matrix to get consistent coordinates 
    chr_coord = pd.read_csv(assemble_matrix_location,"\t",header=None)
    chr_coord.columns = ["location1","location2","prediction"]
    # c) subset data 
    diag_sub = pd.merge(chr_coord,diag_long,how="left",left_on=["location1","location2"],right_on=["location1","location2"])
    diag_sub = diag_sub[["location1","location2","true_counts"]]
    diag_sub.to_csv(ground_truth_location,index=False,sep="\t",header=False)
    diag_sub.iloc[:,0] = [int(i) for i in diag_sub.iloc[:,0].to_list()]
    diag_sub.iloc[:,1] = [int(i) for i in diag_sub.iloc[:,1].to_list()]
    diag_sub.to_csv(str.replace(ground_truth_location,".txt","_for_HiC.tsv.gz"),index=False,sep="\t",header=False)
    print("Complete", datetime.now())
    