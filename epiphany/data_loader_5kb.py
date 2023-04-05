from utils import *

import pandas as pd
import numpy as np
import torch.utils.data 
import torch
import pickle
import h5py as h5
import os
from model import *
from torch.autograd import Variable
import h5py 
import time
#wandb.init()

class Chip2HiCDataset(torch.utils.data.Dataset):
    def __init__(self, chipseq_path=None, diag_list_dir=None, seq_length=200, window_size=14000, chroms=['chr22'], mode='train', save_dir='./Datasets', subtract_mean=False, obs_exp=False):
                
        save_path_X = os.path.join(save_dir, 'H1_X.h5')
        save_path_y = os.path.join(save_dir, 'H1_y_5kb_Akita_ICED_microC.pickle')

        self.seq_length = seq_length
        self.chroms = chroms
        self.buf = 200
        self.window_size = window_size

        self.inputs = {}
        self.labels = {}
        self.sizes = []
        self.subtract_mean = subtract_mean
        self.obs_exp = obs_exp

        print("Loading input:")
        self.inputs = h5.File(save_path_X, 'r')
        print("Loading labels:")
        with open(save_path_y, 'rb') as handle:
            self.labels = pickle.load(handle)          

        for chr in self.chroms:
            diag_log_list = self.labels[chr]
            print(len(diag_log_list[0]))
            self.sizes.append((len(diag_log_list[0]) - 2*self.buf)//self.seq_length + 1)

        print(self.sizes)

        if self.subtract_mean or self.obs_exp:
            self.mean = torch.load(os.path.join(save_dir, 'H1_5kb_Akita_ICE_microC_mean.pt')).numpy()

        return

    def __len__(self):
        
        return int(np.sum(self.sizes))


    def __getitem__(self, index):
        
        arr = np.array(np.cumsum(self.sizes).tolist())
        arr[arr <= index] = 100000
        chrom_idx = np.argmin(arr)
        chr = self.chroms[chrom_idx]
        idx = int(index - ([0] + np.cumsum(self.sizes).tolist())[chrom_idx])
        start = idx*self.seq_length + self.buf
        end = np.minimum(idx*self.seq_length + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf)
        contact_data = []
        for t in range(idx*self.seq_length + self.buf, np.minimum(idx*self.seq_length + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf),1):
            contact_vec  = data_preparation(t,self.labels[chr],self.inputs[chr])
            contact_data.append(contact_vec)

        y_chr = np.array(contact_data)            
        np.nan_to_num(y_chr, copy=False, nan=0.0)
        X_chr = self.inputs[chr][:5, 50*start-(self.window_size//2):50*end+(self.window_size//2)].astype('float32')

        if self.subtract_mean and y_chr.shape[0] > 0:
            y_chr = y_chr - self.mean

        if self.obs_exp and y_chr.shape[0] > 0:
            y_chr = np.log(1 + (y_chr/self.mean))

        if y_chr.shape[0] < self.seq_length:
  
            try:
                pad_y = np.zeros((self.seq_length - y_chr.shape[0], y_chr.shape[1]))
                y_chr = np.concatenate((y_chr, pad_y), axis=0)
            except:
                y_chr = np.zeros((self.seq_length,200))

            pad_X = np.zeros((X_chr.shape[0],self.seq_length*50+self.window_size - X_chr.shape[1]))
            X_chr = np.concatenate((X_chr, pad_X), axis=1)  

        # y_chr = y_chr * 10
        return X_chr.astype('float32'), y_chr.astype('float32')





# test_chroms = ['chr19', 'chr20', 'chr22']
# loader = Chip2HiCDataset(chroms=test_chroms)
# train_loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=0)

# for x,y in train_loader:
#     print(x.shape)
#     print(y.shape)

"""
import time as time
import random 

# Initialize BERT
#net = DNABERT().cuda()
#head = FineTune().cuda()
#net.train(), head.train()
net = DNAEmbedding().cuda()

#print(net.state_dict()['dnabert.encoder.layer.11.output.dense.weight'])

"""
"""
# Create dataset
test_chroms = ["chr" + str(i) for i in range(1, 19)]
x = Chip2HiCDataset(chipseq_path='./ChipSeq_Data', diag_list_dir='./GM12878', seq_length=200, chroms=test_chroms, mode='train',prebuilt=True)
train_loader = torch.utils.data.DataLoader(x, batch_size=1, shuffle=False, num_workers=0)


#params = net.parameters() #list(net.parameters()) + list(head.parameters())
#optimizer = optim.Adam(params, lr=1e-3, weight_decay=0.0005)

# Generate one (Chip, DNA, Label) tuple
for i, (d,d1,l) in enumerate(train_loader):
    chip = d[0] 
    dna = d1[0]
    
    print(chip.shape)
    print(dna.shape)


t0 = time.time()
step = 20

"""
"""
out = net(dna.cuda())
out = torch.squeeze(out)
out = torch.cat((chip.cuda(), out), dim=0)
print(out.shape)

for ep in range(50):

    # Generate two random batches to be backpropagated through
    b1 = random.randint(0, 32000//30)
    b2 = random.randint(0, 32000//30)
    print(b1, b2)

    # Get embedding for sequence sequentially
    #dna_embedded = torch.Tensor(np.zeros((1,768,0)), requires_grad=True).cuda()  #requires_grad=True).cuda()
    dna_embedded = torch.empty(size = (1,768,0), requires_grad=True).cuda()
    for j in range(0,dna.shape[0],step):
        inp = dna[j:j+step].cuda()
        out = net(inp)
        if j == step*b1 or j == step*b2:
            dna_embedded = torch.cat((dna_embedded,out.detach()), dim=2)
        else:
            dna_embedded = torch.cat((dna_embedded,out.detach()), dim=2)
        
        #print(out.shape, dna_embedded.shape)
    

    print(net.state_dict()['dnabert.encoder.layer.11.output.dense.weight'])
    dna_embedded = head(dna_embedded)
    print(dna_embedded.shape)
    optimizer.zero_grad()
    loss = F.mse_loss(dna_embedded, chip.cuda())
    loss.backward()
    optimizer.step()
    wandb.log({'epoch': ep, 'loss': loss.item()})
    print("Loss: ", loss.item())

t1 = time.time()

print(t1 - t0)
#print(chip.shape, dna_embedded.shape, t1-t0)
    
"""




