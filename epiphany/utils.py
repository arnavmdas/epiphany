import os
import pyBigWig
import numpy as np
import pickle
import h5py as h5
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt

######## Data Preprocessing Utils ############
def load_chipseq(dir, chrom="chr1", resolution=100):

    """
    Args:
        dir: path to ChipSeq data
        chrom: chromosome to extract (chr1, chr2, ...)
        resolution: resolution in bp

    Returns:
       list of lists containing the 1D sequences of each bigWig file in specified directory
    """

    files = [i for i in os.listdir(dir) if "bigWig" in i]
    print(files)
    idx = [[i for i, s in enumerate(files) if chip in s][0] for chip in ['DNaseI','H3K27ac','H3K4me3','H3K27me3','CTCF']] #, 'SMC3']]
    files = [files[i] for i in idx]
    bw_list = []
    for file in files:

        bwfile = os.path.join(dir,file)
        print(bwfile)
        bw = pyBigWig.open(bwfile)

        value_list = []
        for i in list(range(0,bw.chroms()[chrom]-resolution,resolution)):
            value_list.append(bw.stats(chrom, i, i+resolution)[0])

        value_list = [0 if v is None else v for v in value_list]
        bw_list.append(value_list)

    return bw_list


### DIAGONAL PREDICTION STUFF
# 1. extract orthogonal stripe from the diagonal vector lists
def contact_extraction(m, diag_log_list, distance=100):
    '''
    Args:
        m: position along the diagonal
        diag_log_list: list of diagonal bands from HiC map
        distance: how long do we want the stripe to be, default is 100,
                which covers the interaction within 100*10kb = 1Mb genomic distance

    Returns:
        return
    '''

    a_element = []
    for k in range(distance):
        a1 = k // 2
        a2 = k % 2
        value = diag_log_list[k][m-a1-a2]
        a_element.append(value)
    return a_element

# def contact_extraction(m, diag_log_list, distance = 100):
#     # horizontal_element = [diag_log_list[k][m] for k in range(distance)]
#     vertical_element = [diag_log_list[k][m-k] for k in range(distance)]
#     return vertical_element


#2. Match ChIP-seq vector with stripe (at same location m)
def data_preparation(m, diag_log_list, chip_list, chip_res = 100, hic_res = 10000, distance = 200):
    '''
    m: position along the diagonal
    chip_list: the bw_list we generated above
    chip_res: the resolution of ChIP-seq data (100bp)
    hic_res: the resolution of Hi-C data (10kb)
    distance: distance from diagonal
    '''
    res_ratio = int(hic_res / chip_res)
    contacts = contact_extraction(m,diag_log_list,distance)
    return contacts #, chip_list[:, (m*res_ratio - int(distance/2)*res_ratio-1000):(m*res_ratio + int(distance/2)*res_ratio)+1000].T

######## PyTorch Utils ############
def save(net, file_name, num_to_keep=1):
    """Saves the net to file, creating folder paths if necessary.

    Args:
        net(torch.nn.module): The network to save
        file_name(str): the path to save the file.
        num_to_keep(int): Specifies how many previous saved states to keep once this one has been saved.
            Defaults to 1. Specifying < 0 will not remove any previous saves.
    """

    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), file_name)
    extension = os.path.splitext(file_name)[1]
    checkpoints = sorted(glob.glob(folder + '/*' + extension), key=os.path.getmtime)
    print('Saved %s\n' % file_name)
    if num_to_keep > 0:
        for ff in checkpoints[:-num_to_keep]:
            os.remove(ff)


def restore(net, save_file):
    """Restores the weights from a saved file

    This does more than the simple Pytorch restore. It checks that the names
    of variables match, and if they don't doesn't throw a fit. It is similar
    to how Caffe acts. This is especially useful if you decide to change your
    network architecture but don't want to retrain from scratch.

    Args:
        net(torch.nn.Module): The net to restore
        save_file(str): The file path
    """

    net_state_dict = net.state_dict()
    restore_state_dict = torch.load(save_file)

    restored_var_names = set()

    print('Restoring:')
    for var_name in restore_state_dict.keys():
        if var_name in net_state_dict:
            var_size = net_state_dict[var_name].size()
            restore_size = restore_state_dict[var_name].size()
            if var_size != restore_size:
                print('Shape mismatch for var', var_name, 'expected', var_size, 'got', restore_size)
            else:
                if isinstance(net_state_dict[var_name], torch.nn.Parameter):
                    net_state_dict[var_name] = restore_state_dict[var_name].data
                try:
                    net_state_dict[var_name].copy_(restore_state_dict[var_name])
                    print(str(var_name) + ' -> \t' + str(var_size) + ' = ' + str(int(np.prod(var_size) * 4 / 10**6)) + 'MB')
                    restored_var_names.add(var_name)
                except Exception as ex:
                    print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                              var_name, var_size, restore_size))
                    raise ex

    ignored_var_names = sorted(list(set(restore_state_dict.keys()) - restored_var_names))
    unset_var_names = sorted(list(set(net_state_dict.keys()) - restored_var_names))
    print('')
    if len(ignored_var_names) == 0:
        print('Restored all variables')
    else:
        print('Did not restore:\n\t' + '\n\t'.join(ignored_var_names))
    if len(unset_var_names) == 0:
        print('No new variables')
    else:
        print('Initialized but did not modify:\n\t' + '\n\t'.join(unset_var_names))

    print('Restored %s' % save_file)


def restore_latest(net, folder, ext='.pt'):
    """Restores the most recent weights in a folder

    Args:
        net(torch.nn.module): The net to restore
        folder(str): The folder path
    Returns:
        int: Attempts to parse the epoch from the state and returns it if possible. Otherwise returns 0.
    """

    checkpoints = sorted(glob.glob(folder + '/*' + ext), key=os.path.getmtime)
    start_it = 0
    if len(checkpoints) > 0:
        restore(net, checkpoints[-1])
        try:
            start_it = int(re.findall(r'\d+', checkpoints[-1])[-1])
        except:
            pass
    return start_it


def generate_image(label, pred, path='./', seq_length=1000, bands=200):
    
    path = os.path.join(path, 'ex.png')
    label = np.squeeze(label.numpy()).T[::-1,:]
    pred = np.squeeze(pred.numpy()).T[::-1,:]
    im1 = np.zeros((seq_length,seq_length))
    for j in range(bands):
        if j > 0:
            np.fill_diagonal(im1[:,j:], pred[bands-1-j,j//2:-j//2])
        else:
            np.fill_diagonal(im1, .5*pred[bands-1-j,:])

    im2 = np.zeros((seq_length,seq_length))
    for j in range(bands-1):
        if j > 0:
            np.fill_diagonal(im2[:,j:], label[bands-1-j,j//2:-j//2])
        else:
            np.fill_diagonal(im2, .5*label[bands-1-j,:])


    plt.imsave(path, im1 + im2.T, cmap='RdYlBu_r', vmin=0) #, vmin=0, vmax=6) #, vmin=-4, vmax=4)
    return plt.imread(path)





