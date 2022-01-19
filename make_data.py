from utils import *

# Make pickle file for labels
chroms =["chr" + str(i) for i in range(1,23)]
labels = {}
inputs = {}
diag_list_dir = './Akita_ICE_5kb'
# chipseq_path = './ChipSeq'

# hf = h5.File('data.h5', 'w')

for chr in chroms:

    if chr == 'chr16':
        continue

    print("Loading ", chr)
    # bw_list = load_chipseq(chipseq_path, chrom=chr, resolution=100)
    # diag_list_path = os.path.join(diag_list_dir, "Jun20_GM12878_" + chr + "_ICED_5kb_HiCExplorer_diagonal.txt")
    diag_list_path = os.path.join(diag_list_dir, "Oct5_GM12878_" + chr + "_ICED_5kb_Akita_rawICE_diagonal.txt")
    with open (diag_list_path, 'rb') as fp:
        diagonal_list = pickle.load(fp)

    # diag_log_list = [np.log2(np.array(i) + 1).tolist() for i in diagonal_list]
    diag_log_list = [(np.array(i)).tolist() for i in diagonal_list]
    print(len(diag_log_list[0]))

    # inputs[chr] = bw_list
    labels[chr] = diag_log_list

    # hf.create_dataset(chr, data=np.array(bw_list))

    # print(np.array(bw_list).shape)


# hf.close()
# hf = h5.File('data.h5', 'r')
# print(hf.keys())
with open('GM12878_y_5kb_Akita_ICED.pickle', 'wb') as handle:
	pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)		
    # self.sizes.append((len(diag_log_list[0]) - 2*self.buf)//self.seq_length + 1)

# Make h5 file for data