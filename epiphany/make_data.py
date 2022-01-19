from utils import *

# Make pickle file for labels
chroms =["chr" + str(i) for i in range(1,23)]
labels = {}
inputs = {}
diag_list_dir = './Akita_ICE_5kb' # CHANGE THIS

for chr in chroms:

    print("Loading ", chr)
    diag_list_path = os.path.join(diag_list_dir, "Oct5_GM12878_" + chr + "_ICED_5kb_Akita_rawICE_diagonal.txt") #CHANGE THIS TO DESIRED FORMAT
    with open (diag_list_path, 'rb') as fp:
        diagonal_list = pickle.load(fp)

    diag_log_list = [(np.array(i)).tolist() for i in diagonal_list]
    labels[chr] = diag_log_list


with open('GM12878_y.pickle', 'wb') as handle:
	pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)		