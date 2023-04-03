# Data Preparation
Data preparation pipeline for Epiphany model.

Scripts in this folder create input and target datasets for training Epiphany model: 

- `bam2bw.sh`: creates .bw file for each epigenomic track.
- `hicdcplus_norm_sig.R`: normalizes Hi-C matrices using [HiC-DC+](https://github.com/mervesa/HiCDCPlus), and calculate significant interactions.
- `convert_hic_norm.py`: converts normalized Hi-C counts into list
- `prepare_data.py`: prepares input and target datasets for the training dataloader

Quick link: [input preparation](#step-1-download-and-prepare-epigenomic-tracks-for-model-input) | [target preparation](#step-2-prepare-normalized-hi-c-contact-maps-as-training-targets) | [pair input and target data](#step-3-create-paired-input-and-target-data-for-epiphany-training)

If you have any questions, please contact <ruy4001@med.cornell.edu>

-----

## Step 1. Download and prepare epigenomic tracks for model input
1. For each epigenomic signal that you want to include as one of the input tracks: use `bam2bw.sh` script to prepare the .bw file. 
For example, for H3K36me3 track of GM12878, run 
```
bash /epiphany/data_preparation/bam2bw.sh \
https://www.encodeproject.org/files/ENCFF353YPB/@@download/ENCFF353YPB.bam \ #replicate 1
https://www.encodeproject.org/files/ENCFF677MAG/@@download/ENCFF677MAG.bam \ #replicate 2
/train_epiphany/epi_downloading_folder \ #folder to store the downloaded files (.bam format)
/train_epiphany/epi_storing_folder \ #folder to store intermediate and final processed file (.bw format)
GM12878_H3K36me3 # name of the track
```
2. Input and output
- Input: this script takes in the downloading path of `.bam` file of epigenomic data (usually with two replicates) 
- Output: `.bw` file of the corresponding track will be saved in `epi_storing_folder`

-----

## Step 2. Prepare normalized Hi-C contact maps as training targets

### **I. Normalize Hi-C contact maps with [HiC-DC+](https://github.com/mervesa/HiCDCPlus) in R**

[HiC-DC+](https://github.com/mervesa/HiCDCPlus) is an R package for analyzing Hi-C and HiChIP data. Here we use it to prepare normalized Hi-C contact data, and simultaneously calculate significant interactions in the Hi-C matrix.  

1. We start from HiC-Pro outputs (`_abs.bed` file and `.matrix` file). If you have `.hic` file, please refer to the [Finding Significant Interactions from Hi-C/HiChIP]([HiC-DC+](https://github.com/mervesa/HiCDCPlus)) section.

```
#1. Start R
/usr/bin/R

# Please adjust the path arguments in the R script before running.

#2. Run R script 
source("/epiphany/data_preparation/hicdcplus_norm_sig.R")
```

2. Output format: taking 10kb resolution as an example
```
   chrI  startI   endI  chrJ  startJ   endJ      D  counts    pvalue  qvalue         mu       sdev
0  chr1   10000  20000  chr1   10000  20000      0       1  0.997758       1  24.521386  15.985778
1  chr1   10000  20000  chr1   20000  30000  10000       0  1.000000       1  31.173523  20.113244
2  chr1   10000  20000  chr1   30000  40000  20000       0  1.000000       1  26.842012  17.425886
3  chr1   10000  20000  chr1   40000  50000  30000       0  1.000000       1  26.939051  17.486099
4  chr1   10000  20000  chr1   50000  60000  40000       0  1.000000       1  20.320882  13.378158
```
In the results table, 
- `counts`: raw counts
- `pvalue`: -log10 p-value for significance test
- `qvalue`: -log10 adjusted p-value
- `mu`: negative binomial z-score normalized counts
- `sdev`: negative binomial standard deviation

We can either use `mu` column as expected read counts (z-score) as normalized counts, or to create a new column of `obs-over-exp` (=`counts`/`mu`) as the observed-over-expected count ratio for the prediction task.

### **II. Convert normalized Hi-C contact matrices into list**

In this step, we extract normalized Hi-C counts of `distance = 1000 bins * resolution` from the diagonal, and stored them in pickle list.
- `norm_hic_file`: path and name of `_result.txt.gz` file output from HiC-DC+
- `chrom_len_dir`: we need to download chromosome size file from UCSC. hg19 can be found [here](https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/), and hg38 can be found [here](https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/).
- `save_dir`: where to save the pickle list
- `normalization_type`: as mentioned above, we can use either `z-value` (`mu`) as the normalized target to train the model, or `obs-over-exp` (`counts`/`mu`) as the target.
- `hic_resolution`: resolution of the Hi-C data being normalized.

```
#1. We need to download chromosome size file from UCSC (e.g. downloading hg19 chrom.size): 

wget https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes -O /train_epiphany/hg19.chrom.sizes

#2. Run data conversion script

python /epiphany/data_preparation/convert_hic_norm.py \
--norm_hic_file /hicdcplus/intermediate/Project_A_Sample_1_10000_result.txt.gz \ 
--chrom_len_dir /train_epiphany/hg19.chrom.sizes \
--save_dir /train_epiphany/hic_storing_folder \
--normalization_type "obs-over-exp" \
--cell_type "GM12878" \
--hic_resolution 10000
```

-----

## Step 3. Create paired input and target data for Epiphany training 
In previous steps, we have created 
- input `bw` tracks in `epi_storing_folder`
- normalized Hi-C count list in `hic_storing_folder`

Now run the `prepare_data.py` script to prepare input and target datasets. Some important arguments are
- `epi_input_dir`: folder where `bw` tracks are stored
- `hic_input_dir`: folder where normalized Hi-C count list is stored
- `target_dir`: where to save the prepared results
- `epi_order`: select some or all epigenomic `bw` tracks, and organize them in certain order. E.g. we want to use DNaseI, H3K27ac, H3K27me3, H3K4me3, and CTCF to predict the Hi-C contact map, then we put in `DNaseI H3K27ac H3K27me3 H3K4me3 CTCF`

```
python /epiphany/data_preparation/prepare_data.py \
--epi_input_dir /train_epiphany/epi_storing_folder \ 
--hic_input_dir /train_epiphany/hic_storing_folder \
--target_dir /train_epiphany/GM12878_processed \
--cell_type "GM12878" \
--epi_order DNaseI H3K27ac H3K27me3 H3K4me3 CTCF \
--file_name "DNaseI-H3K27ac-H3K27me3-H3K4me3-CTCF" \
--hic_resolution 10000 \
--epi_resolution 100
```

Output: 

Under the `target_dir`, we will have 
- `GM12878_DNaseI-H3K27ac-H3K27me3-H3K4me3-CTCF_X.pickle` for input tracks, and
- `GM12878_DNaseI-H3K27ac-H3K27me3-H3K4me3-CTCF_y.pickle` for target.