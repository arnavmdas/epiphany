#hicdc+ for normalization
#
# Usage
# screen
# bsub -n 2 -W 100:00 -R 'span[hosts=1] rusage[mem=64]' -Is /bin/bash
# /home/bin/R
# source("/epiphany/data_preparation/hicdcplus_norm_sig.R") # nolint

library(HiCDCPlus)

######################################
#      Step 0. Define arguments      #
######################################

resolution <- 10000 # resolution of the Hi-C contact map
outdir <- "/hicdcplus/features/" # path to save the constructed features
outpth <- "/hicdcplus/intermediate/" # path to save the results
chromosomes <- paste("chr", seq(1, 22, 1), sep = "") # which chromosomes to normalize # nolint
merged_location <- "/alignments/Hi-C/merged/hic-pro/hic_results/matrix/" # path of aligned hic-pro files # nolint
sample_names <- c("Project_A_Sample_1", "Project_A_Sample_2") # Hi-C file names
hg_assembly <- "hg19"
species <- "Hsapiens"

########################################
#      Step 1. Construct features      #
########################################

#Step 1. construct features (resolution, chrom specific)
construct_features(output_path = paste0(outdir, hg_assembly, "_", as.integer(resolution/1000),"kb_GATC_GANTC", sep = ""), # nolint
gen = species, gen_ver = hg_assembly, sig = c("GATC", "GANTC"),
bin_type = "Bins-uniform", binsize = resolution, chrs = chromosomes)

########################################################
#      Step 2. Calculate significant interactions      #
########################################################

#Step 2. calculate significant interaction

set.seed(1010)
indexfile <- data.frame()
for (sample_name in sample_names){
    gi_list <- generate_bintolen_gi_list(bintolen_path =
    paste0(outdir, "/", hg_assembly, "_", as.integer(resolution / 1000),
    "kb_GATC_GANTC_bintolen.txt.gz", sep = ""))
    abs_file <- paste0(merged_location, sample_name, "/raw/", resolution,
    "/", sample_name, "_", resolution, "_abs.bed", sep = "")
    matrix_file <- paste0(merged_location, sample_name, "/raw/", resolution,
    "/", sample_name, "_", resolution, ".matrix", sep = "")
    gi_list <- add_hicpro_matrix_counts(gi_list, absfile_path = abs_file,
    matrixfile_path = matrix_file, chrs = chromosomes)
    gi_list <- expand_1D_features(gi_list)
    set.seed(1010) #HiC-DC downsamples rows for modeling
    # gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2) # nolint
    gi_list <- HiCDCPlus(gi_list, ssize = 0.1)
    for (i in seq_along(gi_list)){
        indexfile <- unique(rbind(indexfile,
        as.data.frame(gi_list[[i]][gi_list[[i]]$qvalue <= 0.05])[c("seqnames1", "start1", "start2")])) # nolint
    }
    gi_list_write(gi_list, fname = paste0(outpth, sample_name, "_",
    resolution, "_result.txt.gz", sep = ""),
    score = "normcounts") #"_", chrom #columns = "minimal_plus_score",
}
