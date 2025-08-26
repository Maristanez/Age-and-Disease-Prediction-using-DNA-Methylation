library(minfi)
library(R.utils)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)

out_dir <- args[1]
extract_dir <- args[2]
raw_dir <- args[3]
gse_name <- args[4]

gse_dir <- paste0(extract_dir, "/", gse_name)
tar_file <- file.path(raw_dir, paste0(gse_name, "_RAW.tar"))

#Extract IDAT files in tar RAW tar file
if (!dir.exists(gse_dir)) {
  dir.create(gse_dir, recursive = TRUE)
}

idat_files <- list.files(gse_dir, pattern = "idat(\\.gz)?$", ignore.case = TRUE, recursive = TRUE)

if (length(idat_files) == 0) {
  #Extract 
  cat("Extracting:", tar_file, "...\n")
  untar(tar_file, exdir = gse_dir)
} else {
  cat("IDATs already extracted, skipping untar.\n")
}

#Unzip .gz files
gz_files <- list.files(gse_dir, pattern = "\\.idat\\.gz$", full.names = TRUE, recursive = TRUE)
if (length(gz_files) > 0) {
  cat("Decompressing", length(gz_files), "gzipped IDAT files...\n")
  for (f in gz_files) {
    gunzip(f, overwrite = TRUE, remove = TRUE)  # removes .gz and keeps .idat
  }
} else {
  cat("No gzipped IDATs found, skipping gunzip.\n")
}

# Read IDATs
cat("Reading IDAT files...\n")
rgSet <- read.metharray.exp(base = gse_dir, recursive = TRUE)
cat("rgSet loaded.\n")

# Normalize
cat("Normalizing...\n")
mSet <- preprocessNoob(rgSet)
cat("Normalization done.\n")

# Get beta values
cat("Extracting beta values...\n")
beta_values <- getBeta(mSet)

# Convert to data.frame and add CpG ID
beta_df <- as.data.frame(beta_values)
beta_df <- cbind(CpG_ID = rownames(beta_df), beta_df)

# Fix column names to contain only GSM ID
colnames(beta_df)[-1] <- sub("_.*", "", colnames(beta_df)[-1])  # Keeps only part before "_"

# Check
head(beta_df[, 1:5])

# Write to CSV efficiently
outfile <- file.path(out_dir, paste0(gse_name, "_methylation.csv"))
cat("Writing output to", outfile, "...\n")
fwrite(beta_df, outfile)

# Clean up memory
rm(rgSet, mSet, beta_values, beta_df)
gc()