library(minfi)

idat_dir <- "./Dataset Preprocessing/Disease Dataset/IDAT Files"

# Read IDATs
rgSet <- read.metharray.exp(base = idat_dir, recursive = TRUE)
rgSet

# Normalize
mSet <- preprocessNoob(rgSet)

# Get beta values
beta_values <- getBeta(mSet)

# Convert to data.frame and add CpG ID
beta_df <- as.data.frame(beta_values)
beta_df <- cbind(CpG_ID = rownames(beta_df), beta_df)

# Fix column names to contain only GSM ID
colnames(beta_df)[-1] <- sub("_.*", "", colnames(beta_df)[-1])  # Keeps only part before "_"

# Check
head(beta_df[, 1:5])

# Write to CSV
write.csv(beta_df, "GSE153712_methylation.csv", row.names = FALSE)