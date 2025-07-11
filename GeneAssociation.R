library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
library(minfi)
library(readr)
library(dplyr)

# Load annotation
ann450k <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)
ann850k <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)

# Read CpG site lists
diseaseCpGSites <- read_csv("top20s/disease_top_20_cpg_sites.csv")$`Site Name`
ageCpGSites <- read_csv("top20s/age_top_20_cpg_sites.csv")$`Site Name`

# Subset annotations
age_genes <- ann450k[rownames(ann450k) %in% ageCpGSites, c("Name", "UCSC_RefGene_Name")]
disease_genes <- ann850k[rownames(ann850k) %in% diseaseCpGSites, c("Name", "UCSC_RefGene_Name")]

# Convert to data frame if needed
age_genes_df <- as.data.frame(age_genes)
disease_genes_df <- as.data.frame(disease_genes)

View(age_genes_df)
View(disease_genes_df)

# Save to CSV
write_csv(age_genes_df, "Results/age_cpg_gene_map.csv")
write_csv(disease_genes_df, "Results/disease_cpg_gene_map.csv")
