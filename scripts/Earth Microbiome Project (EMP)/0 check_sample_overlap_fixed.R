library(rhdf5)

# 文件路径设置
biom.file <- "deblur/emp_deblur_100bp.qc_filtered.biom"
mapping.file <- "mapping_files/emp_qiime_mapping_qc_filtered.tsv"

# 读取 .biom 文件中的样本 ID
cat("Reading sample IDs from .biom ...\n")
biom.samples <- h5read(biom.file, name = "sample/ids")
writeLines(biom.samples, "biom_samples.txt")
cat("Total samples in .biom =", length(biom.samples), "\n")
cat("First 5 biom sample IDs:\n"); print(head(biom.samples, 5))

# 读取 mapping 文件
cat("Reading SampleIDs from mapping file ...\n")
mapping <- read.table(mapping.file,
                      header = TRUE,
                      sep = "\t",
                      quote = "",
                      comment.char = "",
                      stringsAsFactors = FALSE,
                      fill = TRUE)

# 清洗列名：#SampleID → SampleID
if ("#SampleID" %in% colnames(mapping)) {
  colnames(mapping)[colnames(mapping) == "#SampleID"] <- "SampleID"
}

# 提取 SampleID 和 empo_3 列
mapping.samples <- as.character(mapping$SampleID)
mapping.env <- mapping$empo_3

# 匹配样本 ID
matched <- intersect(biom.samples, mapping.samples)
cat("Matched sample IDs:", length(matched), "\n")
writeLines(matched, "matched_samples.txt")

# 筛选 mapping 中 matched 样本
matched.mapping <- mapping[mapping$SampleID %in% matched, ]

# 统计 matched 样本在各 empo_3 环境中的数量
cat("\n Matched sample distribution across empo_3 environments:\n")
matched.env.counts <- sort(table(matched.mapping$empo_3), decreasing = TRUE)
print(matched.env.counts)


write.table(matched.mapping,
            file = "matched_mapping.tsv",
            sep = "\t",
            quote = FALSE,
            row.names = FALSE)

write.table(matched.env.counts,
            file = "matched_env_counts.tsv",
            sep = "\t",
            quote = FALSE,
            col.names = FALSE)
