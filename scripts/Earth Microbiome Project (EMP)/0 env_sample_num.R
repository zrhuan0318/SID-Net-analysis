# 读取 EMP mapping 文件
mapping <- read.table("mapping_files/emp_qiime_mapping_qc_filtered.tsv",
                      sep = "\t",
                      header = TRUE,
                      quote = "",
                      fill = TRUE,
                      comment.char = "",
                      stringsAsFactors = FALSE)

# 确保列名正确（去除 # 前缀）
if ("#SampleID" %in% colnames(mapping)) {
  colnames(mapping)[colnames(mapping) == "#SampleID"] <- "SampleID"
}

# 查看 empo_3 列中所有唯一的生态环境类型
unique_empo3 <- unique(mapping$empo_3)

# 输出类型数量和名称
cat("一共识别出", length(unique_empo3), "种 empo_3 类型：\n\n")
print(unique_empo3)

# 可选：列出每个类型下的样本数量
cat("\n 每类样本数量如下：\n")
print(sort(table(mapping$empo_3), decreasing = TRUE))
