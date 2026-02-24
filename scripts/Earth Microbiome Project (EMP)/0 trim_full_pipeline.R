library(readr)
library(data.table)

otu_dir <- "otutable/"
output_dir <- "trim_otu/"
summary_dir <- "otu_summary/"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)

otutb.name <- list.files(otu_dir, pattern = "\\.(csv|CSV)$")

n_sample <- 360

summary_overall <- data.frame(
  File = character(), 
  Original_Samples = numeric(),
  Filtered_Samples = numeric(),
  Sampled_Samples = numeric(),
  Retained_OTUs = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(otutb.name)) {
  file_path <- paste0(otu_dir, otutb.name[i])
  df <- read_csv(file_path, show_col_types = FALSE)
  df <- as.data.frame(df)
  
  # 设置行名并去除第一列
  rownames(df) <- df[[1]]
  df <- df[, -1]
  
  original_n <- nrow(df)
  cat("📋", otutb.name[i], "- original samples:", original_n, "\n")
  
  # 去除丰度为 0 的样本
  df <- df[rowSums(df) > 0, ]
  filtered_n <- nrow(df)
  cat("📋", otutb.name[i], "- after filtering:", filtered_n, "samples remain\n")
  
  if (filtered_n < 10) {
    cat("Skipping", otutb.name[i], "- too few valid samples\n")
    next
  }
  
  # 抽样（不超过 n_sample）
  set.seed(i)
  take_n <- min(filtered_n, n_sample)
  df.sample <- df[sample(1:filtered_n, take_n, replace = FALSE), ]
  
  # TSS 标准化
  norm_counts <- apply(df.sample, 1, function(x) x / sum(x))
  norm_counts <- t(norm_counts)
  norm_counts <- round(norm_counts, 6)
  
  otu_totals <- colSums(norm_counts)
  
  top_400_OTUs <- order(otu_totals, decreasing = TRUE)[1:400]
  
  norm_counts <- norm_counts[, top_400_OTUs]
  
  # 输出标准化 OTU 表
  output_path <- paste0(output_dir, otutb.name[i])
  write.csv(norm_counts, output_path, quote = FALSE)
  cat("Saved normalized OTU table:", otutb.name[i], "\n")
  
  otu_total <- colSums(df.sample)
  otu_present <- colSums(df.sample > 0)
  otu_summary <- data.frame(
    OTU_ID = names(otu_total),
    Total_Abundance = otu_total,
    Presence_Count = otu_present
  )
  write.csv(otu_summary, paste0(summary_dir, "summary_", otutb.name[i]), row.names = FALSE)
  cat("Wrote summary table for:", otutb.name[i], "\n")
  
  summary_overall <- rbind(summary_overall, data.frame(
    File = otutb.name[i],
    Original_Samples = original_n,
    Filtered_Samples = filtered_n,
    Sampled_Samples = take_n,
    Retained_OTUs = ncol(df.sample)
  ))
}

write.csv(summary_overall, paste0(summary_dir, "summary_overview.csv"), row.names = FALSE)
cat("Wrote overall summary table: summary_overview.csv\n")


otu_files <- list.files(output_dir, pattern = "\\.csv$", full.names = TRUE)

result <- data.frame(
  File = character(),
  Num_Samples = numeric(),
  Num_OTUs = numeric(),
  stringsAsFactors = FALSE
)

for (file in otu_files) {
  df <- read.csv(file, row.names = 1, check.names = FALSE)
  
  result <- rbind(result, data.frame(
    File = basename(file),
    Num_Samples = nrow(df),
    Num_OTUs = ncol(df)
  ))
}

print(result)
write.csv(result, "otu_dimensions_summary.csv", row.names = FALSE)







