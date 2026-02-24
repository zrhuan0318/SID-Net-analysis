library(rhdf5)
library(SparseM)
library(hdf5r)

# 设置路径
biom.file <- "deblur/emp_deblur_100bp.qc_filtered.biom"
mapping.file <- "mapping_files/emp_qiime_mapping_qc_filtered.tsv"

# 读取 biom 文件内容
counts <- h5read(biom.file, name = "observation/matrix")
sample <- h5read(biom.file, name = "sample/ids")
otu.id <- h5read(biom.file, name = "observation/ids")
# taxon <- h5read(biom.file, name = "observation/metadata/taxonomy")
# write.csv(taxon, "taxon.csv")

# 自动构建 OTU 编号
otu.number <- counts$indices
for (i in 1:(length(counts$indptr) - 1)) {
  if (counts$indptr[i + 1] > counts$indptr[i]) {
    otu.number[counts$indptr[i]:(counts$indptr[i + 1] - 1)] <- i
  }
}

# 正确读取 #SampleID
mapping.raw <- readLines(mapping.file)
mapping.header <- sub("^#", "", mapping.raw[1])
mapping <- read.table(text = paste(mapping.header, mapping.raw[-1], sep = "\n"),
                      header = TRUE,
                      sep = "\t",
                      quote = "",
                      fill = TRUE,
                      comment.char = "",
                      stringsAsFactors = FALSE)

# 子集提取函数
otutable_subset <- function(subset = "Plant surface", samples = 65) {
  cat("🔹 Subset:", subset, " | Sample cutoff:", samples, "\n")
  subset.sample <- mapping$SampleID[mapping$empo_3 == subset]
  subset.sample.position <- pmatch(subset.sample, sample)
  subset.sample.position <- subset.sample.position[!is.na(subset.sample.position)]

  if (length(subset.sample.position) < 10) {
    cat("Skipping subset '", subset, "' due to too few matched samples (", length(subset.sample.position), ")\n")
    return(NULL)
  }

  sel <- counts$indices %in% subset.sample.position
  data.subset <- counts$data[sel]
  indices.subset <- counts$indices[sel]
  otu.subset <- otu.number[sel]

  if (length(otu.subset) < 2) {
    cat("OTU subset too short: ", length(otu.subset), "\n")
    return(NULL)
  }

  indptr.subset <- c()
  for (i in 1:(length(otu.subset) - 1)) {
    if (!is.na(otu.subset[i]) && !is.na(otu.subset[i + 1]) && otu.subset[i] != otu.subset[i + 1]) {
      indptr.subset <- c(indptr.subset, i)
    }
  }
  indptr.subset <- c(indptr.subset, length(otu.subset))

  A <- matrix(rnorm(50), 10, 5)
  A[abs(A) < 0.5] <- 0
  A.csr <- as.matrix.csr(t(A) %*% A)
  slot(A.csr, "ra") <- as.numeric(data.subset)
  slot(A.csr, "ja") <- as.integer(factor(indices.subset))
  slot(A.csr, "ia") <- as.integer(c(0, indptr.subset))
  slot(A.csr, "dimension") <- as.integer(c(length(indptr.subset), length(subset.sample.position)))

  subset.matrix <- as.matrix(A.csr)
  colnames(subset.matrix) <- paste0("D", 1:ncol(subset.matrix))

  if (nrow(subset.matrix) == length(unique(otu.subset))) {
    rownames(subset.matrix) <- paste0("otu", unique(otu.subset))
  } else {
    rownames(subset.matrix) <- paste0("otu", seq_len(nrow(subset.matrix)))
    warning("Row count does not match unique OTU count; fallback names used.")
  }

  subset.matrix.rare <- subset.matrix[rowSums(subset.matrix) > sum(subset.matrix) / 1e5, , drop = FALSE]
  subset.matrix.present <- subset.matrix.rare
  subset.matrix.present[subset.matrix.present != 0] <- 1
  subset.matrix.rare.present <- t(subset.matrix.rare[rowSums(subset.matrix.present) > samples, , drop = FALSE])

  return(subset.matrix.rare.present)
}

# 创建输出目录
if (!dir.exists("otutable")) dir.create("otutable")

# 获取 empo_3 分组信息
group <- sort(table(mapping$empo_3), decreasing = TRUE)
group.name <- names(group)
group.name <- group.name[!is.na(group.name) & group.name != "empo_3"]

cat("实际参与分析的 empo_3 分组类型：\n")
print(group.name)

# 遍历分组动态设定稀释阈值并输出
for (g in group.name) {
  sample_cutoff <- if (group[g] > 2000) 200 else round(group[g] / 10)
  otutable <- otutable_subset(subset = g, samples = sample_cutoff)
  if (!is.null(otutable)) {
    write.csv(otutable, paste0("otutable/", gsub(" ", "_", g), ".csv"))
  }
}
