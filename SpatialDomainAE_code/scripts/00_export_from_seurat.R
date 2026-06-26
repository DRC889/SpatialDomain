#!/usr/bin/env Rscript
# 00_export_from_seurat.R — Export Seurat spatial object to Python-friendly CSVs
#
# Input:  data/raw/spatial_seurat_1DP_13_nygen.rds (Mendeley)
# Output: data/processed/spot_metadata.csv
#         data/processed/coords_{sample}.csv
#         data/processed/expression_matrix.csv.gz (spots x genes, SCT normalized)
#         data/processed/gene_names.csv

library(SeuratObject)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
PROJ <- normalizePath(file.path(dirname(script_path), ".."))
OUT <- file.path(PROJ, "data", "processed")

cat("Loading Seurat object...\n")
obj <- readRDS(file.path(PROJ, "data", "raw", "spatial_seurat_1DP_13_nygen.rds"))
cat("  Dims:", dim(obj), "\n")

# --- 1. Export metadata ---
meta <- obj@meta.data
keep_cols <- c("barcode", "orig.ident", "nCount_Spatial", "nFeature_Spatial",
               "spatial_x", "spatial_y", "Sample", "Condition", "Timepoint",
               "seurat_clusters", "IntegratedAnno",
               "DetailedRegionAnnoShort", "BrainAreas",
               grep("^RCTD_", colnames(meta), value = TRUE))
meta_export <- meta[, keep_cols]
write.csv(meta_export, file.path(OUT, "spot_metadata.csv"), row.names = TRUE)
cat("  Saved spot_metadata.csv:", nrow(meta_export), "spots\n")

# --- 2. Export per-sample image coordinates + scale factors ---
for (name in names(obj@images)) {
    img <- obj@images[[name]]
    coords <- img@coordinates
    write.csv(coords, file.path(OUT, paste0("coords_", name, ".csv")))

    sf <- img@scale.factors
    sf_df <- data.frame(
        spot = sf$spot, fiducial = sf$fiducial,
        hires = sf$hires, lowres = sf$lowres
    )
    write.csv(sf_df, file.path(OUT, paste0("scalefactors_", name, ".csv")),
              row.names = FALSE)
    cat("  Saved coords +  scalefactors for", name, ":", nrow(coords), "spots\n")
}

# --- 3. Export expression matrix ---
# Access assay data directly via slots (avoid Seurat dependency)
assay_names <- names(obj@assays)
cat("  Available assays:", paste(assay_names, collapse=", "), "\n")

# Pick best assay: SCT > integrated > Spatial
if ("SCT" %in% assay_names) {
    assay_name <- "SCT"
} else if ("integrated" %in% assay_names) {
    assay_name <- "integrated"
} else {
    assay_name <- assay_names[1]
}
cat("  Using assay:", assay_name, "\n")

assay_obj <- obj@assays[[assay_name]]
cat("  Assay class:", class(assay_obj), "\n")

# Try to get data layer directly from slots
if ("data" %in% slotNames(assay_obj)) {
    expr <- assay_obj@data
} else if ("layers" %in% slotNames(assay_obj)) {
    # Seurat v5 uses layers
    layer_names <- names(assay_obj@layers)
    cat("  Layers:", paste(layer_names, collapse=", "), "\n")
    if ("data" %in% layer_names) {
        expr <- assay_obj@layers[["data"]]
    } else {
        expr <- assay_obj@layers[[1]]
    }
} else {
    stop("Cannot find expression data in assay object")
}

cat("  Expression matrix:", nrow(expr), "genes x", ncol(expr), "spots\n")

# Convert sparse to dense if needed, then transpose
if (inherits(expr, "dgCMatrix") || inherits(expr, "sparseMatrix")) {
    cat("  Converting sparse matrix...\n")
}

# Save gene names
gene_names <- rownames(expr)
if (is.null(gene_names)) gene_names <- rownames(assay_obj)
write.csv(data.frame(gene = gene_names), file.path(OUT, "gene_names.csv"),
          row.names = FALSE)

# Save as sparse-friendly format: save as Matrix Market
library(Matrix)
writeMM(expr, file.path(OUT, "expression_matrix.mtx"))
# Also save cell names
write.csv(data.frame(barcode = colnames(expr)), file.path(OUT, "spot_barcodes.csv"),
          row.names = FALSE)
cat("  Saved expression_matrix.mtx + gene_names.csv + spot_barcodes.csv\n")

cat("\n=== Export complete ===\n")
