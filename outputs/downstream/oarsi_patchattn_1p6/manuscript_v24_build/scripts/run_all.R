#!/usr/bin/env Rscript
# Regenerate all v2.4 figures, tables and the v2.4 docx (run on the server in the r_pub env).
d <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
for (s in c("01_fig1_method_overview.R", "02_fig2_cohort_preprocessing.R", "03_fig3_cohort_characteristics.R",
            "04_fig4_attention.R", "05_fig5_laterality_flip.R", "06_fig6_oarsi.R", "07_fig7_kl.R",
            "10_tables.R", "20_build_docx.R", "30_build_docx_en.R", "31_build_docx_v27.R", "32_build_docx_v28.R", "33_build_docx_v29.R")) {
  message("== ", s)
  st <- system2(file.path(R.home("bin"), "Rscript"), c("--vanilla", file.path(d, s)))
  if (st != 0) stop("failed: ", s)
}
