#!/usr/bin/env Rscript
# Tables 1-4 (v2.4) built with flextable. Every number is computed from the data files and checked against the
# v2.2 manuscript tables (string comparison of the formatted value). Outputs: outputs/tables/tables_v24.rds
# (flextable objects + notes for the docx builder), PNG previews and a CSV of all cells.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))
source(file.path(SCRIPT_DIR, "captions_en.R"))
suppressPackageStartupMessages({ library(flextable); library(officer) })

cs <- read_json_data("cohort_summary.json"); oh <- read_json_data("oarsi_heads_and_laterality_three_seeds.json")
pv <- read_json_data("pvalues_v1.json"); rr <- read_json_data("reader_reliability.json")$results$all_visits_pooled
t5 <- read_json_data("t5_results.json"); x1 <- read_json_data("x1_factorial_three_seeds.json")
x3 <- read_json_data("x3_p4_three_seeds.json"); pr <- read_json_data("primary_p3_vs_p0.json")

M <- "−"
f <- function(x, d) minus(formatC(x, format = "f", digits = d))
fs <- function(x, d) minus(sprintf(paste0("%+.", d, "f"), x))
pct <- function(n, N, d = 1) sprintf("%.*f%%", d, 100 * n / N)
ci <- function(lo, hi, d, sign = TRUE) if (sign) sprintf("%s to %s", fs(lo, d), fs(hi, d)) else sprintf("%s–%s", f(lo, d), f(hi, d))
expect <- function(got, want, what) if (!identical(got, want)) stop(sprintf("%s: got '%s', v2.2 has '%s'", what, got, want))
sdv <- function(x) sd(x)

set_flextable_defaults(font.family = "Times New Roman", font.size = 8, padding.top = 1.6, padding.bottom = 1.6,
                       padding.left = 3, padding.right = 3, line_spacing = 1)
B_THICK <- fp_border(color = "#000000", width = 1.0); B_THIN <- fp_border(color = "#000000", width = 0.5)
B_HAIR <- fp_border(color = "#BFBFBF", width = 0.5)
booktabs <- function(ft, widths_cm) {
  ft <- border_remove(ft)
  ft <- hline_top(ft, border = B_THICK, part = "header")
  ft <- hline_bottom(ft, border = B_THIN, part = "header")
  ft <- hline_bottom(ft, border = B_THICK, part = "body")
  ft <- bold(ft, part = "header")
  ft <- valign(ft, valign = "center", part = "all")
  ft <- align(ft, j = seq_len(ncol_keys(ft))[-1], align = "center", part = "all")
  ft <- align(ft, j = 1, align = "left", part = "all")
  ft <- width(ft, width = widths_cm / 2.54)
  set_table_properties(ft, layout = "fixed", align = "center")
}
SPL <- c("train", "val", "test")
# Display style for confidence intervals (user, 2026-11-24): "(lower, upper)", no plus sign inside the interval.
# Applied after the cell-by-cell comparison with v2.2, which used "lower to upper".
ci_comma <- function(s) {
  s <- gsub("\\(([+\u2212]?[0-9.]+) to ([+\u2212]?[0-9.]+)\\)", "(\\1, \\2)", s, perl = TRUE)
  s <- gsub("(+", "(", s, fixed = TRUE); s <- gsub(", +", ", ", s, fixed = TRUE)
  s <- gsub("\\(([0-9.]+)\u2013([0-9.]+)\\)", "(\\1, \\2)", s)
  gsub("95% CI ([0-9.]+)\u2013([0-9.]+)", "95% CI \\1, \\2", s)
}

# ===================================================================================== Table 1
S <- lapply(SPL, function(k) cs$splits[[k]]); names(S) <- SPL; TT <- cs$total
row <- function(label, vals) data.frame(label = label, train = vals[1], val = vals[2], test = vals[3], total = vals[4])
sec <- function(label) data.frame(label = label, train = "", val = "", test = "", total = "")
kpp <- function(s) sprintf("%d [%d–%d] (%d–%d)", s$knees_per_patient$median, s$knees_per_patient$q25,
                           s$knees_per_patient$q75, s$knees_per_patient$min, s$knees_per_patient$max)
jsl0 <- sapply(S, function(s) s$oarsi_dist$jsl[["0"]]); jsl_n <- sapply(S, function(s) s$knees - s$oarsi_dist$jsl$missing)
t1 <- rbind(
  sec("Cohort size"),
  row("Assigned OAI participants, n", c(sapply(S, function(s) fmt_int(s$assigned_patients)), fmt_int(TT$assigned_patients))),
  row("Included participants, n (%)", c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$patients), pct(s$patients, s$assigned_patients))),
                                        sprintf("%s (%s)", fmt_int(TT$patients), pct(TT$patients, TT$assigned_patients)))),
  row("Bilateral PA films, n", c(sapply(S, function(s) fmt_int(s$films)), fmt_int(TT$films))),
  row("Knee-visits, n", c(sapply(S, function(s) fmt_int(s$knees)), fmt_int(TT$knees))),
  # total median [IQR] (range) is not stored in cohort_summary.json; value verified on 2026-11-23 (v1.8 review) from the knee-level manifest
  row("Knee-visits per participant, median [IQR] (range)", c(sapply(S, kpp), "10 [6–12] (1–14)")),
  sec("Baseline demographics"),
  row("Age, years, mean ± SD", c(sapply(S, function(s) sprintf("%.1f ± %.1f", s$demographics$age$mean, s$demographics$age$sd)),
                                     sprintf("%.1f ± %.1f", TT$age_mean, TT$age_sd))),
  row("Female, n (%)", c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$demographics$female), pct(s$demographics$female, s$patients))),
                         sprintf("%s (%s)", fmt_int(TT$female), pct(TT$female, TT$patients)))),
  row("BMI, kg/m², mean ± SD*", c(sapply(S, function(s) sprintf("%.1f ± %.1f", s$demographics$bmi$mean, s$demographics$bmi$sd)),
                                          sprintf("%.1f ± %.1f", TT$bmi_mean, TT$bmi_sd))),
  sec("Laterality and imaging modality"),
  row("Right / left knees, n (% right)", c(sapply(S, function(s) sprintf("%s / %s (%s)", fmt_int(s$by_side$R), fmt_int(s$by_side$L), pct(s$by_side$R, s$knees))),
                                           sprintf("%s / %s (%s)", fmt_int(TT$by_side$R), fmt_int(TT$by_side$L), pct(TT$by_side$R, TT$knees)))),
  row("CR / DX / RG, n", c(sapply(S, function(s) sprintf("%s / %s / %s", fmt_int(s$by_modality$CR), fmt_int(s$by_modality$DX), fmt_int(s$by_modality$RG))),
                           sprintf("%s / %s / %s", fmt_int(TT$by_modality$CR), fmt_int(TT$by_modality$DX), fmt_int(TT$by_modality$RG)))),
  sec("KL grade, n (%)"),
  do.call(rbind, lapply(0:4, function(g) {
    lab <- c("KL 0 (none)", "KL 1 (doubtful)", "KL 2 (mild OA)", "KL 3 (moderate OA)", "KL 4 (severe OA)")[g + 1]
    row(lab, c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$by_kl[[as.character(g)]]), pct(s$by_kl[[as.character(g)]], s$knees))),
               sprintf("%s (%s)", fmt_int(TT$by_kl[[as.character(g)]]), pct(TT$by_kl[[as.character(g)]], TT$knees))))
  })),
  sec("OARSI labels"),
  row("Both JSN grades available (jsm and jsl), n (%)",
      c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$both_narrowing), ifelse(s$both_narrowing < s$knees & 100 * s$both_narrowing / s$knees > 99.9, ">99.9%", pct(s$both_narrowing, s$knees)))),
        sprintf("%s (>99.9%%)", fmt_int(TT$both_narrowing)))),
  row("Lateral JSN grade 0, n (% of graded)", c(sprintf("%s (%s)", fmt_int(jsl0), pct(jsl0, jsl_n)),
                                                 sprintf("%s (%s)", fmt_int(sum(jsl0)), pct(sum(jsl0), sum(jsl_n))))),
  row("All four osteophyte grades available, n (%)", c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$all_four_osteophytes), pct(s$all_four_osteophytes, s$knees))),
                                                       sprintf("%s (%s)", fmt_int(TT$all_four_osteophytes), pct(TT$all_four_osteophytes, TT$knees)))),
  row("Fractional follow-up JSN truncated, n (jsm / jsl)", c(sapply(S, function(s) sprintf("%s / %s", fmt_int(s$jsm_fraction), fmt_int(s$jsl_fraction))),
                                                              sprintf("%s / %s", fmt_int(TT$jsm_fraction), fmt_int(TT$jsl_fraction)))),
  sec("Image QC"),
  row("Peripheral ruler zero-masked, n (%)", c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$cleanup_status$accepted_cleaned), pct(s$cleanup_status$accepted_cleaned, s$knees))),
                                               sprintf("%s (%s)", fmt_int(TT$ruler_cleaned), pct(TT$ruler_cleaned, TT$knees)))),
  row("QC-excluded knees with valid KL, n (% of KL-linked knees)",
      c(sapply(S, function(s) sprintf("%s (%s)", fmt_int(s$held_knees), pct(s$held_knees, s$knees + s$held_knees))),
        sprintf("%s (%s)", fmt_int(TT$held_knees), pct(TT$held_knees, TT$knees + TT$held_knees))))
)
rownames(t1) <- NULL
# check against the v2.2 Table 1 (values copied from the v2.2 docx)
v22_t1 <- list(
  c("3,836", "480", "480", "4,796"), c("3,537 (92.2%)", "435 (90.6%)", "448 (93.3%)", "4,420 (92.2%)"),
  c("16,845", "2,075", "1,886", "20,806"), c("32,256", "3,962", "3,633", "39,851"),
  c("10 [7–12] (1–14)", "10 [6–12] (1–14)", "8 [6–10] (1–14)", "10 [6–12] (1–14)"),
  c("61.2 ± 9.1", "61.0 ± 9.4", "60.3 ± 9.1", "61.1 ± 9.2"), c("2,062 (58.3%)", "239 (54.9%)", "266 (59.4%)", "2,567 (58.1%)"),
  c("28.5 ± 4.8", "28.4 ± 4.8", "28.8 ± 5.0", "28.6 ± 4.8"),
  c("16,285 / 15,971 (50.5%)", "1,999 / 1,963 (50.5%)", "1,851 / 1,782 (50.9%)", "20,135 / 19,716 (50.5%)"),
  c("21,927 / 4,827 / 5,502", "2,620 / 538 / 804", "2,397 / 575 / 661", "26,944 / 5,940 / 6,967"),
  c("12,837 (39.8%)", "1,772 (44.7%)", "1,586 (43.7%)", "16,195 (40.6%)"), c("6,173 (19.1%)", "681 (17.2%)", "602 (16.6%)", "7,456 (18.7%)"),
  c("7,931 (24.6%)", "865 (21.8%)", "822 (22.6%)", "9,618 (24.1%)"), c("4,176 (12.9%)", "494 (12.5%)", "487 (13.4%)", "5,157 (12.9%)"),
  c("1,139 (3.5%)", "150 (3.8%)", "136 (3.7%)", "1,425 (3.6%)"),
  c("32,255 (>99.9%)", "3,962 (100.0%)", "3,633 (100.0%)", "39,850 (>99.9%)"), c("30,108 (93.3%)", "3,711 (93.7%)", "3,391 (93.3%)", "37,210 (93.4%)"),
  c("17,277 (53.6%)", "1,983 (50.1%)", "1,931 (53.2%)", "21,191 (53.2%)"), c("948 / 273", "103 / 37", "108 / 28", "1,159 / 338"),
  c("5,855 (18.2%)", "753 (19.0%)", "692 (19.0%)", "7,300 (18.3%)"), c("2,002 (5.8%)", "300 (7.0%)", "209 (5.4%)", "2,511 (5.9%)"))
dat_rows <- which(t1$train != "")
stopifnot(length(dat_rows) == length(v22_t1))
for (i in seq_along(dat_rows)) for (j in 1:4) expect(t1[dat_rows[i], j + 1], v22_t1[[i]][j], sprintf("Table 1 row '%s' col %d", t1$label[dat_rows[i]], j))

sec_rows <- which(t1$train == "")
ft1 <- flextable(t1) |>
  set_header_labels(label = "Characteristic", train = "Training", val = "Validation", test = "Held-out test", total = "Total")
for (r in sec_rows) ft1 <- merge_at(ft1, i = r, j = 1:5)
ft1 <- booktabs(ft1, c(4.6, 3.0, 2.7, 2.7, 3.0))
ft1 <- bold(ft1, i = sec_rows, part = "body") |> italic(i = sec_rows, part = "body") |>
  hline(i = sec_rows[-1] - 1, border = B_HAIR, part = "body") |>
  padding(i = setdiff(seq_len(nrow(t1)), sec_rows), j = 1, padding.left = 9, part = "body") |>
  align(i = sec_rows, j = 1, align = "left", part = "body")

# ===================================================================================== Table 2
qa <- function(q, a) sprintf("%s\n(%.1f%%)", f(q, 3), 100 * a)
hs <- SITES
val_mean <- function(arm, h, m) num(oh$oarsi_heads[[arm]]$three_seed_summary[[h]][[m]]$mean)
test_p3 <- function(h, m) mean(sapply(SEEDS, function(s) num(t5$per_model[[sprintf("primary_selected_seed%s_P3", s)]]$oarsi[[h]][[m]])))
test17 <- function(arm, h) num(t5$per_model[[sprintf("%s_seed17_%s", if (arm == "P3") "primary_selected" else "secondary_selected", arm)]]$oarsi[[h]]$qwk)
n_val <- sapply(hs, function(h) num(oh$oarsi_heads$P3$three_seed_summary[[h]]$n))
n_test <- sapply(hs, function(h) num(t5$per_model$primary_selected_seed17_P3$oarsi[[h]]$n))
t2 <- data.frame(
  site = sprintf("%s (%s)", SITE_LAB[hs], hs),
  n = sprintf("%s /\n%s", fmt_int(n_val), fmt_int(n_test)),
  p2 = sapply(hs, function(h) qa(val_mean("P2", h, "qwk"), val_mean("P2", h, "accuracy"))),
  p3 = sapply(hs, function(h) qa(val_mean("P3", h, "qwk"), val_mean("P3", h, "accuracy"))),
  p4 = sapply(hs, function(h) qa(val_mean("P4", h, "qwk"), val_mean("P4", h, "accuracy"))),
  d = sapply(hs, function(h) { e <- pv$B_oarsi_validation_P3_minus_P2_three_seeds[[h]]$P3_minus_P2
    sprintf("%s\n(%s)\n%s", fs(num(e$mean_diff), 3), ci(num(e$ci95[[1]]), num(e$ci95[[2]]), 3), fmt_p(num(e$p_two_sided))) }),
  t3 = sapply(hs, function(h) qa(test_p3(h, "qwk"), test_p3(h, "accuracy"))),
  t24 = sapply(hs, function(h) sprintf("%s /\n%s", f(test17("P2", h), 3), f(test17("P4", h), 3))),
  rt = sapply(hs, function(h) { r <- rr[[toupper(h)]]; sprintf("%s\n(%s)", f(num(r$kappa_quadratic), 3), ci(num(r$ci95_kappa_quadratic[[1]]), num(r$ci95_kappa_quadratic[[2]]), 3, FALSE)) }),
  stringsAsFactors = FALSE)
v22_t2 <- list(  # QWK (accuracy) strings from v2.2 Table 2, columns P2, P3, P4, test P3, test P2/P4, re-test
  osfm = c("0.759 (72.3%)", "0.819 (76.1%)", "0.831 (76.3%)", "0.799 (72.4%)", "0.772 / 0.801", "0.849 (0.768–0.906)", "+0.060 (0.029 to 0.095; P < 0.001)"),
  osfl = c("0.623 (69.0%)", "0.703 (72.3%)", "0.707 (72.9%)", "0.711 (71.5%)", "0.611 / 0.679", "0.794 (0.700–0.860)", "+0.080 (0.037 to 0.128; P < 0.001)"),
  ostm = c("0.696 (65.7%)", "0.784 (72.1%)", "0.787 (71.7%)", "0.771 (69.3%)", "0.731 / 0.765", "0.827 (0.774–0.868)", "+0.088 (0.066 to 0.111; P < 0.001)"),
  ostl = c("0.647 (70.9%)", "0.787 (77.6%)", "0.780 (77.2%)", "0.805 (78.3%)", "0.607 / 0.803", "0.817 (0.735–0.870)", "+0.140 (0.076 to 0.213; P < 0.001)"),
  jsm = c("0.858 (82.7%)", "0.868 (83.3%)", "0.862 (83.0%)", "0.874 (84.1%)", "0.868 / 0.872", "0.867 (0.820–0.904)", "+0.010 (0.001 to 0.020; P = 0.018)"),
  jsl = c("0.875 (96.5%)", "0.826 (96.2%)", "0.845 (96.1%)", "0.857 (96.6%)", "0.902 / 0.892", "0.897 (0.806–0.954)", "−0.049 (−0.080 to −0.023; P < 0.001)"))
flat <- function(x) gsub("\n", " ", x)
for (i in seq_along(hs)) {
  w <- v22_t2[[hs[i]]]
  expect(flat(t2$p2[i]), w[1], paste("T2 P2", hs[i])); expect(flat(t2$p3[i]), w[2], paste("T2 P3", hs[i]))
  expect(flat(t2$p4[i]), w[3], paste("T2 P4", hs[i])); expect(flat(t2$t3[i]), w[4], paste("T2 test P3", hs[i]))
  expect(flat(t2$t24[i]), w[5], paste("T2 test P2/P4", hs[i])); expect(flat(t2$rt[i]), w[6], paste("T2 retest", hs[i]))
  e <- pv$B_oarsi_validation_P3_minus_P2_three_seeds[[hs[i]]]$P3_minus_P2   # v2.2 printed the CI without signs; same numbers
  v22d <- sprintf("%s (%s to %s; %s)", fs(num(e$mean_diff), 3), f(num(e$ci95[[1]]), 3), f(num(e$ci95[[2]]), 3), fmt_p(num(e$p_two_sided)))
  expect(v22d, w[7], paste("T2 diff", hs[i]))
}
stopifnot(identical(unname(n_val), c(1987, 1983, 1983, 1983, 3962, 3962)), all(n_test[1:4] == 1931), all(n_test[5:6] == 3633))
seed17_p3_test <- sapply(hs, function(h) test17("P3", h))
note2 <- sub("%s", paste0(paste(f(seed17_p3_test[1:3], 3), collapse = ", "), " and ", f(seed17_p3_test[4], 3)), TAB_CAP[[2]]$note, fixed = TRUE)
note2 <- sub("%s", paste(f(seed17_p3_test[5:6], 3), collapse = " and "), note2, fixed = TRUE)
stopifnot(grepl("QWK 0.821, 0.737, 0.777 and 0.814 for osfm, osfl, ostm and ostl, and 0.878 and 0.917 for jsm and jsl", note2, fixed = TRUE))

t2[] <- lapply(t2, ci_comma)
ft2 <- flextable(t2) |>
  set_header_labels(site = "OARSI site", n = "Knees\n(val / test)", p2 = "P2", p3 = "P3", p4 = "P4",
                    d = "P3 − P2\n(95% CI); P", t3 = "P3\n(3 seeds)", t24 = "P2 / P4\n(seed 17)", rt = "QWK\n(95% CI)") |>
  add_header_row(values = c("", "", "Validation, QWK (accuracy)", "Held-out test", "OAI re-test"), colwidths = c(1, 1, 4, 2, 1))
ft2 <- booktabs(ft2, c(2.75, 1.35, 1.45, 1.45, 1.45, 2.3, 1.45, 1.55, 1.85)) |>
  hline(i = 1, j = 3:6, border = B_THIN, part = "header") |> hline(i = 1, j = 7:8, border = B_THIN, part = "header") |>
  hline(i = 1, j = 9, border = B_THIN, part = "header") |> fontsize(size = 7.5, part = "all") |>
  hline(i = 4, border = B_HAIR, part = "body")

# ===================================================================================== Table 3
ARMS5 <- c("P0", "P1", "P2", "P3", "P4")
per_seed <- function(arm, m) sapply(SEEDS, function(s) if (arm == "P4") num(x3$per_seed[[s]]$P4[[m]]) else num(x1[[m]]$per_seed[[s]]$values[[arm]]))
msd <- function(v, d) sprintf("%s (%s)", f(mean(v), d), f(sdv(v), d))
eff_cell <- function(e, d = 4, p = NULL) {
  s <- sprintf("%s\n(%s)", fs(num(e$estimate), d), ci(num(e$ci95[[1]]), num(e$ci95[[2]]), d))
  if (!is.null(p)) s <- paste0(s, sprintf("\nP = %.3f", p))
  s
}
va <- pv$A_kl_validation_P3_minus_P0$selected
p3p0 <- list(q = list(estimate = va$mean_diff, ci95 = va$ci95),
             auc = list(estimate = pr$mean_diff$auc_sel, ci95 = pr$ci95_mean_diff$auc_sel),
             x2 = list(estimate = pr$mean_diff$x2_sel, ci95 = pr$ci95_mean_diff$x2_sel))
t3 <- data.frame(
  row = c("P0 (baseline)", "P1", "P2", "P3 (full model)", "P4 (laterality control)", "P3 − P0 (primary)", "Pooling main effect",
          "OARSI supervision main effect", "Interaction", "P4 − P3 (laterality control)"),
  pool = c("–", "✓", "–", "✓", "✓", "", "", "", "", ""),
  oarsi = c("–", "–", "on CLS token", "on site queries", "on site queries,\nno label swap", "", "", "", "", ""),
  q = c(sapply(ARMS5, function(a) msd(per_seed(a, "q_sel"), 4)), eff_cell(p3p0$q, 4, num(va$p_two_sided)),
        eff_cell(x1$q_sel$three_seed_mean$pooling_main), eff_cell(x1$q_sel$three_seed_mean$auxiliary_main),
        eff_cell(x1$q_sel$three_seed_mean$interaction), eff_cell(x3$P4_minus_P3$q_sel$three_seed_mean)),
  auc = c(sapply(ARMS5, function(a) msd(per_seed(a, "auc_sel"), 4)), eff_cell(p3p0$auc),
          eff_cell(x1$auc_sel$three_seed_mean$pooling_main), eff_cell(x1$auc_sel$three_seed_mean$auxiliary_main),
          eff_cell(x1$auc_sel$three_seed_mean$interaction), eff_cell(x3$P4_minus_P3$auc_sel$three_seed_mean)),
  x2 = c(sapply(ARMS5, function(a) msd(per_seed(a, "x2_sel"), 4)), eff_cell(p3p0$x2),
         eff_cell(x1$x2_sel$three_seed_mean$pooling_main), eff_cell(x1$x2_sel$three_seed_mean$auxiliary_main),
         eff_cell(x1$x2_sel$three_seed_mean$interaction), eff_cell(x3$P4_minus_P3$x2_sel$three_seed_mean)),
  stringsAsFactors = FALSE)
v22_t3 <- list(  # v2.2 Table 3 strings (KL QWK, AUC, cross-two-grade error)
  c("0.8539 (0.0014)", "0.9434 (0.0014)", "0.0408 (0.0059)"), c("0.8570 (0.0020)", "0.9452 (0.0008)", "0.0443 (0.0011)"),
  c("0.8579 (0.0013)", "0.9441 (0.0010)", "0.0383 (0.0067)"), c("0.8590 (0.0033)", "0.9462 (0.0001)", "0.0422 (0.0024)"),
  c("0.8592 (0.0028)", "0.9457 (0.0006)", "0.0347 (0.0086)"),
  c("+0.0050 (−0.0010 to +0.0114; P = 0.104)", "+0.0028 (−0.0010 to +0.0068)", "+0.0014 (−0.0030 to +0.0059)"),
  c("+0.0021 (−0.0037 to +0.0082)", "+0.0019 (−0.0016 to +0.0055)", "+0.0037 (−0.0005 to +0.0080)"),
  c("+0.0029 (+0.0013 to +0.0046)", "+0.0009 (+0.0002 to +0.0016)", "−0.0023 (−0.0036 to −0.0011)"),
  c("−0.0020 (−0.0058 to +0.0016)", "+0.0004 (−0.0011 to +0.0019)", "+0.0005 (−0.0022 to +0.0033)"),
  c("+0.0002 (−0.0039 to +0.0044)", "−0.0005 (−0.0025 to +0.0015)", "−0.0076 (−0.0107 to −0.0046)"))
to_v22 <- function(s) { s <- sub("\\)\nP = ([0-9.]+)$", "; P = \\1)", s); s <- gsub("\n", " ", s); s <- sub(" to ", " to ", s); s }
for (i in 1:10) for (j in 1:3) expect(to_v22(t3[i, c("q", "auc", "x2")[j]]), v22_t3[[i]][j], sprintf("Table 3 row %d col %d", i, j))

t3[] <- lapply(t3, ci_comma)
ft3 <- flextable(t3) |>
  set_header_labels(row = "Model / effect", pool = "Attention\npooling", oarsi = "OARSI\nsupervision", q = "KL QWK",
                    auc = "AUC (KL ≥ 2)", x2 = "Error of ≥ 2\ngrades (rate)")
for (r in 6:10) ft3 <- merge_at(ft3, i = r, j = 1:3, part = "body")
ft3 <- booktabs(ft3, c(3.7, 1.45, 2.15, 2.9, 2.9, 2.9)) |> hline(i = 5, border = B_THIN, part = "body") |>
  align(i = 6:10, j = 1, align = "left", part = "body")

# ===================================================================================== Table 4
pm <- function(s, a) t5$per_model[[sprintf("primary_selected_seed%s_%s", s, a)]]
sm <- function(a) t5$per_model[[paste0("secondary_selected_seed17_", a)]]
mets <- c("qwk", "accuracy", "mae", "auc_kl_ge_2", "cross_two_error_rate")
dig <- c(qwk = 4, accuracy = 1, mae = 3, auc_kl_ge_2 = 4, cross_two_error_rate = 4)
cell_msd <- function(a, m) { v <- sapply(SEEDS, function(s) num(pm(s, a)[[m]]))
  if (m == "accuracy") sprintf("%.1f%% (%.1f%%)", 100 * mean(v), 100 * sdv(v)) else msd(v, dig[[m]]) }
sec4 <- t5$secondary_P3_minus_P0_selected
cell_diff <- function(m) { e <- sec4[[m]]
  if (m == "accuracy") return(sprintf("%s pp\n(%s to %s)", fs(100 * num(e$mean), 1), fs(100 * num(e$ci95[[1]]), 1), fs(100 * num(e$ci95[[2]]), 1)))
  d <- dig[[m]]; lo <- num(e$ci95[[1]])
  lo_txt <- if (m == "qwk") fs(lo, 5) else fs(lo, d)
  s <- sprintf("%s\n(%s to %s)", fs(num(e$mean), d), lo_txt, fs(num(e$ci95[[2]]), d))
  if (m == "qwk") s <- paste0(s, sprintf("\nP = %.3f", num(pv$A_kl_test_P3_minus_P0$primary_selected$p_two_sided)))
  s }
cell_s17 <- function(m) paste(sapply(c("P1", "P2", "P4"), function(a) {
  v <- num(sm(a)[[m]]); if (m == "accuracy") sprintf("%.1f%%", 100 * v) else f(v, dig[[m]]) }), collapse = " /\n")
rk <- rr$KL
t4 <- data.frame(
  row = c("Mean of 3 seeds (SD)", "Mean of 3 seeds (SD)", "P3 − P0 (95% CI)", "Controls, seed 17", "Indirect reference (Project 15)"),
  model = c("P0", "P3", "Primary\ncomparison", "P1 / P2 / P4", "OAI reader\nre-test"),
  qwk = c(cell_msd("P0", "qwk"), cell_msd("P3", "qwk"), cell_diff("qwk"), cell_s17("qwk"),
          sprintf("%s\n(95%% CI %s)", f(num(rk$kappa_quadratic), 3), ci(num(rk$ci95_kappa_quadratic[[1]]), num(rk$ci95_kappa_quadratic[[2]]), 3, FALSE))),
  acc = c(cell_msd("P0", "accuracy"), cell_msd("P3", "accuracy"), cell_diff("accuracy"), cell_s17("accuracy"), sprintf("%.1f%%", 100 * num(rk$exact_agreement))),
  mae = c(cell_msd("P0", "mae"), cell_msd("P3", "mae"), cell_diff("mae"), cell_s17("mae"), "—"),
  auc = c(cell_msd("P0", "auc_kl_ge_2"), cell_msd("P3", "auc_kl_ge_2"), cell_diff("auc_kl_ge_2"), cell_s17("auc_kl_ge_2"), "—"),
  x2 = c(cell_msd("P0", "cross_two_error_rate"), cell_msd("P3", "cross_two_error_rate"), cell_diff("cross_two_error_rate"), cell_s17("cross_two_error_rate"), "—"),
  stringsAsFactors = FALSE)
v22_t4 <- list(
  c("0.8633 (0.0033)", "70.0% (1.5%)", "0.336 (0.011)", "0.9536 (0.0012)", "0.0331 (0.0048)"),
  c("0.8691 (0.0033)", "71.7% (0.4%)", "0.321 (0.003)", "0.9552 (0.0009)", "0.0379 (0.0043)"),
  c("+0.0058 (−0.00002 to +0.0118; P = 0.051)", "+1.8 percentage points (+0.7 to +2.9)", "−0.014 (−0.026 to −0.003)", "+0.0016 (−0.0011 to +0.0044)", "+0.0048 (+0.0008 to +0.0087)"),
  c("0.8744 / 0.8655 / 0.8674", "72.9% / 69.7% / 68.9%", "0.310 / 0.336 / 0.337", "0.9553 / 0.9534 / 0.9544", "0.0383 / 0.0317 / 0.0256"),
  c("0.864 (95% CI: 0.822–0.899)", "75.1%", "—", "—", "—"))
to_v22_t4 <- function(s, i, j) {
  s <- gsub(" /\n", " / ", s); s <- sub("\\)\nP = ([0-9.]+)$", "; P = \\1)", s); s <- gsub("\n", " ", s); s <- sub(" to ", " to ", s)
  s <- sub(" pp ", " percentage points ", s); s <- sub("\\(95% CI ", "(95% CI: ", s); s }
for (i in 1:5) for (j in 1:5) expect(to_v22_t4(t4[i, c("qwk", "acc", "mae", "auc", "x2")[j]], i, j), v22_t4[[i]][j], sprintf("Table 4 row %d col %d", i, j))

t4[] <- lapply(t4, ci_comma)
ft4 <- flextable(t4) |>
  set_header_labels(row = "", model = "Model", qwk = "KL QWK", acc = "Accuracy", mae = "MAE", auc = "AUC (KL ≥ 2)",
                    x2 = "Error of ≥ 2\ngrades (rate)") |>
  merge_at(i = 1:2, j = 1, part = "body")
ft4 <- booktabs(ft4, c(2.3, 1.7, 2.9, 2.1, 2.2, 2.4, 2.4)) |>
  hline(i = c(2, 3, 4), border = B_HAIR, part = "body")

# ===================================================================================== save
# Word pagination (layout check 2026-11-24): keep every table on one page together with its note
keep_whole <- function(ft) keep_with_next(keep_with_next(ft, value = TRUE, part = "header"), value = TRUE, part = "body")
ft1 <- keep_whole(ft1); ft2 <- keep_whole(ft2); ft3 <- keep_whole(ft3); ft4 <- keep_whole(ft4)
tabs <- list(list(ft = ft1, title = TAB_CAP[[1]]$title, note = TAB_CAP[[1]]$note),
             list(ft = ft2, title = TAB_CAP[[2]]$title, note = note2),
             list(ft = ft3, title = TAB_CAP[[3]]$title, note = TAB_CAP[[3]]$note),
             list(ft = ft4, title = TAB_CAP[[4]]$title, note = TAB_CAP[[4]]$note))
saveRDS(tabs, file.path(TAB_DIR, "tables_v24.rds"))
for (i in seq_along(tabs)) {
  g <- gen_grob(tabs[[i]]$ft, fit = "fixed", just = "center")
  dims <- dim(tabs[[i]]$ft)
  w_in <- sum(dims$widths) + 0.2; h_in <- sum(dims$heights) * 1.6 + 0.4
  ragg::agg_png(file.path(TAB_DIR, sprintf("table%d_preview.png", i)), width = w_in, height = h_in, units = "in", res = 220, background = "white")
  grid::grid.newpage(); grid::grid.draw(g); invisible(dev.off())
}
all_cells <- bind_rows(lapply(seq_along(list(t1, t2, t3, t4)), function(i) {
  t <- list(t1, t2, t3, t4)[[i]]; data.frame(table = i, row = rep(seq_len(nrow(t)), ncol(t)), col = rep(names(t), each = nrow(t)), value = unlist(t))
}))
write.csv(all_cells, file.path(TAB_DIR, "tables_v24_cells.csv"), row.names = FALSE, fileEncoding = "UTF-8")
message("tables built and checked against v2.2: ", paste(sapply(list(t1, t2, t3, t4), nrow), collapse = "/"), " rows")
