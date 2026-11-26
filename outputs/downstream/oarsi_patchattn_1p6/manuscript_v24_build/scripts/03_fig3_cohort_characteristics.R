#!/usr/bin/env Rscript
# Figure 3 (v2.4). Cohort KL composition (a), visit composition (b), four-site osteophyte completeness by KL (c),
# automated QC exclusion rate by KL (d). Inputs: kl_and_qc_by_split.csv, visits_and_modalities.csv, cohort_summary.json.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))
suppressPackageStartupMessages(library(readr))

SPL <- c("Training", "Validation", "Held-out test")
recode_split <- function(x) factor(ifelse(x == "Held-out Test", "Held-out test", x), levels = SPL)
kl <- read_csv(file.path(DATA_DIR, "kl_and_qc_by_split.csv"), show_col_types = FALSE) %>%
  mutate(split_label = recode_split(split_label), kl_lab = factor(paste("KL", kl), levels = paste("KL", 0:4)),
         excl_pct = 100 * held_n / (usable_n + held_n))
vis <- read_csv(file.path(DATA_DIR, "visits_and_modalities.csv"), show_col_types = FALSE) %>%
  filter(kind == "visit") %>%
  mutate(split_label = recode_split(split_label),
         visit_lab = factor(sprintf("%s\n%d mo", category, months),
                            levels = sprintf("%s\n%d mo", c("V00", "V01", "V03", "V05", "V06", "V08", "V10"), c(0, 12, 24, 36, 48, 72, 96))))
cs <- read_json_data("cohort_summary.json")
n_knees <- c(Training = cs$splits$train$knees, Validation = cs$splits$val$knees, `Held-out test` = cs$splits$test$knees)
overall_ost <- 100 * cs$total$all_four_osteophytes / cs$total$knees

# ---- consistency checks against the manuscript text ---------------------------------------
stopifnot_close(kl$excl_pct, kl$held_rate_pct, 0.01, "QC exclusion rate definition")
# v2.2 caption said "KL 2-4: 96%-99%"; the data give 94.9%-98.7% (corrected in the v2.4 caption)
stopifnot(all(abs(range(kl$all_ost_pct[kl$kl == 0]) - c(16.14, 16.96)) < 0.01),
          all(abs(range(kl$all_ost_pct[kl$kl >= 2]) - c(94.89, 98.67)) < 0.01),
          abs(min(kl$excl_pct[kl$kl == 4]) - 13.2) < 0.05, abs(max(kl$excl_pct[kl$kl == 4]) - 15.5) < 0.05,
          abs(overall_ost - 53.2) < 0.05)
for (s in SPL) stopifnot(sum(kl$usable_n[kl$split_label == s]) == n_knees[[s]],
                         sum(vis$n[vis$split_label == s]) == n_knees[[s]])

leg_lab <- sprintf("%s (n = %s knee-visits)", SPL, fmt_int(n_knees[SPL]))
fill_scale <- scale_fill_manual(values = SPLIT_COL, labels = leg_lab, name = NULL)
col_scale <- scale_colour_manual(values = SPLIT_COL, labels = leg_lab, name = NULL)

p_a <- ggplot(kl, aes(kl_lab, usable_pct, fill = split_label)) +
  geom_col(position = position_dodge(0.8), width = 0.74, colour = "white", linewidth = 0.15) +
  fill_scale +
  scale_y_continuous(limits = c(0, 50), breaks = seq(0, 50, 10), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "KL grade composition", x = "Reference KL grade", y = "Knee-visits (%)")

p_b <- ggplot(vis, aes(visit_lab, pct, fill = split_label)) +
  geom_col(position = position_dodge(0.8), width = 0.74, colour = "white", linewidth = 0.15) +
  fill_scale +
  scale_y_continuous(limits = c(0, 25), breaks = seq(0, 25, 5), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Visit composition", x = "OAI visit (months from baseline)", y = "Knee-visits (%)")

line_panel <- function(yvar, ylim, brk, title, ytitle) {
  ggplot(kl, aes(kl_lab, .data[[yvar]], colour = split_label, group = split_label)) +
    geom_line(linewidth = 0.6) +
    geom_point(aes(fill = split_label), shape = 21, colour = "white", size = 2.0, stroke = 0.4) +
    scale_colour_manual(values = SPLIT_COL, guide = "none") + scale_fill_manual(values = SPLIT_COL, guide = "none") +
    scale_y_continuous(limits = ylim, breaks = brk, expand = expansion(mult = c(0, 0.03))) +
    labs(title = title, x = "Reference KL grade", y = ytitle)
}
p_c <- line_panel("all_ost_pct", c(0, 100), seq(0, 100, 25), "Four-site osteophyte completeness",
                  "Knees with all four OST grades (%)") +
  geom_hline(yintercept = overall_ost, linetype = "22", linewidth = 0.3, colour = MUTED) +
  annotate("text", x = 5.35, y = overall_ost + 4, hjust = 1, family = FONT, size = pt2mm(5.6), colour = MUTED,
           label = sprintf("whole cohort %.1f%%", overall_ost))
p_d <- line_panel("excl_pct", c(0, 18), seq(0, 18, 3), "Automated QC exclusion",
                  "QC-excluded knees (%)")

fig <- (p_a + p_b + p_c + p_d) + plot_layout(ncol = 2, guides = "collect", widths = c(1, 1.25)) +
  plot_annotation(tag_levels = "a") &
  theme(legend.position = "top", legend.justification = "center", legend.margin = margin(0, 0, 2, 0),
        legend.key.size = unit(2.8, "mm"), legend.text = element_text(size = 6.3)) & tag_theme
save_fig(fig, "fig3_cohort_characteristics", h_mm = 104)
