#!/usr/bin/env Rscript
# Figure 5 (v2.4). Re-flip laterality test on the validation set at the selected epochs (P3 vs P4, three seeds).
# (a) knee-site pairs whose prediction is unchanged after re-flipping the image;
# (b) knee-site pairs whose prediction equals the original prediction for the contralateral site.
# Inputs: oarsi_heads_and_laterality_three_seeds.json, pvalues_flip_v2.json.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))

raw <- read_json_data("oarsi_heads_and_laterality_three_seeds.json")
pfl <- read_json_data("pvalues_flip_v2.json")$D_laterality_flip_validation_all_six
SUBS <- c(all_knees_all_heads = "All six sites", all_knees_narrowing_heads = "JSN sites",
          all_knees_osteophyte_heads = "OST sites", narrowing_jsm_ne_jsl = "Medial ≠ lateral JSN",
          osteophyte_osfm_ne_osfl_or_ostm_ne_ostl = "Medial ≠ lateral OST", narrowing_jsl_gt_jsm = "Lateral JSN > medial JSN")
GROUP <- c(rep("All validation knees", 3), rep("Asymmetric subsets\n(reference grades)", 3)); names(GROUP) <- names(SUBS)

d <- bind_rows(lapply(names(SUBS), function(sb) bind_rows(lapply(c("P3", "P4"), function(arm) bind_rows(lapply(SEEDS, function(s) {
  r <- raw$laterality_per_run[[sprintf("seed%s_%s", s, arm)]]$subsets[[sb]]
  data.frame(sub = sb, arm = arm, seed = paste("Seed", s), n = r$knees, same = 100 * num(r$same), swapped = 100 * num(r$swapped))
}))))))
stopifnot(all(tapply(d$n, d$sub, function(x) length(unique(x))) == 1))
nn <- tapply(d$n, d$sub, `[`, 1)
d <- d %>% mutate(lab = factor(sprintf("%s\n(n = %s)", SUBS[sub], fmt_int(nn[sub])), levels = rev(sprintf("%s\n(n = %s)", SUBS, fmt_int(nn[names(SUBS)])))),
                  grp = factor(GROUP[sub], levels = unique(GROUP)))
m <- d %>% group_by(sub, lab, grp, arm) %>%
  summarise(across(c(same, swapped), list(mean = mean, lo = min, hi = max)), .groups = "drop")
# check against pvalues_flip_v2 and manuscript section 3.1
for (sb in names(SUBS)) for (arm in c("P3", "P4"))
  stopifnot_close(m$same_mean[m$sub == sb & m$arm == arm] / 100, num(pfl[[sb]]$same_fraction_three_seed_mean[[arm]]), 1e-9, paste("same", sb, arm))
stopifnot_close(round(m$same_mean[m$sub == "all_knees_all_heads"], 1), c(70.4, 94.4), 1e-9, "3.1 headline")
stopifnot(all(sapply(names(SUBS), function(sb) num(pfl[[sb]]$P4_minus_P3$p_two_sided)) < 0.001))

ARMS2 <- c(P3 = "P3 (M/L labels swapped on flip)", P4 = "P4 (not swapped)")
panel <- function(v, title, xlab, show_y = TRUE) {
  mm <- m %>% transmute(lab, grp, arm, mean = .data[[paste0(v, "_mean")]], lo = .data[[paste0(v, "_lo")]], hi = .data[[paste0(v, "_hi")]])
  wide <- mm %>% select(lab, grp, arm, mean) %>% pivot_wider(names_from = arm, values_from = mean)
  dd <- d %>% transmute(lab, grp, arm, seed, val = .data[[v]])
  ggplot(mm, aes(y = lab)) +
    geom_segment(data = wide, aes(x = P3, xend = P4, yend = lab), colour = "#D5DAE0", linewidth = 2.2, lineend = "round") +
    geom_point(data = dd, aes(x = val, colour = arm), size = 1.35, alpha = 0.6, stroke = 0) +
    geom_point(aes(x = mean, fill = arm), shape = 21, colour = "white", size = 2.6, stroke = 0.45) +
    geom_text(aes(x = mean, label = sprintf("%.1f", mean), colour = arm), nudge_y = -0.34, family = FONT,
              size = pt2mm(5.7), fontface = "bold", show.legend = FALSE) +
    facet_grid(grp ~ ., scales = "free_y", space = "free_y", switch = "y") +
    scale_fill_manual(values = ARM_COL[c("P3", "P4")], labels = ARMS2, name = NULL) +
    scale_colour_manual(values = ARM_COL[c("P3", "P4")], guide = "none") +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 25), labels = function(x) paste0(x, "%"), expand = c(0.01, 0)) +
    scale_y_discrete(expand = expansion(add = 0.62)) +
    labs(title = title, x = xlab, y = NULL) +
    theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.25),
          strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 1, face = "bold", size = 6.2, colour = MUTED),
          panel.spacing.y = unit(2.5, "mm"),
          axis.text.y = if (show_y) element_text(size = 6.3, lineheight = 0.95) else element_blank(),
          axis.ticks.y = if (show_y) element_line() else element_blank(),
          strip.text.y = if (show_y) element_text() else element_blank())
}
p_a <- panel("same", "Unchanged after re-flipping", "Knee–site pairs (%)") +
  theme(plot.margin = margin(3, 10, 3, 3))
p_b <- panel("swapped", "Moved to the contralateral site", "Knee–site pairs (%)", show_y = FALSE) +
  theme(strip.text.y.left = element_blank(), plot.margin = margin(3, 6, 3, 6))
fig <- (p_a | p_b) + plot_layout(widths = c(1, 0.92), guides = "collect") + plot_annotation(tag_levels = "a") &
  theme(legend.position = "top", legend.margin = margin(0, 0, 0, 0)) & tag_theme
save_fig(fig, "fig5_laterality_flip", h_mm = 82)
