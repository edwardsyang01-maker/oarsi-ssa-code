#!/usr/bin/env Rscript
# Figure 6 (v2.4). Validation OARSI grading agreement per site (P2, P3, P4; three seeds) against the OAI reader
# re-test reference (a) and the paired P3 - P2 gain with participant-level cluster-bootstrap 95% CI (b).
# Inputs: oarsi_heads_and_laterality_three_seeds.json, pvalues_v1.json, reader_reliability.json.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))

raw <- read_json_data("oarsi_heads_and_laterality_three_seeds.json")
pv <- read_json_data("pvalues_v1.json")$B_oarsi_validation_P3_minus_P2_three_seeds
rr <- read_json_data("reader_reliability.json")$results$all_visits_pooled

pts <- bind_rows(lapply(c("P2", "P3", "P4"), function(arm) bind_rows(lapply(SITES, function(h) bind_rows(lapply(SEEDS, function(s)
  data.frame(arm = arm, site = h, seed = paste("Seed", s), qwk = num(raw$oarsi_heads[[arm]]$per_seed[[s]][[h]]$qwk))))))))
mm <- pts %>% group_by(arm, site) %>% summarise(mean = mean(qwk), lo = min(qwk), hi = max(qwk), .groups = "drop")
ref <- data.frame(site = SITES, qwk = sapply(SITES, function(h) num(rr[[toupper(h)]]$kappa_quadratic)))
gain <- bind_rows(lapply(SITES, function(h) {
  e <- pv[[h]]$P3_minus_P2
  data.frame(site = h, mean = num(e$mean_diff), lo = num(e$ci95[[1]]), hi = num(e$ci95[[2]]), p = num(e$p_two_sided))
}))
gain_seed <- bind_rows(lapply(SITES, function(h) data.frame(site = h, seed = paste("Seed", SEEDS),
                                                           diff = sapply(SEEDS, function(s) num(pv[[h]]$P3_minus_P2$per_seed_diff[[s]])))))
# checks against Table 2 / section 3.2
stopifnot_close(round(gain$mean, 3), c(0.060, 0.080, 0.088, 0.140, 0.010, -0.049), 1e-9, "P3-P2 gains")
p3m <- mm[mm$arm == "P3", ]
stopifnot_close(round(p3m$mean[match(SITES, p3m$site)], 3), c(0.819, 0.703, 0.784, 0.787, 0.868, 0.826), 1e-9, "P3 validation QWK")
stopifnot_close(round(ref$qwk, 3), c(0.849, 0.794, 0.827, 0.817, 0.867, 0.897), 1e-9, "reader re-test QWK")
for (arm in c("P2", "P3", "P4")) for (h in SITES)
  stopifnot_close(mm$mean[mm$arm == arm & mm$site == h], num(raw$oarsi_heads[[arm]]$three_seed_summary[[h]]$qwk$mean), 1e-12, "means")

gain$seed_hi <- tapply(gain_seed$diff, gain_seed$site, max)[gain$site]
gain$seed_lo <- tapply(gain_seed$diff, gain_seed$site, min)[gain$site]
xpos <- setNames(seq_along(SITES) + c(0, 0, 0, 0, 0.35, 0.35), SITES)
off <- c(P2 = -0.25, P3 = 0, P4 = 0.25)
mm$x <- xpos[mm$site] + off[mm$arm]; pts$x <- xpos[pts$site] + off[pts$arm]
ref$x <- xpos[ref$site]
xlabs <- scale_x_continuous(breaks = xpos, labels = c("Medial", "Lateral", "Medial", "Lateral", "Medial", "Lateral"), expand = expansion(add = 0.62))
grp_lab <- data.frame(x = c(mean(xpos[1:2]), mean(xpos[3:4]), mean(xpos[5:6])), lab = c("Femoral OST", "Tibial OST", "JSN"))
add_groups <- function(p, y, ybar, ylim, xpad = 0.62) p +
  geom_text(data = grp_lab, aes(x = x, y = y, label = lab), inherit.aes = FALSE, family = FONT, size = pt2mm(6.2), fontface = "bold", vjust = 1) +
  geom_segment(data = grp_lab, aes(x = x - 0.62, xend = x + 0.62, y = ybar, yend = ybar), inherit.aes = FALSE, colour = "#9AA3AE", linewidth = 0.3) +
  coord_cartesian(xlim = range(xpos) + c(-xpad, xpad), ylim = ylim, expand = FALSE, clip = "off") + theme(plot.margin = margin(3, 4, 24, 3))

p_a <- ggplot() +
  geom_vline(xintercept = mean(xpos[c("ostl", "jsm")]), colour = "#C9CED6", linewidth = 0.3, linetype = "13") +
  geom_segment(data = ref, aes(x = x - 0.42, xend = x + 0.42, y = qwk, yend = qwk, linetype = "OAI reader re-test"),
               colour = REF, linewidth = 0.5) +
  geom_segment(data = mm, aes(x = x, xend = x, y = lo, yend = hi, colour = arm), linewidth = 1.6, alpha = 0.35, lineend = "round") +
  geom_point(data = mm, aes(x = x, y = mean, fill = arm), shape = 21, size = 2.5, colour = "white", stroke = 0.45) +
  scale_colour_manual(values = ARM_COL[c("P2", "P3", "P4")], guide = "none") +
  scale_fill_manual(values = ARM_COL[c("P2", "P3", "P4")], name = NULL,
                    labels = c(P2 = "P2 (global CLS token)", P3 = "P3 (site queries)", P4 = "P4 (no label swap)")) +
  scale_linetype_manual(values = c("OAI reader re-test" = "solid"), name = NULL) +
  scale_y_continuous(breaks = seq(0.55, 0.90, 0.05)) +
  xlabs +
  guides(fill = guide_legend(order = 1), linetype = guide_legend(order = 2, override.aes = list(linewidth = 0.6)),
         shape = guide_legend(order = 3)) +
  labs(title = "Validation agreement per OARSI site (3 seeds)", x = NULL, y = "QWK (4 grades)")
p_a <- add_groups(p_a, 0.55 - 0.046, 0.55 - 0.036, c(0.55, 0.92))

p_b <- ggplot(gain, aes(x = xpos[site])) +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.3) +
  geom_col(aes(y = mean, fill = mean > 0), width = 0.58) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0, linewidth = 0.45, colour = INK) +
  geom_point(data = gain_seed, aes(x = xpos[site] + c(-0.17, 0, 0.17)[match(seed, names(SEED_SHAPE))], y = diff, shape = seed), size = 1.05, colour = "#4B5563",
             fill = "white", stroke = 0.3, inherit.aes = FALSE) +
  geom_text(aes(y = ifelse(mean > 0, pmax(hi, seed_hi) + 0.018, pmin(lo, seed_lo) - 0.014), label = fmt_p(p), vjust = ifelse(mean > 0, 0, 1)),
            family = FONT, size = pt2mm(5.6), colour = INK) +
  scale_fill_manual(values = c(`TRUE` = ARM_COL[["P3"]], `FALSE` = ARM_COL[["P2"]]), guide = "none") +
  scale_shape_manual(values = SEED_SHAPE, name = NULL) +
  scale_y_continuous(breaks = seq(-0.15, 0.20, 0.05), labels = function(x) minus(sprintf("%+.2f", x))) +
  xlabs +
  labs(title = "Paired gain from site queries (P3 − P2, 95% CI)", x = NULL, y = "Difference in QWK")
p_b <- add_groups(p_b, -0.175 - 0.052, -0.175 - 0.040, c(-0.175, 0.24), xpad = 0.85)

fig <- (p_a | p_b) + plot_layout(widths = c(1.12, 1), guides = "collect") + plot_annotation(tag_levels = "a") &
  theme(legend.position = "top", legend.margin = margin(0, 0, 0, 0), legend.spacing.x = unit(1.5, "mm"),
        axis.text.x = element_text(size = 6.1, lineheight = 1.0)) & tag_theme
save_fig(fig, "fig6_oarsi", h_mm = 80)
