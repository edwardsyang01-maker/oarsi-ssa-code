#!/usr/bin/env Rscript
# Figure 4 (v2.4). Validation-set (3,962 knees) mean attention of each query, averaged over the 8 heads, on the
# 28 x 28 patch grid (a); P3 medial-half share and vertical centroid (b); P3 top-5% concentration and Spearman rho
# with the site grade (c). Input: x5_attention_summary.json, mean_val_knee_448.png.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))

att <- read_json_data("x5_attention_summary.json")
RUNS <- c(seed17_P3 = "P3\nseed 17", seed23_P3 = "P3\nseed 23", seed29_P3 = "P3\nseed 29",
          seed17_P4 = "P4\nseed 17", seed17_P1 = "P1\nseed 17")
QUERIES <- c("kl", SITES)
Q_LAB <- c(kl = "KL query", SITE_LAB2)

# ---- panel a data ----------------------------------------------------------------------
tiles <- list()
for (r in names(RUNS)) for (q in QUERIES) {
  m <- att$runs[[r]]$mean_attention_map_28x28[[q]]
  if (is.null(m)) next
  v <- matrix(num(m), nrow = 28, byrow = TRUE) * 784         # multiples of the uniform weight 1/784
  stopifnot(abs(sum(v) / 784 - 1) < 1e-4)
  g <- expand.grid(col = 0:27, row = 0:27)
  tiles[[length(tiles) + 1]] <- data.frame(run = r, query = q, x = g$col + 0.5, y = 27.5 - g$row,
                                           val = v[cbind(g$row + 1, g$col + 1)])
}
tiles <- bind_rows(tiles) %>%
  mutate(run = factor(run, levels = names(RUNS), labels = RUNS), query = factor(query, levels = QUERIES, labels = Q_LAB[QUERIES]))
vmax <- max(tiles$val)
blank <- expand.grid(run = factor(RUNS[["seed17_P1"]], levels = RUNS), query = factor(Q_LAB[SITES], levels = Q_LAB[QUERIES]))
note <- data.frame(run = factor(RUNS[["seed17_P1"]], levels = RUNS), query = factor(Q_LAB[["jsl"]], levels = Q_LAB[QUERIES]),
                   x = 28, y = 14, label = "P1 pools with a single KL query and has no OARSI site queries")

knee <- png::readPNG(file.path(ASSET_DIR, "mean_val_knee_448.png"))
if (length(dim(knee)) == 3) knee <- knee[, , 1]
lum <- pmin(0.985, pmax(0.62, 0.975 - 0.33 * knee))            # light, inverted backdrop so colour reads clearly
backdrop <- array(c(lum * 0.985, lum * 0.99, lum), dim = c(dim(knee), 3))

HEAT <- c("#FFF3B0", "#FDBE5C", "#F2804A", "#D2463F", "#9C1F4E", "#4E0F4A")
alpha_of <- function(v) pmin(1, pmax(0, (v - 1) / 9))^0.65

p_a <- ggplot(tiles) +
  annotation_raster(backdrop, 0, 28, 0, 28, interpolate = TRUE) +
  geom_raster(aes(x, y, fill = val, alpha = alpha_of(val))) +
  scale_alpha_identity() +
  geom_rect(data = transform(blank, xmin = -0.5, xmax = 28.5, ymin = -0.5, ymax = 28.5), aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax), fill = "white", colour = NA, inherit.aes = FALSE) +
  geom_text(data = note, aes(x, y, label = label), hjust = 1, family = FONT, size = pt2mm(6), colour = MUTED, inherit.aes = FALSE) +
  geom_segment(data = transform(distinct(tiles, run, query), x = 14, xend = 14, y = 0, yend = 28), aes(x = x, xend = xend, y = y, yend = yend),
               colour = "#5D6673", linewidth = 0.25, linetype = "22") +
  facet_grid(run ~ query, switch = "y") +
  scale_fill_gradientn(colours = HEAT, limits = c(0, vmax), breaks = c(0, 10, 20, 30),
                       name = "Mean attention weight (multiples of uniform, 1/784)") +
  scale_x_continuous(breaks = c(7, 21), labels = c("Lat.", "Med."), expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_fixed(xlim = c(0, 28), ylim = c(0, 28), clip = "off") +
  guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5, barwidth = unit(55, "mm"),
                                barheight = unit(2.2, "mm"), ticks.colour = "white", frame.colour = NA)) +
  theme_void(base_family = FONT) +
  theme(strip.text.x = element_text(size = 6.4, face = "bold", lineheight = 1.0, margin = margin(b = 2)),
        strip.text.y.left = element_text(size = 6.4, face = "bold", angle = 0, hjust = 1, lineheight = 1.0, margin = margin(r = 3)),
        axis.text.x = element_text(size = 5.4, colour = MUTED, margin = margin(t = 1)),
        panel.spacing.x = unit(1.4, "mm"), panel.spacing.y = unit(1.4, "mm"),
        legend.position = "bottom", legend.title = element_text(size = 6.3), legend.text = element_text(size = 6),
        legend.margin = margin(t = 0), plot.margin = margin(2, 2, 2, 2))

# ---- panels b-c data (P3, three seeds) -------------------------------------------------------
P3R <- c("seed17_P3", "seed23_P3", "seed29_P3")
side_of <- c(kl = "KL", SITE_SIDE)
qd <- bind_rows(lapply(QUERIES, function(q) {
  med <- sapply(P3R, function(r) num(att$runs[[r]]$metrics[[q]]$right_half_fraction$mean))   # canonical right-knee view: right half = medial
  row <- sapply(P3R, function(r) num(att$runs[[r]]$metrics[[q]]$vertical_centroid_row$mean))
  top <- sapply(P3R, function(r) 100 * num(att$runs[[r]]$metrics[[q]]$top5pct_share$mean))
  rho <- if (q == "kl") NA_real_ else mean(sapply(P3R, function(r) num(att$runs[[r]]$concentration_by_site_grade[[q]]$spearman_grade_vs_top5_share)))
  data.frame(q = q, lab = c(kl = "KL query", SITE_LAB)[q], side = side_of[[q]],
             med = mean(med), med_lo = min(med), med_hi = max(med), row = mean(row),
             top = mean(top), top_lo = min(top), top_hi = max(top), rho = rho)
})) %>% mutate(lab = factor(lab, levels = rev(c("KL query", SITE_LAB[SITES]))),
               side = factor(side, levels = c("Medial", "Lateral", "KL")))
# values quoted in section 3.1
stopifnot_close(round(qd$med, 3), c(0.619, 0.583, 0.433, 0.572, 0.410, 0.591, 0.502), 1e-9, "medial-half shares (3.1)")

p_b <- ggplot(qd, aes(y = lab)) +
  geom_vline(xintercept = 0.5, linetype = "22", linewidth = 0.3, colour = MUTED) +
  geom_segment(aes(x = 0.5, xend = med, yend = lab, colour = side), linewidth = 1.6, alpha = 0.35, lineend = "butt") +
  geom_errorbar(aes(xmin = med_lo, xmax = med_hi), width = 0, orientation = "y", linewidth = 0.4, colour = INK) +
  geom_point(aes(x = med, fill = side), shape = 21, size = 2.1, colour = "white", stroke = 0.4) +
  geom_text(aes(x = 0.665, label = sprintf("%.3f", med)), hjust = 0, family = FONT, size = pt2mm(5.8)) +
  geom_text(aes(x = 0.730, label = sprintf("%.1f", row)), hjust = 0, family = FONT, size = pt2mm(5.8), colour = MUTED) +
  annotate("text", x = c(0.665, 0.718), y = 7.75, label = c("share", "centroid row"), hjust = 0, family = FONT,
           size = pt2mm(5.6), fontface = "bold", colour = c(INK, MUTED)) +
  scale_colour_manual(values = SIDE_COL, guide = "none") +
  scale_fill_manual(values = SIDE_COL, name = NULL, labels = c("Medial site query", "Lateral site query", "KL query")) +
  scale_x_continuous(limits = c(0.35, 0.78), breaks = seq(0.35, 0.65, 0.05), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.1))) +
  coord_cartesian(clip = "off") +
  labs(title = "Medial-half attention share (P3)", x = "Share of attention on the medial half", y = NULL) +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.25))

p_c <- ggplot(qd, aes(y = lab)) +
  geom_vline(xintercept = 100 * 40 / 784, linetype = "22", linewidth = 0.3, colour = MUTED) +
  geom_col(aes(x = top, fill = side), width = 0.62) +
  geom_errorbar(aes(xmin = top_lo, xmax = top_hi), width = 0, orientation = "y", linewidth = 0.4, colour = INK) +
  geom_text(aes(x = top_hi + 2, label = ifelse(is.na(rho), sprintf("%.1f%%", top), sprintf("%.1f%%  (ρ = %.2f)", top, rho))),
            hjust = 0, family = FONT, size = pt2mm(5.8)) +
  scale_fill_manual(values = SIDE_COL, guide = "none") +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 80, 20), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.1))) +
  labs(title = "Attention in the top 5% of patches (P3)", x = "Attention mass in the top 40 of 784 patches (%)", y = NULL) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), plot.margin = margin(3, 4, 3, 6),
        panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.25))

bottom <- (p_b | p_c) + plot_layout(widths = c(1.05, 1), guides = "collect") &
  theme(legend.position = "bottom", legend.margin = margin(t = -2))
fig <- wrap_elements(full = p_a) / bottom + plot_layout(heights = c(2.95, 1))
fig <- fig + plot_annotation(tag_levels = "a") & tag_theme
save_fig(fig, "fig4_attention", h_mm = 176)
