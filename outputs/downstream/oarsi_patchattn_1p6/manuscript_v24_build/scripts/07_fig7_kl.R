#!/usr/bin/env Rscript
# Figure 7 (v2.4). KL grading. (a) QWK differences and 2 x 2 factorial effects (participant-level cluster bootstrap,
# 10,000 resamples); (b) KL QWK per arm on validation (3 seeds) and held-out test (P0/P3 three seeds, P1/P2/P4 seed 17);
# (c) lambda sensitivity of P3 seed 17 on validation (KL QWK and mean OARSI QWK).
# Inputs: x1_factorial_three_seeds.json, x2_lambda_sensitivity.json, x3_p4_three_seeds.json, t5_results.json,
# pvalues_v1.json, reader_reliability.json.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))

x1 <- read_json_data("x1_factorial_three_seeds.json"); x2 <- read_json_data("x2_lambda_sensitivity.json")
x3 <- read_json_data("x3_p4_three_seeds.json"); t5 <- read_json_data("t5_results.json")
pv <- read_json_data("pvalues_v1.json"); rr <- read_json_data("reader_reliability.json")
retest_kl <- num(rr$results$all_visits_pooled$KL$kappa_quadratic)

# ---- panel a: effects -------------------------------------------------------------------
va <- pv$A_kl_validation_P3_minus_P0$selected; ta <- pv$A_kl_test_P3_minus_P0$primary_selected
f <- x1$q_sel$three_seed_mean
eff <- data.frame(
  lab = c("P3 − P0", "Attention pooling", "OARSI supervision", "Interaction", "P3 − P0"),
  set = c("Validation (3 seeds)", "Validation (3 seeds)", "Validation (3 seeds)", "Validation (3 seeds)", "Held-out test (3 seeds)"),
  kind = c("Primary comparison", "Main effect", "Main effect", "Interaction", "Primary comparison"),
  est = c(num(va$mean_diff), num(f$pooling_main$estimate), num(f$auxiliary_main$estimate), num(f$interaction$estimate), num(ta$mean_diff)),
  lo = c(num(va$ci95[[1]]), num(f$pooling_main$ci95[[1]]), num(f$auxiliary_main$ci95[[1]]), num(f$interaction$ci95[[1]]), num(ta$ci95[[1]])),
  hi = c(num(va$ci95[[2]]), num(f$pooling_main$ci95[[2]]), num(f$auxiliary_main$ci95[[2]]), num(f$interaction$ci95[[2]]), num(ta$ci95[[2]])),
  p = c(num(va$p_two_sided), NA, NA, NA, num(ta$p_two_sided)))
stopifnot_close(round(eff$est, 4), c(0.0050, 0.0021, 0.0029, -0.0020, 0.0058), 1e-9, "effects (Table 3/4)")
stopifnot_close(round(eff$p[c(1, 5)], 3), c(0.104, 0.051), 1e-9, "P values")
eff <- eff %>% mutate(row = factor(paste(lab, set, sep = "|"), levels = rev(paste(lab, set, sep = "|"))),
                      txt = minus(sprintf("%+.4f (%.4f, %.4f)%s", est, lo, hi, ifelse(is.na(p), "", sprintf("; P = %.3f", p)))))
eff$txt[5] <- minus(sprintf("%+.4f (%.5f, %.4f); P = %.3f", eff$est[5], eff$lo[5], eff$hi[5], eff$p[5]))
EFF_COL <- c("Primary comparison" = ARM_COL[["P3"]], "Main effect" = "#3C7CBE", "Interaction" = "#7D8794")
# numeric rows with a gap between the validation block (y = 7..4) and the held-out test row (y = 1.6)
eff$y <- c(7, 6, 5, 4, 1.6)

p_a <- ggplot(eff, aes(y = y)) +
  geom_vline(xintercept = 0, linetype = "22", linewidth = 0.3, colour = MUTED) +
  geom_errorbar(aes(xmin = lo, xmax = hi, colour = kind), width = 0, orientation = "y", linewidth = 0.6) +
  geom_point(aes(x = est, fill = kind), shape = 21, size = 2.3, colour = "white", stroke = 0.45) +
  annotate("text", x = -0.0064, y = c(7.62, 2.62), label = c("Validation (3 seeds)", "Held-out test (3 seeds)"), hjust = 0, vjust = 0.5,
           family = FONT, size = pt2mm(5.9), fontface = "bold", colour = MUTED) +
  scale_colour_manual(values = EFF_COL, guide = "none") + scale_fill_manual(values = EFF_COL, guide = "none") +
  scale_y_continuous(breaks = eff$y, labels = eff$lab, limits = c(0.9, 8.0), expand = c(0, 0),
                     sec.axis = sec_axis(~ ., breaks = eff$y, labels = eff$txt)) +
  scale_x_continuous(limits = c(-0.0068, 0.0128), breaks = seq(-0.005, 0.010, 0.005), labels = function(x) minus(sprintf("%+.3f", x)),
                     expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(title = "KL QWK differences (95% CI)", x = "Difference in KL QWK", y = NULL) +
  theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.25),
        axis.text.y.right = element_text(hjust = 0, size = 6.2), axis.ticks.y.right = element_blank(),
        axis.line.y.right = element_blank(), plot.margin = margin(3, 4, 3, 3))

# ---- panel b: KL QWK per arm --------------------------------------------------------------
vb <- bind_rows(
  bind_rows(lapply(SEEDS, function(s) data.frame(set = "Validation", arm = c("P0", "P1", "P2", "P3"), seed = paste("Seed", s),
                                                 qwk = num(x1$q_sel$per_seed[[s]]$values[c("P0", "P1", "P2", "P3")])))),
  bind_rows(lapply(SEEDS, function(s) data.frame(set = "Validation", arm = "P4", seed = paste("Seed", s), qwk = num(x3$per_seed[[s]]$P4$q_sel)))),
  bind_rows(lapply(SEEDS, function(s) data.frame(set = "Held-out test", arm = c("P0", "P3"), seed = paste("Seed", s),
                                                 qwk = c(num(t5$per_model[[sprintf("primary_selected_seed%s_P0", s)]]$qwk),
                                                         num(t5$per_model[[sprintf("primary_selected_seed%s_P3", s)]]$qwk))))),
  data.frame(set = "Held-out test", arm = c("P1", "P2", "P4"), seed = "Seed 17",
             qwk = sapply(c("P1", "P2", "P4"), function(a) num(t5$per_model[[paste0("secondary_selected_seed17_", a)]]$qwk))))
bm <- vb %>% group_by(set, arm) %>% summarise(mean = mean(qwk), lo = min(qwk), hi = max(qwk), n = n(), .groups = "drop")
chk <- function(s, a) round(bm$mean[bm$set == s & bm$arm == a], 4)
stopifnot_close(c(chk("Validation", "P0"), chk("Validation", "P1"), chk("Validation", "P2"), chk("Validation", "P3"), chk("Validation", "P4"),
                  chk("Held-out test", "P0"), chk("Held-out test", "P3")),
                c(0.8539, 0.8570, 0.8579, 0.8590, 0.8592, 0.8633, 0.8691), 1e-9, "KL QWK (Tables 3/4)")
stopifnot_close(round(sapply(c("P1", "P2", "P4"), function(a) chk("Held-out test", a)), 4), c(0.8744, 0.8655, 0.8674), 1e-9, "seed-17 controls")
lvl <- c(paste("Validation", c("P0", "P1", "P2", "P3", "P4")), paste("Held-out test", c("P0", "P3", "P1", "P2", "P4")))
xl <- c("P0", "P1", "P2", "P3", "P4", "P0", "P3", "P1†", "P2†", "P4†")
place <- function(df) { df$slot <- match(paste(df$set, df$arm), lvl); df$x <- df$slot + ifelse(df$slot > 5, 0.6, 0); df }
bm <- place(bm); vb <- place(vb)
xbreaks <- seq_along(lvl) + ifelse(seq_along(lvl) > 5, 0.6, 0)

p_b <- ggplot() +
  annotate("rect", xmin = 5.6 + 0.5, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#FBF7F1") +
  geom_hline(yintercept = retest_kl, linetype = "22", linewidth = 0.35, colour = REF) +
  annotate("text", x = 0.55, y = retest_kl + 0.0006, label = sprintf("OAI reader re-test %.3f", retest_kl), hjust = 0, vjust = 0,
           family = FONT, size = pt2mm(5.5), colour = REF) +
  geom_segment(data = filter(bm, n > 1), aes(x = x - 0.34, xend = x + 0.34, y = mean, yend = mean, colour = arm), linewidth = 0.9) +
  geom_point(data = vb, aes(x = x + ifelse(set == "Held-out test" & arm %in% c("P1", "P2", "P4"), 0,
                                           c(-0.2, 0, 0.2)[match(seed, names(SEED_SHAPE))]), y = qwk, shape = seed, fill = arm),
             size = 1.6, colour = "white", stroke = 0.35) +
  annotate("text", x = c(3, 8.6 + 0.5), y = 0.8795, label = c("Validation (3 seeds)", "Held-out test"),
           family = FONT, size = pt2mm(6), fontface = "bold", colour = MUTED) +
  scale_colour_manual(values = ARM_COL, guide = "none") + scale_fill_manual(values = ARM_COL, guide = "none") +
  scale_shape_manual(values = SEED_SHAPE, name = NULL) +
  scale_x_continuous(breaks = xbreaks, labels = xl, expand = expansion(add = 0.5)) +
  theme(axis.text.x = element_text(size = 6)) +
  scale_y_continuous(limits = c(0.850, 0.881), breaks = seq(0.850, 0.880, 0.005), labels = function(x) sprintf("%.3f", x), expand = c(0, 0)) +
  guides(shape = guide_legend(override.aes = list(fill = "#6B7280", size = 1.8))) +
  labs(title = "KL agreement per arm", x = NULL, y = "KL QWK (5 grades)") +
  theme(legend.position = "top", legend.margin = margin(0, 0, -2, 0), panel.grid.major.y = element_line(colour = GRID, linewidth = 0.25))

# ---- panel c: lambda sensitivity (seed 17, validation) -------------------------------------
lam <- data.frame(cond = factor(c("P1", "×0.5", "×1", "×2"), levels = c("P1", "×0.5", "×1", "×2")),
                  lambda = c(0, num(x2$lambda_P3$x0.5), num(x2$lambda_P3$x1), num(x2$lambda_P3$x2)),
                  kl = c(num(x1$q_sel$per_seed[["17"]]$values$P1), num(x2$x0.5$q_sel), num(x2$x1$q_sel), num(x2$x2$q_sel)),
                  oarsi = c(NA, num(x2$x0.5$oarsi_mean_qwk), num(x2$x1$oarsi_mean_qwk), num(x2$x2$oarsi_mean_qwk)))
p0_17 <- num(x2$seed17_P0$q_sel)
stopifnot_close(round(lam$kl, 4), c(0.8591, 0.8610, 0.8628, 0.8604), 1e-9, "lambda KL"); stopifnot(round(p0_17, 4) == 0.8548)
stopifnot_close(round(lam$oarsi[-1], 3), c(0.816, 0.821, 0.812), 1e-9, "lambda OARSI")
lam$xlab <- sprintf("%s\n%.3f", lam$cond, lam$lambda); lam$xlab[1] <- "P1\n0"
lam$xlab <- factor(lam$xlab, levels = lam$xlab)
p_c1 <- ggplot(lam, aes(xlab, kl, group = 1)) +
  geom_hline(yintercept = p0_17, linetype = "13", linewidth = 0.35, colour = ARM_COL[["P0"]]) +
  geom_hline(yintercept = retest_kl, linetype = "22", linewidth = 0.35, colour = REF) +
  annotate("text", x = 4.45, y = p0_17 - 0.0004, label = "P0 seed 17", hjust = 1, vjust = 1, family = FONT, size = pt2mm(5.3), colour = MUTED) +
  geom_line(colour = "#9AA3AE", linewidth = 0.45) +
  geom_point(aes(fill = cond == "×1"), shape = 21, size = 2.3, colour = "white", stroke = 0.45) +
  geom_text(aes(label = sprintf("%.4f", kl)), nudge_y = -0.0011, family = FONT, size = pt2mm(5.4)) +
  annotate("text", x = 0.55, y = retest_kl + 0.0003, label = "OAI re-test", hjust = 0, vjust = 0, family = FONT, size = pt2mm(5.3), colour = REF) +
  scale_fill_manual(values = c(`TRUE` = ARM_COL[["P3"]], `FALSE` = ARM_COL[["P1"]]), guide = "none") +
  scale_x_discrete(expand = expansion(add = 0.75)) +
  scale_y_continuous(limits = c(0.853, 0.8665), breaks = seq(0.854, 0.866, 0.004), labels = function(x) sprintf("%.3f", x)) +
  labs(title = "Auxiliary weight λ (seed 17)", x = NULL, y = "KL QWK") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), plot.margin = margin(3, 4, 0, 3))
p_c2 <- ggplot(lam[-1, ], aes(xlab, oarsi, group = 1)) +
  geom_line(colour = "#9AA3AE", linewidth = 0.45) +
  geom_point(aes(fill = cond == "×1"), shape = 21, size = 2.3, colour = "white", stroke = 0.45) +
  geom_text(aes(label = sprintf("%.3f", oarsi)), nudge_y = 0.0022, family = FONT, size = pt2mm(5.4)) +
  annotate("text", x = 1, y = 0.8165, label = "no OARSI\nheads", family = FONT, size = pt2mm(5.2), colour = MUTED, lineheight = 0.95) +
  scale_x_discrete(drop = FALSE, limits = levels(lam$xlab), expand = expansion(add = 0.75)) +
  scale_fill_manual(values = c(`TRUE` = ARM_COL[["P3"]], `FALSE` = ARM_COL[["P1"]]), guide = "none") +
  scale_y_continuous(limits = c(0.806, 0.826), breaks = seq(0.808, 0.824, 0.008), labels = function(x) sprintf("%.3f", x)) +
  labs(x = "\u03bb (multiple of calibrated value)", y = "Mean OARSI QWK") +
  theme(axis.title.x = element_text(size = 6.2)) +
  theme(axis.text.x = element_text(size = 5.6, lineheight = 0.95), plot.margin = margin(0, 4, 3, 3))
p_c <- (p_c1 / p_c2) + plot_layout(heights = c(1.15, 1))

fig <- (free(p_a) / ((free(p_b, type = "label") | p_c) + plot_layout(widths = c(1.6, 1)))) + plot_layout(heights = c(0.62, 1)) +
  plot_annotation(tag_levels = list(c("a", "b", "c", ""))) & tag_theme
save_fig(fig, "fig7_kl", h_mm = 110)
