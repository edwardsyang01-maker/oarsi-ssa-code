#!/usr/bin/env Rscript
# Figure 2 (v2.4). (a) Split-specific inclusion/exclusion flow (all counts read from cohort_summary.json and
# checked for arithmetic consistency); (b-f) image-level preprocessing stages on de-identified example films.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))

cs <- read_json_data("cohort_summary.json")
tr <- cs$splits$train; va <- cs$splits$val; te <- cs$splits$test; to <- cs$total

# ---- arithmetic checks (fail loudly if the source changes) -------------------------------
for (s in list(tr, va)) {
  stopifnot(s$starting_archives - s$excluded_kl_link_films - s$excluded_tech_films == s$starting_pool_films,
            2 * s$starting_pool_films - s$crop_failed_knees == s$pre_qc_knees,
            s$pre_qc_knees - s$held_knees == s$knees)
}
stopifnot(te$starting_archives - te$excluded_tech_films - te$excluded_repair_trigger_films == te$completed_image_films,
          2 * te$completed_image_films == te$locked_knees,
          te$locked_knees - te$excluded_kl_link_knees - te$held_knees == te$knees,
          tr$patients + va$patients + te$patients == to$patients,
          tr$films + va$films + te$films == to$films,
          tr$knees + va$knees + te$knees == to$knees,
          to$patients == 4420, to$films == 20806, to$knees == 39851)

# ---- panel a ---------------------------------------------------------------------------
NEUT_FILL <- "#F6F7F9"; NEUT_COL <- "#B7BEC8"
bx <- function(xmin, xmax, ymin, ymax, fill, col, lw = 0.35)
  data.frame(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill, col = col, lw = lw, lt = "solid", r = 1.2)
TXT <- list()
add_text <- function(x, ytop, title, body = NULL, pt = 5.8, title_col = INK) {
  TXT[[length(TXT) + 1]] <<- data.frame(x = x, y = ytop - 2.0, label = title, face = "bold", size = pt2mm(pt + 0.2), col = title_col, vjust = 0.5)
  if (!is.null(body)) TXT[[length(TXT) + 1]] <<- data.frame(x = x, y = ytop - 3.6, label = body, face = "plain", size = pt2mm(pt), col = INK, vjust = 1)
}

cols <- list(train = c(1, 52), val = c(54.5, 105.5), test = c(108, 159))
cx <- sapply(cols, mean)
split_name <- c(train = "Training", val = "Validation", test = "Held-out test")
R1 <- c(55.5, 64); R2 <- c(31, 52); R3 <- c(18, 27.5); R4 <- c(1.5, 14.5)

B <- list(bx(40, 120, 68, 77, "#EEF1F5", "#7C8794", 0.4))
add_text(80, 77, "OAI cohort with a fixed participant-level split (n = 4,796)",
         "bilateral standing fixed-flexion PA radiographs, 7 visits (V00–V10)")
for (k in names(cols)) {
  x0 <- cols[[k]][1]; x1 <- cols[[k]][2]; sc <- SPLIT_COL[[split_name[k]]]; sf <- SPLIT_FILL_LIGHT[[split_name[k]]]
  B[[length(B) + 1]] <- bx(x0, x1, R1[1], R1[2], sf, sc, 0.45)
  B[[length(B) + 1]] <- bx(x0, x1, R2[1], R2[2], NEUT_FILL, NEUT_COL)
  B[[length(B) + 1]] <- bx(x0, x1, R3[1], R3[2], NEUT_FILL, NEUT_COL)
  B[[length(B) + 1]] <- bx(x0, x1, R4[1], R4[2], sf, sc, 0.45)
}
pick <- function(k) switch(k, train = tr, val = va, test = te)
for (k in c("train", "val")) {
  s <- pick(k)
  add_text(cx[k], R1[2], sprintf("%s split: %s participants", split_name[k], fmt_int(s$assigned_patients)),
           sprintf("%s non-empty bilateral PA films", fmt_int(s$starting_archives)), title_col = SPLIT_COL[[split_name[k]]])
  add_text(cx[k], R2[2], "Metadata and technical screening",
           paste0("−", fmt_int(s$excluded_kl_link_films), " films without valid bilateral KL grades\n",
                  "−", fmt_int(s$excluded_tech_films), " films failing DICOM rescale-header\nor dynamic-range checks\n",
                  fmt_int(s$starting_pool_films), " films → ", fmt_int(s$pre_qc_knees), " knee crops (", s$crop_failed_knees, " crop failures)\n",
                  "gray-level anomalies: blinded visual choice\nof an alternative conversion"))
  add_text(cx[k], R3[2], "1.6× crop, ruler masking and automated QC",
           paste0("−", fmt_int(s$held_knees), " knees excluded by QC"))
  add_text(cx[k], R4[2], sprintf("Usable %s set", tolower(split_name[k])),
           sprintf("%s participants · %s films\n%s knee-visits (%s ruler-masked)\nJSN graded: %s · all four OST graded: %s",
                   fmt_int(s$patients), fmt_int(s$films), fmt_int(s$knees), fmt_int(s$cleanup_status$accepted_cleaned),
                   fmt_int(s$both_narrowing), fmt_int(s$all_four_osteophytes)), title_col = SPLIT_COL[[split_name[k]]])
}
add_text(cx["test"], R1[2], sprintf("Held-out test split: %s participants", fmt_int(te$assigned_patients)),
         sprintf("%s bilateral PA films (%d single-knee removed)", fmt_int(te$starting_archives), te$excluded_single_knee_baseline),
         title_col = SPLIT_COL[["Held-out test"]])
add_text(cx["test"], R2[2], "Locked automatic processing (before labels)",
         paste0("−", fmt_int(te$excluded_tech_films), " films with technical processing failure\n",
                "−", fmt_int(te$excluded_repair_trigger_films), " films with gray-level anomalies\n(whole film excluded, no manual selection)\n",
                fmt_int(te$completed_image_films), " films → ", fmt_int(te$locked_knees), " knee crops"))
add_text(cx["test"], R3[2], "KL linkage and automated QC",
         paste0("−", fmt_int(te$excluded_kl_link_knees), " knees without valid paired KL grade\n",
                "−", fmt_int(te$held_knees), " knees excluded by QC"))
add_text(cx["test"], R4[2], "Usable held-out test set",
         sprintf("%s participants · %s films\n%s knee-visits (%s ruler-masked)\nJSN graded: %s · all four OST graded: %s",
                 fmt_int(te$patients), fmt_int(te$films), fmt_int(te$knees), fmt_int(te$cleanup_status$accepted_cleaned),
                 fmt_int(te$both_narrowing), fmt_int(te$all_four_osteophytes)), title_col = SPLIT_COL[["Held-out test"]])

boxes <- do.call(rbind, B)
texts <- do.call(rbind, TXT)
arrows <- rbind(
  data.frame(id = 1, x = c(80, 80, cx["train"], cx["train"]), y = c(68, 66.1, 66.1, R1[2])),
  data.frame(id = 2, x = c(80, 80), y = c(68, R1[2])),
  data.frame(id = 3, x = c(80, 80, cx["test"], cx["test"]), y = c(68, 66.1, 66.1, R1[2]))
)
i <- 4
for (k in names(cols)) for (pr in list(c(R1[1], R2[2]), c(R2[1], R3[2]), c(R3[1], R4[2]))) {
  arrows <- rbind(arrows, data.frame(id = i, x = c(cx[k], cx[k]), y = pr)); i <- i + 1
}
poly <- boxes_to_poly(boxes)
p_a <- ggplot() +
  geom_path(data = arrows, aes(x, y, group = id), colour = "#4A5361", linewidth = 0.35,
            arrow = grid::arrow(length = unit(1.1, "mm"), type = "closed", angle = 25), linejoin = "mitre") +
  geom_polygon(data = poly, aes(x, y, group = id, fill = fill, colour = col, linewidth = lw)) +
  scale_fill_identity() + scale_colour_identity() + scale_linewidth_identity() +
  geom_text(data = texts, aes(x, y, label = label, fontface = face, size = size, colour = col, vjust = vjust),
            family = FONT, lineheight = 1.1) +
  scale_size_identity() +
  coord_fixed(xlim = c(0, 160), ylim = c(0, 78), expand = FALSE, clip = "off") +
  theme_void(base_family = FONT) + theme(plot.margin = margin(1, 0, 3, 0))

# ---- panels b-f: radiograph stages ------------------------------------------------------
img_c01 <- png::readPNG(file.path(ASSET_DIR, "C01.png"))
img_clean <- png::readPNG(file.path(ASSET_DIR, "pilot_clean_page_01.png"))
img_held <- png::readPNG(file.path(ASSET_DIR, "V03.png"))
tile_bilat_raw <- img_c01[56:325, 11:400, 1:3, drop = FALSE]          # same crops as the v1.6 figure script
pad <- ncol(tile_bilat_raw) - nrow(tile_bilat_raw)
tile_bilat <- array(0, dim = c(ncol(tile_bilat_raw), ncol(tile_bilat_raw), 3))
tile_bilat[(floor(pad / 2) + 1):(floor(pad / 2) + nrow(tile_bilat_raw)), , ] <- tile_bilat_raw
tile_before <- img_clean[989:1436, 5:452, 1:3, drop = FALSE]
tile_after <- img_clean[989:1436, 461:908, 1:3, drop = FALSE]
tile_canon <- tile_after[, rev(seq_len(ncol(tile_after))), , drop = FALSE]
tile_held <- img_held[85:532, 17:464, 1:3, drop = FALSE]

ruler_pct <- 100 * to$ruler_cleaned / to$knees
stopifnot(abs(ruler_pct - 18.3) < 0.05)

tile <- function(img, title, subtitle, border = "#2A2F36") {
  ggplot() +
    annotation_raster(img, 0, 1, 0, 1, interpolate = TRUE) +
    annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1, fill = NA, colour = border, linewidth = 0.4) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE, clip = "off") +
    labs(title = title, subtitle = subtitle) +
    theme_void(base_family = FONT) +
    theme(plot.title = element_text(size = 6.4, face = "bold", hjust = 0.5, colour = INK, margin = margin(b = 1)),
          plot.subtitle = element_text(size = 5.6, hjust = 0.5, colour = MUTED, margin = margin(b = 2.5)),
          plot.margin = margin(2, 2.5, 1, 2.5))
}
lab_box <- function(x, y, label, hjust = 0.5)
  annotate("label", x = x, y = y, label = label, hjust = hjust, family = FONT, fontface = "bold", size = pt2mm(5.4),
           colour = "#FFE36E", fill = "#111827CC", linewidth = 0, label.padding = unit(0.12, "lines"))

p_b <- tile(tile_bilat, "Bilateral PA film", "YOLOv2 knee detection") +
  annotate("rect", xmin = 0.05, xmax = 0.1, ymin = 0.895, ymax = 0.945, fill = NA, colour = "#22D3EE", linewidth = 0.45) +
  annotate("text", x = 0.12, y = 0.92, label = "YOLOv2", hjust = 0, family = FONT, size = pt2mm(5.3), colour = "white") +
  annotate("rect", xmin = 0.52, xmax = 0.57, ymin = 0.895, ymax = 0.945, fill = NA, colour = "#FB923C", linewidth = 0.45) +
  annotate("text", x = 0.59, y = 0.92, label = "earlier ref.", hjust = 0, family = FONT, size = pt2mm(5.3), colour = "white") +
  lab_box(0.05, 0.07, "R", 0) + lab_box(0.95, 0.07, "L", 1)
p_c <- tile(tile_before, "1.6× crop, 448 × 448", "red line: ruler-mask edge")
p_d <- tile(tile_after, "Ruler zero-masking", sprintf("applied to %.1f%% of knees", ruler_pct))
p_e <- tile(tile_canon, "Canonical orientation", "left knee flipped") +
  lab_box(0.04, 0.07, "Lateral", 0) + lab_box(0.96, 0.07, "Medial", 1)
p_f <- tile(tile_held, "Automated QC exclusion", "ruler inside 1.05× core zone", border = SPLIT_COL[["Held-out test"]]) +
  annotate("rect", xmin = 0.172, xmax = 0.827, ymin = 0.182, ymax = 0.820, fill = NA, colour = "#FFE36E",
           linetype = "22", linewidth = 0.5) +
  lab_box(0.5, 0.88, "1.05× core zone")

strip <- p_b | p_c | p_d | p_e | p_f
fig <- (p_a / strip) + plot_layout(heights = c(78, 36)) + plot_annotation(tag_levels = "a") & tag_theme
save_fig(fig, "fig2_cohort_preprocessing", h_mm = 123)
