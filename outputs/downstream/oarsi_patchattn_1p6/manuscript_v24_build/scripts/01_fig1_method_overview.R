#!/usr/bin/env Rscript
# Figure 1 (v2.4). (a) P3 architecture and training objective; (b) 2 x 2 factorial arms plus P4.
# Content follows manuscript sections 2.4-2.5: DINOv3 ViT-L/16, blocks 1-18 frozen, 19-24 + final LayerNorm
# fine-tuned, 7 learnable queries, single-layer 8-head cross-attention with learnable 2D positional embedding,
# KL head = CLS + zero-initialised KL-query branch, six 4-class OARSI heads, L = L_KL + lambda(e) L_OARSI.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))), "00_common.R"))

NEW_FILL <- "#FCF1EB"; NEW_COL <- ARM_COL[["P3"]]
FRZ_FILL <- "#E9ECF0"; FRZ_COL <- "#8C95A1"
TRN_FILL <- "#FFFFFF"; TRN_COL <- INK
CON_FILL <- "#F6F7F9"; CON_COL <- "#B3BAC4"

bx <- function(xmin, xmax, ymin, ymax, fill, col, lw = 0.35, lt = "solid", r = 1.3)
  data.frame(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill, col = col, lw = lw, lt = lt, r = r)
tx <- function(x, y, label, pt = 6, face = "plain", col = INK, hjust = 0.5, vjust = 0.5)
  data.frame(x = x, y = y, label = label, size = pt2mm(pt), face = face, col = col, hjust = hjust, vjust = vjust)
route <- function(id, x, y, lt = "solid") data.frame(id = id, x = x, y = y, lt = lt)

draw_schematic <- function(boxes, texts, routes, xlim, ylim, rasters = list()) {
  poly <- boxes_to_poly(boxes)
  g <- ggplot() +
    geom_polygon(data = poly, aes(x, y, group = id, fill = fill, colour = col, linewidth = lw, linetype = lt)) +
    scale_fill_identity() + scale_colour_identity() + scale_linewidth_identity() + scale_linetype_identity()
  for (r in rasters) g <- g + annotation_raster(r$img, r$xmin, r$xmax, r$ymin, r$ymax, interpolate = TRUE)
  if (nrow(routes)) {
    g <- g + geom_path(data = routes, aes(x, y, group = id, linetype = lt), colour = "#3A4350", linewidth = 0.35,
                       arrow = grid::arrow(length = unit(1.2, "mm"), type = "closed", angle = 25),
                       lineend = "butt", linejoin = "mitre")
  }
  g + geom_text(data = texts, aes(x, y, label = label, size = size, fontface = face, colour = col, hjust = hjust, vjust = vjust),
                family = FONT, lineheight = 1.08) +
    scale_size_identity() +
    coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE, clip = "off") +
    theme_void(base_family = FONT) +
    theme(plot.margin = margin(1, 1, 1, 1), plot.background = element_rect(fill = "white", colour = NA))
}

# ---------------------------------------------------------------- panel a
knee <- png::readPNG(file.path(ASSET_DIR, "mean_val_knee_448.png"))
if (length(dim(knee)) == 3) knee <- knee[, , 1]

ba <- rbind(
  # containers
  bx(1, 29, 12, 64, TRN_FILL, TRN_COL),                    # input
  bx(33.5, 67.5, 12, 64, CON_FILL, CON_COL),              # backbone container
  bx(71.5, 114.5, 12, 64, "#FFF8F4", NEW_COL, lw = 0.5),  # pooling container (new)
  # backbone blocks
  bx(37, 64, 41, 53, FRZ_FILL, FRZ_COL),
  bx(37, 64, 18, 35, TRN_FILL, TRN_COL),
  # cross-attention
  bx(75, 111, 15, 46, TRN_FILL, NEW_COL, lw = 0.45),
  # heads and loss
  bx(118.5, 156.5, 50, 63, TRN_FILL, TRN_COL),
  bx(118.5, 156.5, 30, 44, NEW_FILL, NEW_COL, lw = 0.45),
  bx(118.5, 156.5, 8, 27.5, CON_FILL, TRN_COL),
  # supervision source
  bx(1, 29, 0.8, 8.8, CON_FILL, CON_COL),
  # legend keys
  bx(2, 5, 71.1, 73.9, NEW_FILL, NEW_COL, lw = 0.45, r = 0.6),
  bx(37, 40, 71.1, 73.9, TRN_FILL, TRN_COL, r = 0.6),
  bx(57, 60, 71.1, 73.9, FRZ_FILL, FRZ_COL, r = 0.6)
)
# query chips: KL, osfm, osfl, ostm, ostl, jsm, jsl
chip_col <- c(SIDE_COL[["KL"]], rep(NEW_COL, 6))
chip_x0 <- 93 - (7 * 3.4 + 6 * 1.1) / 2 + (0:6) * 4.5
chips <- do.call(rbind, lapply(1:7, function(i) bx(chip_x0[i], chip_x0[i] + 3.4, 50.6, 54.0, chip_col[i], chip_col[i], lw = 0.2, r = 0.5)))
ba <- rbind(ba, chips)

ta <- rbind(
  tx(15, 60.6, "Input knee image", 6.5, "bold"),
  tx(5.2, 32.2, "Lateral", 5.4, col = MUTED), tx(24.8, 32.2, "Medial", 5.4, col = MUTED),
  tx(15, 22.3, "448 × 448 single-knee crop\ncanonical right-knee view\ntraining: random horizontal\nflip (p = 0.5)", 5.8),
  tx(50.5, 60.6, "DINOv3 ViT-L/16", 6.5, "bold"),
  tx(50.5, 57.0, "self-supervised pretrained", 5.6, col = MUTED),
  tx(50.5, 47, "Patch embedding +\nblocks 1–18 (frozen)", 6),
  tx(50.5, 26.5, "Blocks 19–24 +\nfinal LayerNorm\n(fine-tuned)", 6),
  tx(93, 60.6, "Site-query attention pooling", 6.5, "bold"),
  tx(93, 57.0, "7 learnable queries: 1 KL + 6 OARSI sites", 5.6, col = MUTED),
  tx(93, 25.5, "Single-layer cross-attention\n(8 heads)\nkeys: patch tokens + learnable\n2D positional embedding\nvalues: patch tokens", 6),
  tx(110.2, 42.6, "KL query", 5.4, col = MUTED, hjust = 1),
  tx(110.2, 38.1, "site queries", 5.4, col = MUTED, hjust = 1),
  tx(137.5, 56.5, "KL head (grades 0–4)\nCLS token + KL-query output\n(KL-query projection zero-initialized)", 5.8),
  tx(137.5, 37, "Six OARSI site heads (grades 0–3)\none linear head per site query\nfemoral/tibial OST and JSN, M/L", 5.8),
  tx(137.5, 25.2, "Training loss", 6, "bold"),
  tx(137.5, 14.4, "KL loss: class-weighted CE + 0.5 × RPS\nOARSI loss: CE over available grades\nλ: OARSI/KL gradient-norm ratio 1/4,\nlinearly warmed up over 3 epochs", 5.5),
  tx(15, 4.8, "Central OAI readings", 6, "bold"),
  tx(80, 6.3, "KL 0–4 and six OARSI grades 0–3; M/L OARSI labels swapped on flip (P2, P3); missing grades masked",
     5.6, col = MUTED),
  tx(50.5, 15.0, "outputs: CLS + 784 patch tokens", 5.3, col = MUTED),
  tx(102, 69.0, "CLS token", 5.6, col = MUTED),
  # legend
  tx(6, 72.5, "Added to the backbone", 5.8, hjust = 0), tx(41, 72.5, "Fine-tuned", 5.8, hjust = 0),
  tx(61, 72.5, "Frozen", 5.8, hjust = 0), tx(79, 72.5, "Supervision (training only)", 5.8, hjust = 0)
)

ra <- rbind(
  route(1, c(29, 37), c(47, 47)),                                  # input -> frozen
  route(2, c(50.5, 50.5), c(41, 35)),                               # frozen -> fine-tuned
  route(3, c(64, 75), c(24, 24)),                                   # patch tokens -> cross-attention
  route(4, c(64, 69.6, 69.6, 137.5, 137.5), c(32.5, 32.5, 67.5, 67.5, 63)),   # CLS -> KL head
  route(5, c(93, 93), c(50.6, 46)),                                 # queries -> cross-attention
  route(6, c(111, 116.8, 116.8, 118.5), c(42.6, 42.6, 54.5, 54.5)),   # KL query -> KL head
  route(7, c(111, 118.5), c(38.1, 38.1)),                             # site queries -> OARSI heads
  route(8, c(137.5, 137.5), c(30, 27.5)),                                 # OARSI heads -> loss
  route(9, c(156.5, 158.3, 158.3, 156.5), c(56.5, 56.5, 17.75, 17.75)), # KL head -> loss
  route(10, c(29, 100, 100, 120), c(4.8, 4.8, 4.8, 4.8), lt = "22"),# supervision -> loss
  route(11, c(71.5, 77), c(72.5, 72.5), lt = "22")                  # legend key
)
ra <- ra[!(ra$id == 10), ]
ra <- rbind(ra, route(10, c(29, 137.5, 137.5), c(4.6, 4.6, 8), lt = "22"))
p_a <- draw_schematic(ba, ta, ra, xlim = c(0, 160), ylim = c(0, 75),
                      rasters = list(list(img = knee, xmin = 5.2, xmax = 24.8, ymin = 34.2, ymax = 53.8)))
p_a <- p_a + annotate("rect", xmin = 5.2, xmax = 24.8, ymin = 34.2, ymax = 53.8, fill = NA, colour = "#C9CED6", linewidth = 0.25) +
  annotate("segment", x = 15, xend = 15, y = 34.2, yend = 53.8, colour = "white", linewidth = 0.25, linetype = "22") +
  annotate("text", x = 137.5, y = 21.6, family = FONT, size = pt2mm(6.2), colour = INK, parse = TRUE,
           label = "italic(L) == italic(L)[KL] + lambda(italic(e)) %.% italic(L)[OARSI]")

# ---------------------------------------------------------------- panel b
cell <- function(arm, xmin, xmax, ymin, ymax, text, main = FALSE) {
  list(
    box = rbind(bx(xmin, xmax, ymin, ymax, ARM_FILL_LIGHT[[arm]], if (main) ARM_COL[[arm]] else "#C3C9D1", lw = if (main) 0.55 else 0.3),
                bx(xmin + 1.6, xmin + 8.6, (ymin + ymax) / 2 - 2.4, (ymin + ymax) / 2 + 2.4, ARM_COL[[arm]], ARM_COL[[arm]], lw = 0.2, r = 0.8)),
    text = rbind(tx(xmin + 5.1, (ymin + ymax) / 2, arm, 6.8, "bold", col = "white"),
                 tx(xmin + 10.6, (ymin + ymax) / 2, text, 5.9, hjust = 0))
  )
}
cells <- list(
  cell("P0", 23, 65, 19, 32, "Baseline\nKL head on global CLS token"),
  cell("P1", 67, 109, 19, 32, "1 KL query + cross-attention\nKL head on CLS + KL query\nno OARSI supervision"),
  cell("P2", 23, 65, 3, 16, "6 OARSI heads on CLS token\nKL head on CLS token\nλ = 0.167"),
  cell("P3", 67, 109, 3, 16, "Full model: 7 queries\n(1 KL + 6 sites); OARSI heads\non site queries; λ = 0.295", main = TRUE),
  cell("P4", 116, 159, 3, 16, "Laterality control: identical to P3,\nbut M/L OARSI labels are not\nswapped on horizontal flip")
)
bb <- do.call(rbind, lapply(cells, `[[`, "box"))
tb <- do.call(rbind, lapply(cells, `[[`, "text"))
bb <- rbind(bb, bx(116, 159, 19, 32, "#FFFFFF", "#D5DAE0", lw = 0.3))
tb <- rbind(tb,
  tx(44, 35.6, "No attention pooling", 6.3, "bold"),
  tx(88, 35.6, "Attention pooling", 6.3, "bold"),
  tx(137.5, 35.6, "Effects and laterality control", 6.3, "bold"),
  tx(21, 25.5, "KL supervision\nonly", 6.3, "bold", hjust = 1),
  tx(21, 9.5, "KL + OARSI site\nsupervision", 6.3, "bold", hjust = 1),
  tx(118.5, 25.5, "Factorial effects on KL (Table 3)\npooling = [(P1 − P0) + (P3 − P2)] / 2\nOARSI = [(P2 − P0) + (P3 − P1)] / 2\ninteraction = P3 − P2 − P1 + P0", 5.6, hjust = 0)
)
rb <- route(1, c(109, 116), c(9.5, 9.5), lt = "22")
p_b <- draw_schematic(bb, tb, rb, xlim = c(0, 160), ylim = c(0, 38))

fig <- (p_a / p_b) + plot_layout(heights = c(75, 38)) +
  plot_annotation(tag_levels = "a") & tag_theme
save_fig(fig, "fig1_method_overview", h_mm = 121)
