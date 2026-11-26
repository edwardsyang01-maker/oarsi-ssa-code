# v2.4 manuscript figures and tables: shared settings.
# One visual system for all seven figures: 160 mm width (A4 text width of the manuscript),
# 7 pt Arial-metric text (Liberation Sans on the server), one palette per meaning.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(grid)
  library(png)
  library(ragg)
  library(scales)
})

get_script_dir <- function() {
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
}
SCRIPT_DIR <- get_script_dir()
BASE_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."))
DATA_DIR <- file.path(BASE_DIR, "data")
ASSET_DIR <- file.path(BASE_DIR, "assets")
FIG_DIR <- file.path(BASE_DIR, "outputs", "figures")
TAB_DIR <- file.path(BASE_DIR, "outputs", "tables")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

FONT <- "Arial"
FIG_W <- 160          # mm, equals the text width of the A4 manuscript (25 mm margins)
BASE_PT <- 7
pt2mm <- function(pt) pt / ggplot2::.pt   # geom_text size from point size

read_json_data <- function(f) jsonlite::fromJSON(file.path(DATA_DIR, f), simplifyVector = FALSE)
num <- function(x) as.numeric(unlist(x))

# ---- palette (one meaning, one colour) ------------------------------------------------
INK <- "#1B2330"
MUTED <- "#5E6875"
FAINT <- "#9AA3AE"
GRID <- "#E8EBF0"
REF <- "#2F3743"          # OAI reader re-test reference lines

ARM_COL <- c(P0 = "#A3ABB5", P1 = "#86AFD8", P2 = "#4A6582", P3 = "#D1603D", P4 = "#2C9C80")
ARM_FILL_LIGHT <- c(P0 = "#F1F3F5", P1 = "#EAF2FA", P2 = "#E8EDF3", P3 = "#FBEAE3", P4 = "#E3F4EF")
SPLIT_COL <- c("Training" = "#3D6FA6", "Validation" = "#3E9E96", "Held-out test" = "#E0973A")
SPLIT_FILL_LIGHT <- c("Training" = "#ECF2F9", "Validation" = "#E8F5F3", "Held-out test" = "#FDF3E5")
SIDE_COL <- c(Medial = "#B8506E", Lateral = "#3C7CBE", KL = "#454E5B")
SEED_SHAPE <- c("Seed 17" = 21, "Seed 23" = 22, "Seed 29" = 24)
SEEDS <- c("17", "23", "29")

# ---- OARSI sites: one naming scheme used in every figure and table --------------------
SITES <- c("osfm", "osfl", "ostm", "ostl", "jsm", "jsl")
SITE_LAB <- c(osfm = "Femoral OST, medial", osfl = "Femoral OST, lateral",
              ostm = "Tibial OST, medial", ostl = "Tibial OST, lateral",
              jsm = "JSN, medial", jsl = "JSN, lateral")
SITE_LAB2 <- c(osfm = "Femoral OST\nmedial", osfl = "Femoral OST\nlateral",
               ostm = "Tibial OST\nmedial", ostl = "Tibial OST\nlateral",
               jsm = "JSN\nmedial", jsl = "JSN\nlateral")
SITE_SIDE <- c(osfm = "Medial", osfl = "Lateral", ostm = "Medial", ostl = "Lateral", jsm = "Medial", jsl = "Lateral")

fmt_int <- function(x) formatC(x, format = "d", big.mark = ",")
fmt_p <- function(p) ifelse(p < 0.001, "P < 0.001", sprintf("P = %.3f", p))
fmt_signed <- function(x, d = 3) {
  s <- formatC(abs(x), format = "f", digits = d)
  paste0(ifelse(x < 0, "−", "+"), s)
}
minus <- function(s) gsub("-", "−", s, fixed = TRUE)

theme_pub <- function(base = BASE_PT) {
  theme_classic(base_size = base, base_family = FONT) +
    theme(
      text = element_text(colour = INK),
      axis.line = element_line(linewidth = 0.3, colour = INK),
      axis.ticks = element_line(linewidth = 0.3, colour = INK),
      axis.ticks.length = unit(1.5, "pt"),
      axis.text = element_text(size = base - 0.5, colour = INK),
      axis.title = element_text(size = base, colour = INK),
      plot.title = element_text(size = base, face = "bold", hjust = 0, margin = margin(b = 4)),
      plot.subtitle = element_text(size = base - 0.5, colour = MUTED, hjust = 0, margin = margin(b = 3)),
      legend.title = element_text(size = base - 0.5, face = "bold"),
      legend.text = element_text(size = base - 0.5),
      legend.key.size = unit(3.2, "mm"),
      legend.background = element_blank(),
      panel.grid.major.y = element_line(colour = GRID, linewidth = 0.25),
      strip.background = element_blank(),
      strip.text = element_text(size = base - 0.3, face = "bold"),
      plot.tag = element_text(size = 9, face = "bold", family = FONT, colour = "black"),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(3, 4, 3, 3)
    )
}
theme_set(theme_pub())

tag_theme <- theme(plot.tag = element_text(size = 9, face = "bold", family = FONT, colour = "black",
                                          margin = margin(r = 1.5, b = 1.5)),
                   plot.tag.position = "topleft", plot.tag.location = "margin")

# Rounded rectangle as polygon (coordinates in mm; use with coord_fixed())
rrect <- function(xmin, xmax, ymin, ymax, r = 1.2, n = 10, id = 1) {
  r <- min(r, (xmax - xmin) / 2, (ymax - ymin) / 2)
  arc <- function(cx, cy, a0, a1) {
    t <- seq(a0, a1, length.out = n)
    data.frame(x = cx + r * cos(t), y = cy + r * sin(t))
  }
  d <- rbind(arc(xmax - r, ymax - r, 0, pi / 2), arc(xmin + r, ymax - r, pi / 2, pi),
             arc(xmin + r, ymin + r, pi, 3 * pi / 2), arc(xmax - r, ymin + r, 3 * pi / 2, 2 * pi))
  d$id <- id
  d
}
boxes_to_poly <- function(b, r = 1.2) {
  do.call(rbind, lapply(seq_len(nrow(b)), function(i) {
    p <- rrect(b$xmin[i], b$xmax[i], b$ymin[i], b$ymax[i], r = if (!is.null(b$r)) b$r[i] else r, id = i)
    p$fill <- b$fill[i]; p$col <- b$col[i]; p$lw <- b$lw[i]; p$lt <- if (!is.null(b$lt)) b$lt[i] else "solid"
    p
  }))
}

save_fig <- function(p, name, h_mm, w_mm = FIG_W, dpi = 500) {
  png_path <- file.path(FIG_DIR, paste0(name, ".png"))
  ragg::agg_png(png_path, width = w_mm, height = h_mm, units = "mm", res = dpi, background = "white")
  print(p)
  invisible(dev.off())
  grDevices::cairo_pdf(file.path(FIG_DIR, paste0(name, ".pdf")), width = w_mm / 25.4, height = h_mm / 25.4,
                       family = FONT, bg = "white")
  print(p)
  invisible(dev.off())
  message(sprintf("saved %s (%d x %d mm)", name, round(w_mm), round(h_mm)))
  invisible(png_path)
}

stopifnot_close <- function(got, want, tol, what) {
  if (any(abs(got - want) > tol)) stop(sprintf("check failed: %s (got %s, want %s)", what,
                                               paste(signif(got, 6), collapse = ","), paste(want, collapse = ",")))
}
