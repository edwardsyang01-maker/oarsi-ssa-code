#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROWS = (('seed17_P3', 'P3, seed 17'), ('seed23_P3', 'P3, seed 23'), ('seed29_P3', 'P3, seed 29'),
        ('seed17_P4', 'P4, seed 17'), ('seed17_P1', 'P1, seed 17'))
COLUMNS = (('kl', 'KL query', ''), ('osfm', 'OSFM', 'femur, medial'), ('osfl', 'OSFL', 'femur, lateral'),
           ('ostm', 'OSTM', 'tibia, medial'), ('ostl', 'OSTL', 'tibia, lateral'),
           ('jsm', 'JSM', 'narrowing, medial'), ('jsl', 'JSL', 'narrowing, lateral'))
RAMP = ['#f6f9fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
INK, MUTED = '#1f1f1d', '#6b6b66'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, help='attention_summary.json')
    parser.add_argument('--out', required=True, help='output directory')
    args = parser.parse_args()
    out = Path(args.out)
    for name in ('attention_mean_maps.png', 'attention_mean_maps.pdf'):
        assert not (out / name).exists(), f'output already exists; refusing to overwrite: {name}'
    runs = json.load(open(args.summary, encoding='utf-8'))['runs']
    maps = {(r, q): np.array(m) * 784 for r in runs for q, m in runs[r]['mean_attention_map_28x28'].items()}
    vmax_kl = max(v.max() for (r, q), v in maps.items() if q == 'kl')
    vmax_site = max(v.max() for (r, q), v in maps.items() if q != 'kl')
    cmap = LinearSegmentedColormap.from_list('blue_seq', RAMP)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
    fig = plt.figure(figsize=(10.2, 7.4), dpi=200)
    grid = fig.add_gridspec(len(ROWS), len(COLUMNS), left=0.105, right=0.985, top=0.9, bottom=0.115, wspace=0.07, hspace=0.07)
    images = {}
    for i, (run, row_label) in enumerate(ROWS):
        for j, (query, title, subtitle) in enumerate(COLUMNS):
            if (run, query) not in maps:
                continue
            ax = fig.add_subplot(grid[i, j])
            kind = 'kl' if query == 'kl' else 'site'
            images[kind] = ax.imshow(maps[(run, query)], cmap=cmap, vmin=0, vmax=vmax_kl if kind == 'kl' else vmax_site,
                                     interpolation='nearest')
            ax.axvline(13.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#c9c9c4'); spine.set_linewidth(0.6)
            if i == 0:
                ax.set_title(title + (f'\n{subtitle}' if subtitle else '\n'), fontsize=8, color=INK, pad=4, linespacing=1.3)
            if j == 0:
                ax.set_ylabel(f'{row_label}\nepoch {runs[run]["best_epoch"]}', fontsize=8, color=INK, rotation=0,
                              ha='right', va='center', labelpad=8, linespacing=1.4)
            if i == len(ROWS) - 2 and j > 0 or i == len(ROWS) - 1 and j == 0:
                ax.set_xlabel('lateral  |  medial', fontsize=7, color=MUTED, labelpad=3)
    for kind, left, width, label in (('kl', 0.105, 0.115, 'KL query: attention, multiples of uniform'),
                                     ('site', 0.36, 0.5, 'Site queries: attention, multiples of uniform (shared scale)')):
        cax = fig.add_axes([left, 0.05, width, 0.014])
        bar = fig.colorbar(images[kind], cax=cax, orientation='horizontal')
        bar.outline.set_linewidth(0.4); bar.outline.set_edgecolor('#c9c9c4')
        bar.ax.tick_params(labelsize=7, colors=MUTED, length=2, width=0.4)
        bar.set_label(label, fontsize=7, color=MUTED, labelpad=3)
    fig.suptitle('Mean attention of each query over the fixed 32-image validation panel (28 × 28 patches)',
                 fontsize=9.5, color=INK, x=0.105, ha='left', y=0.975)
    fig.savefig(out / 'attention_mean_maps.png', dpi=200, facecolor='white')
    fig.savefig(out / 'attention_mean_maps.pdf', facecolor='white')
    print(json.dumps({'vmax_kl': float(vmax_kl), 'vmax_site': float(vmax_site)}))


if __name__ == '__main__':
    main()
