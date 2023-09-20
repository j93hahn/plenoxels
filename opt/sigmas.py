import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap
from util import config_util
import seaborn as sns
import natsort
from fabric.utils.event import read_stats
from scratch.illustrate.plotter import make_plot
from scratch.algos.grid import sample_grid_box
from scratch.algos.rays import integrate_weights, compute_cdf, extract_counts
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes, InsetPosition
from matplotlib.ticker import MultipleLocator

import imageio
import cv2
import gc
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# experiment folder mapping for common conventions
mapping = {
    '0.3': '3e-1',
    '0.003': '3e-3',
    '30.0': '3e1',
    '0.0005': '5e-4',
    '0.05': '5e-2',
    '5e-06': '5e-6'
}


# grouping by color and differentiating by linestyle
color = {
    '3e1': 'red',
    '3e-1': 'blue',
    '3e-3': 'green',
    '0': '-',
    '15000': '-.'
}


# inspiration for particular font taken from GroupNorm paper
plt.rcParams["font.family"] = "STIXGeneral"


# iterate through all scenes
for data in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship', \
             'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']:
    if data != 'trex':
        continue
    print(f"working on data {data}")

    # generate figure
    fig, ax = plt.subplots()
    ax.set_xlabel("σ-values (log scale)", fontsize=10.0)
    ax.set_ylabel("cumulative distribution function (CDF)", fontsize=10.0)
    skipped = False

    # generate and mark inset axes - https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo2.html#sphx-glr-gallery-axes-grid1-inset-locator-demo2-py
    axins = inset_axes(ax, width="100%", height="100%", loc="upper left",
                       bbox_to_anchor=(0.45, 0.2, 0.5, 0.4),    # <------ SPECIFY
                       bbox_transform=ax.transAxes)
    x1, x2, y1, y2 = 1e0, 1e3 + 250, 0.94, 1.01                      # <------ SPECIFY
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", alpha=0.7)

    # iterate through all folders in the current directory
    for folder in natsort.natsorted(glob.glob('*')):
        components = folder.split('_')
        if len(components) > 1:
            if components[1] == data:
                if not os.path.isfile(f"{folder}/ckpt/ckpt.npz"):
                    # skip if ckpt.npz does not exist
                    skipped = True
                    continue

                # load the model and sample sigmas at uniformly distributed xyz locations
                grid = svox2.SparseGrid.load(f"{folder}/ckpt/ckpt.npz", device=device)
                num = 200
                pts = sample_grid_box(torch.tensor([num, num, num]), device)
                sigma, _ = grid.sample(pts, use_kernel=True, grid_coords=False, want_colors=False)
                sigmas = sigma.detach().cpu().squeeze()
                sigmas[sigmas < 1e-2] = 1e-2    # clamp to 1e-2 to avoid log(0) or log(negative)

                # compute the CDF
                cdf, bins = compute_cdf(sigmas, 1000)
                cdf = torch.cat([torch.tensor([0.0]), cdf])

                # extract the decay steps, initial learning rate, and final learning rate for the label
                decay_steps = components[2]
                init_lr = mapping[components[3]]
                final_lr = mapping[components[4]]

                # plot the CDF
                ax.plot(bins, cdf, label=f"params: [{decay_steps}, {init_lr}, {final_lr}]", linestyle=color[decay_steps], color=color[init_lr], linewidth=1.0)
                axins.plot(bins, cdf, linestyle=color[decay_steps], color=color[init_lr], linewidth=1.0)

                # free memory and clear cache
                del grid, pts, sigma, sigmas, cdf, bins
                gc.collect()
                torch.cuda.empty_cache()

    # log scale on x-axis and turn off minor ticks
    ax.set_xscale('log')
    ax.minorticks_off()
    axins.set_xscale('log')
    axins.minorticks_off()
    axins.plot([39.1516, 61.0363], [0.9801, 0.9801], linestyle='-', color='black', linewidth=0.5)
    axins.text(70, 0.975, "σ-diff: 21.8847", fontsize=7.0)

    # set up plot aesthetics - get rid of black bounding box enclosing the plot
    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')
    axins.spines[['right', 'top']].set_visible(True)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')

    # set up black ticks pointing inwards at axes locations and gridlines just for the y-axis
    ax.tick_params(direction='in', axis='both', width=0.5, color='black', length=4, labelsize=10.0)
    ax.yaxis.grid(True, which='both', linewidth=0.5, color='lightgray')
    axins.tick_params(direction='in', axis='both', width=0.5, color='black', length=4, labelsize=7.0)
    axins.yaxis.grid(True, which='both', linewidth=0.5, color='lightgray')

    # set up legend ordering here
    handles, labels = ax.get_legend_handles_labels()
    if not skipped:     # all models are present and available
        order = [2, 0, 1, 5, 3, 4]
    else:               # skip the model that is missing
        order = [2, 0, 1, 4, 3]

    # move legend to bottom of plot and outside of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=2, prop={'size': 6})

    # save plot and close figure
    fig.savefig(f"../plots/{data}_sigma.png", dpi=300)
    plt.close(fig)

    if data == 'trex':
        exit()


"""
0, 3e1, 5e-2

at 0.9801, bins = 61.0363

15000, 3e1, 5e-2

at 0.9801, bins = 39.1516

diff = 21.8847

"""
