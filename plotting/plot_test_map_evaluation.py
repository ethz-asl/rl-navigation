import sys
sys.path.insert(0, "/usr/local/lib/python2.7/dist-packages")

import cPickle as pickle
import pylab as pl
import os
import numpy as np
import pylab as pl
import pandas as pd

# ROS imports
import rospkg

pl.close('all')

def convert_to_float(input):
  output = np.zeros_like(input, dtype=float)
  for ii in range(len(input)):
    output[ii] = float(input[ii])
  return output


# Figure setup
fig_width_pt = 650                # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
# golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*0.3      # height in inches
fig_size =  [fig_width,fig_height]
fig = pl.figure("Test map evaluation",figsize=(fig_width, fig_height))
fontsize = 8
params = {"backend": "ps",
          "axes.labelsize": fontsize,
          "font.size": fontsize,
        #   "title.fontsize": fontsize,
        #   "legend.fontsize": fontsize,
          "xtick.labelsize": fontsize,
          "ytick.labelsize": fontsize,
          "text.usetex": True,
          "font.family": "serif",
          "font.serif": "Computer Modern Roman",
          "figure.figsize": fig_size}
pl.rcParams.update(params)

data_path = "data/test_map_eval_with_info_new.csv"

data = pd.read_csv(data_path)
data = data.drop('rates', axis=1)
model_names = data.columns.values

success_fract = data.loc[0]
timeout_fract = data.loc[1]
crash_fract = data.loc[2]
try:
  il_setup = data.loc[3]
  rl_setup = data.loc[4]
  reward_setup = data.loc[5]
except:
  pass

bar_width = 1.0
dist_bars = 3.0
num_models = len(model_names)
x_pos = np.arange(1, dist_bars*num_models, dist_bars)

stacked_ticks = []
for ii in range(len(il_setup)):
  stacked_ticks.append(il_setup[ii] + "\n" + rl_setup[ii] + "\n" + reward_setup[ii])

# Prepare axis
ax = fig.gca()
ax.grid(color='k', linestyle=':', linewidth=0.2)
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Separate the different classes of models
height = 102
background_alpha = 0.15
x1 = 8*dist_bars + (dist_bars-bar_width) / 2.0 - 1.0
x2 = 10*dist_bars + (dist_bars-bar_width) / 2.0 - 1.0
x3 = 12*dist_bars + (dist_bars-bar_width) / 2.0 - 1.0
x4 = 16*dist_bars + (dist_bars-bar_width) / 2.0 - 1.0
x5 = 18*dist_bars + (dist_bars-bar_width) / 2.0 - 1.0
linewidth = 2
pl.axvline(x1, ymin=0, ymax=1.07, color='k', clip_on=False, lw=linewidth)
pl.axvline(x2, ymin=0, ymax=1.07, color='k', clip_on=False, lw=linewidth)
pl.axvline(x3, ymin=0, ymax=1.07, color='k', clip_on=False, lw=linewidth)
pl.axvline(x4, ymin=0, ymax=1.07, color='k', clip_on=False, lw=linewidth)
ax.text(x1/2.0, 103, "R-IL", ha='center', fontsize=8)
ax.text((x2-x1)/2. + x1, 103, "R-IL$_{200}$", ha='center', fontsize=8)
ax.text((x3-x2)/2. + x2, 103, "pure IL", ha='center', fontsize=8)
ax.text((x4-x3)/2. + x3, 103, "pure RL", ha='center', fontsize=8)
ax.text((x5-x4)/2. + x4, 103, "comp.", ha='center', fontsize=8)

# patch1 = pl.matplotlib.patches.Rectangle([x1,0], width=x2-x1, height=height, facecolor='c', alpha=background_alpha, edgecolor=None, linewidth=0)
# ax.add_artist(patch1)
# patch2 = pl.matplotlib.patches.Rectangle([x2,0], width=10, height=height, facecolor='m', alpha=background_alpha, edgecolor=None, linewidth=0)
# ax.add_artist(patch2)

success_matrix = convert_to_float(success_fract.as_matrix())
timeout_matrix = convert_to_float(timeout_fract.as_matrix())
crash_matrix = convert_to_float(crash_fract.as_matrix())

success_bars = pl.bar(x_pos, success_matrix, color='darkgreen', width=bar_width, alpha=1)
timeout_bars = pl.bar(x_pos, timeout_matrix, bottom=success_matrix, color='orange', width=bar_width)
crash_bars = pl.bar(x_pos, crash_matrix, bottom=success_matrix + timeout_matrix, color='red', width=bar_width)

# Adjust axis
offset_border = 2.5
ax.set_xlim([0.0, max(x_pos)+offset_border])
ax.set_ylim([0, 100])
ticks_array = x_pos + bar_width / dist_bars

pl.xticks(ticks_array, stacked_ticks, fontsize=6, rotation=0)
ax.text(max(ticks_array) + offset_border, -7.5, "IL", ha='left', fontsize=6)
ax.text(max(ticks_array) + offset_border, -15, "RL", ha='left', fontsize=6)
ax.text(max(ticks_array) + offset_border, -23, "reward", ha='left', fontsize=6)
pl.axvline(max(ticks_array) + offset_border-0.32, ymin=-0.25, ymax=0, color='k', clip_on=False, lw=0.5)
# ax.set_xticks(x_pos + bar_width / dist_bars)
# ax.set_xticklabels(model_names, rotation=40)
ax.set_ylabel("ratio in [\%]")
ax.legend((success_bars[0], timeout_bars[0], crash_bars[0]), ("success", "timeout", "crash"), ncol=1, fancybox=True, framealpha=0.5,
          bbox_to_anchor=(0.15, 1.7))


# pl.tight_layout()
pl.gcf().subplots_adjust(bottom=0.17, top=0.65, left=0.065, right=0.95, hspace=0.12)
# data.plot.bar(stacked=True)


pl.show(block=False)

fig.savefig("figures/test_map_eval.pdf")
