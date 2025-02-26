import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt

plt.rcParams.update({
    'xtick.labelsize': 8,
    'xtick.major.size': 3,
    'ytick.labelsize': 8,
    'ytick.major.size': 3,
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.labelpad': 0.,
    'legend.fontsize': 7,
    'figure.subplot.wspace': 0.4,
    'figure.subplot.hspace': 0.4,
    'figure.subplot.left': 0.1,
    'figure.dpi': 600,
    'figure.figsize':  [6.0, 2.5],
    'lines.linewidth': 1,
    'lines.color': 'k',
})

elec_color = '#00d2ff'
syn_color = '#00cc00'
cell_color = '#c4c4c4'


def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=-0.12, ypos=1.15):

    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.text(xpos, ypos, letters[idx].capitalize(),
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='demibold',
                fontsize=9,
                transform=ax.transAxes)

def simplify_axes(axes):

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

def color_axes(axes, clr):
    if not type(axes) is list:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis='x', colors=clr)
        ax.tick_params(axis='y', colors=clr)
        for spine in list(ax.spines.values()):
            spine.set_edgecolor(clr)

