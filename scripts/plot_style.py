'''Shared figure style and output folder for the scripts.'''

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT/'figures'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.grid': True,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'legend.frameon': False,
})

BLUE, RED, GREY = '#1f4e79', '#c00000', '#7f7f7f'


def save(fig, name):
    '''Save as PDF (for LaTeX) and PNG (for quick viewing) in figures/.'''
    for ax in fig.axes:
        if ax.get_xscale() == 'log':
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(FIG_DIR/f'{name}.pdf')
    fig.savefig(FIG_DIR/f'{name}.png', dpi=150)
    plt.close(fig)
