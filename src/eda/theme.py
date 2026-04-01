import matplotlib.pyplot as plt
import yaml
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STYLE_YAML = os.path.join(PROJECT_ROOT, "configs", "style.yaml")

def set_theme():
    """Load colors from yaml and apply Meta / Google Research figure style to matplotlib."""
    with open(STYLE_YAML, "r") as f:
        colors = yaml.safe_load(f)["colors"]

    plt.rcParams.update({
        # Typography
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'medium',        # not bold — papers use medium weight
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,

        # Spines — thin and gray, not black
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Grid — very subtle or off
        'axes.grid': True,
        'grid.color': '#E5E5E5',
        'grid.alpha': 0.7,
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',                # solid, not dashed — cleaner look
        'axes.axisbelow': True,               # grid behind data

        # Ticks — thin, inward, minimal
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.pad': 4,
        'ytick.major.pad': 4,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Figure
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,           # tight crop like paper figures

        # Lines and patches
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'patch.edgecolor': 'none',

        # Legend
        'legend.frameon': False,              # no legend box — standard in papers
        'legend.borderpad': 0.3,
    })
    
    return colors
