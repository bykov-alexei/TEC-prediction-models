from .functions import plot_mercater_map
import matplotlib.pyplot as plt


def plot(filename, title, map, max_scale):

    size = 6
    plt.rcParams['figure.figsize'] = (size, size*0.9)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['figure.dpi'] = 300

    props = dict(boxstyle='round', facecolor='white', alpha=1)

    plot_mercater_map(map, 0, max_scale)
    plt.tight_layout()
    plt.title(str(title))
    plt.savefig(filename)
    plt.clf()
