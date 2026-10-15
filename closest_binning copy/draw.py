import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
# ==========================================
# GLOBAL STYLE CONFIGURATION
# ==========================================
FIG_SIZE = (10, 5)
FONT_SIZE_LABEL = 30
FONT_SIZE_TICKS = 22
FONT_SIZE_LEGEND = 22
LINE_WIDTH = 5
MARKER_SIZE = 16
DPI = 300
GRID_ALPHA_MAJOR = 0.3
GRID_ALPHA_MINOR = 0.15

# Colors (Mapped to Width, Depth, and Jenks)
COLOR_NAIVE = '#FFC900'
COLOR_DP = '#1F77B4'
COLOR_GRAPH = '#2CA02C'
COLOR_DIST = '#ff7f0e'

def plot_success_rates():
    
    """Reads the CSV and generates a styled plot."""
    
    df = pd.read_csv('closest_stable_percentages_summary.csv')

    # Extract axes data
    x_multipliers = df['C_Multiplier']  
    width_rates = df['Equi_Width_Stable_Pct']
    depth_rates = df['Equi_Depth_Stable_Pct']
    jenks_rates = df['Jenks_Stable_Pct']

    # Initialize plot
    plt.figure(figsize=FIG_SIZE)

    # Plot lines utilizing global styles
    plt.plot(x_multipliers, width_rates, marker='o', linestyle='-', 
             linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
             color='#1f77b4', label='Equi-Width')
             
    plt.plot(x_multipliers, depth_rates, marker='s', linestyle='-', 
             linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
             color='#ff7f0e', label='Equi-Depth')
             
    plt.plot(x_multipliers, jenks_rates, marker='^', linestyle='-', 
             linewidth=LINE_WIDTH, markersize=MARKER_SIZE, 
             color=COLOR_NAIVE, label='Jenks')

    # Axes labels
    plt.xlabel(r'$\times C_{\mathrm{default}}$', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Stable binnings (%)', fontsize=FONT_SIZE_LABEL)

    # Tick formatting
    plt.xticks(x_multipliers, fontsize=FONT_SIZE_TICKS)
    # Using steps of 20 instead of 10 to prevent overlapping with FONT_SIZE_TICKS=22
    plt.yticks(range(0, 101, 20), fontsize=FONT_SIZE_TICKS) 
    plt.ylim(-5, 105)

    # Grids
    plt.grid(True, which='major', linestyle='-', alpha=GRID_ALPHA_MAJOR)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle='--', alpha=GRID_ALPHA_MINOR)

    # Legend
    plt.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)

    # Save and display
    plt.tight_layout()
    plt.savefig('success_rates_plot.png', dpi=DPI)
    plt.show()
    # plot_success_rates(file_path)


##############
# k vs. Time Plots for Accidents and Crimes
##############

def plot_k_accidents_count_width():

    df = pd.read_csv(
        ' /closest_binning/changing_k/count_Accident_k_experiment.csv'
    )

    df = df.sort_values("k")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Width
    ax.plot(
        df['k'],
        df['Naive_Time_Width'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    # Depth
    ax.plot(
        df['k'],
        df['DP_Time_Width'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    # Jenks
    ax.plot(
        df['k'],
        df['Graph_Time_Width'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    # Log scale
    ax.set_yscale('log')

    # Axis labels
    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (s, log scale)', fontsize=FONT_SIZE_LABEL)

    # X ticks
    ax.set_xticks(df['k'])

    # Tick sizes
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    # Major grid
    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    # Minor ticks + grid
    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # # Legend
    # ax.legend(
    #     loc='best',
    #     fontsize=FONT_SIZE_LEGEND,
    #     framealpha=0.9
    # )

    # Layout
    plt.tight_layout()

    # Save
    plt.savefig(
        ' /closest_binning/images/width_k_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_k_accidents_count_plot.png")

    plt.show()




def plot_k_accidents_count_depth():

    df = pd.read_csv(
        ' /closest_binning/changing_k/count_Accident_k_experiment.csv'
    )

    df = df.sort_values("k")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Width
    ax.plot(
        df['k'],
        df['Naive_Time_Depth'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    # Depth
    ax.plot(
        df['k'],
        df['DP_Time_Depth'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    # Jenks
    ax.plot(
        df['k'],
        df['Graph_Time_Depth'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    # Log scale
    ax.set_yscale('log')

    # Axis labels
    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (s, log scale)', fontsize=FONT_SIZE_LABEL)

    # X ticks
    ax.set_xticks(df['k'])

    # Tick sizes
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    # Major grid
    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    # Minor ticks + grid
    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # Legend
    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    # Layout
    plt.tight_layout()

    # Save
    plt.savefig(
        ' /closest_binning/images/depth_k_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as depth_k_accidents_count_plot.png")

    plt.show()



def plot_k_accidents_count_jenks():

    df = pd.read_csv(
        ' /closest_binning/changing_k/count_Accident_k_experiment.csv'
    )

    df = df.sort_values("k")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Width
    ax.plot(
        df['k'],
        df['Naive_Time_Jenks'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    # Depth
    ax.plot(
        df['k'],
        df['DP_Time_Jenks'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    # Jenks
    ax.plot(
        df['k'],
        df['Graph_Time_Jenks'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    # Log scale
    ax.set_yscale('log')

    # Axis labels
    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    # X ticks
    ax.set_xticks(df['k'])

    # Tick sizes
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    # Major grid
    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    # Minor ticks + grid
    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # Legend
    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    # Layout
    plt.tight_layout()

    # Save
    plt.savefig(
        ' /closest_binning/images/jenks_k_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as jenks_k_accidents_count_plot.png")

    plt.show()



def plot_k_crimes_count_width():

    df = pd.read_csv(
        ' /closest_binning/changing_k/coutn_crimes_k_experiment.csv'
    )

    df = df.sort_values("k")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['k'], df['Naive_Time_Width'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    ax.plot(
        df['k'], df['DP_Time_Width'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    ax.plot(
        df['k'], df['Graph_Time_Width'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (s, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['k'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # ax.legend(
    #     loc='best',
    #     fontsize=FONT_SIZE_LEGEND,
    #     framealpha=0.9
    # )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_k_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_k_crimes_count_plot.png")

    plt.show()


def plot_k_crimes_count_depth():

    df = pd.read_csv(
        ' /closest_binning/changing_k/coutn_crimes_k_experiment.csv'
    )

    df = df.sort_values("k")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['k'], df['Naive_Time_Depth'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    ax.plot(
        df['k'], df['DP_Time_Depth'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    ax.plot(
        df['k'], df['Graph_Time_Depth'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['k'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/depth_k_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as depth_k_crimes_count_plot.png")

    plt.show()


def plot_k_crimes_count_jenks():

    df = pd.read_csv(
        ' /closest_binning/changing_k/coutn_crimes_k_experiment.csv'
    )

    df = df.sort_values("k")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['k'], df['Naive_Time_Jenks'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    ax.plot(
        df['k'], df['DP_Time_Jenks'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    ax.plot(
        df['k'], df['Graph_Time_Jenks'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['k'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/jenks_k_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as jenks_k_crimes_count_plot.png")

    plt.show()



def plot_k_song_sum_width():

    df = pd.read_csv(
        ' /closest_binning/algorithm_SUM_comparison_runtimes_full_experiment.csv'
    )

    df = df[df["Epsilon"] == 3].copy()

    df = df.sort_values("k")
    df['Naive_Time_Width'] = pd.to_numeric(
    df['Naive_Time_Width'],
    errors='coerce')

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['k'], df['Naive_Time_Width'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Naive',
        zorder=6
    )

    ax.plot(
        df['k'], df['DP_Time_Width'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='DP',
        zorder=6
    )

    ax.plot(
        df['k'], df['Graph_Time_Width'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Graph-Based',
        zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('k', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (s, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['k'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # ax.legend(
    #     loc='best',
    #     fontsize=FONT_SIZE_LEGEND,
    #     framealpha=0.9
    # )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_k_song_sum_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_k_song_sum_plot.png")

    plt.show()


#################
# epsilon vs. Time Plots for Accidents and Crimes
#################

def plot_epsilon_accidents_count_width():

    df = pd.read_csv(
        ' /closest_binning/changing_eps/count_accidents_eps_experiment.csv'
    )

    df = df.sort_values("Epsilon")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Width'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Width'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Width'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )
    # from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

    ax.set_yscale('log')

    # Show only 2 and 3 instead of 2×10⁻² and 3×10⁻²
    ax.yaxis.set_major_locator(FixedLocator([0.02, 0.03]))
    ax.yaxis.set_major_formatter(FixedFormatter(['20', '30']))

    # Remove minor tick labels
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Prevent scientific notation offset text
    ax.yaxis.get_offset_text().set_visible(False)
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:g}'))

    ax.set_xlabel('Epsilon', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # ax.legend(
    #     loc='best',
    #     fontsize=FONT_SIZE_LEGEND,
    #     framealpha=0.9
    # )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_epsilon_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_epsilon_accidents_count_plot.png")

    plt.show()


def plot_epsilon_accidents_count_depth():

    df = pd.read_csv(
        ' /closest_binning/changing_eps/count_accidents_eps_experiment.csv'
    )

    df = df.sort_values("Epsilon")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Depth'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Depth'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Depth'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel(r'$\epsilon$', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/depth_epsilon_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as depth_epsilon_accidents_count_plot.png")

    plt.show()


def plot_epsilon_accidents_count_jenks():

    df = pd.read_csv(
        ' /closest_binning/changing_eps/count_accidents_eps_experiment.csv'
    )

    df = df.sort_values("Epsilon")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Jenks'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Jenks'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Jenks'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel(r'$\epsilon$', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/jenks_epsilon_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as jenks_epsilon_accidents_count_plot.png")

    plt.show()


def plot_epsilon_crimes_count_width():

    df = pd.read_csv(
        ' /closest_binning/changing_eps/count_Crimes_eps_experiment.csv'
    )

    df = df.sort_values("Epsilon")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Width'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Width'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Width'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('Epsilon', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # ax.legend(
    #     loc='best',
    #     fontsize=FONT_SIZE_LEGEND,
    #     framealpha=0.9
    # )
    ax.yaxis.set_major_locator(FixedLocator([0.007, 0.008, 0.009, 0.010]))

    ax.yaxis.set_major_formatter(FixedFormatter(['7', '8', '9', '10']))

    # Remove minor tick labels

    ax.yaxis.set_minor_formatter(NullFormatter())

    # Remove scientific notation offset

    ax.yaxis.get_offset_text().set_visible(False)
    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_epsilon_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_epsilon_crimes_count_plot.png")

    plt.show()


def plot_epsilon_crimes_count_depth():

    df = pd.read_csv(
        ' /closest_binning/changing_eps/count_Crimes_eps_experiment.csv'
    )

    df = df.sort_values("Epsilon")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Depth'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Depth'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Depth'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel(r'$\epsilon$', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/depth_epsilon_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as depth_epsilon_crimes_count_plot.png")

    plt.show()


def plot_epsilon_crimes_count_jenks():

    df = pd.read_csv(
        ' /closest_binning/changing_eps/count_Crimes_eps_experiment.csv'
    )

    df = df.sort_values("Epsilon")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Jenks'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Jenks'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Jenks'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel(r'$\epsilon$', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/jenks_epsilon_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as jenks_epsilon_crimes_count_plot.png")

    plt.show()


def plot_epsilon_song_sum_width():

    df = pd.read_csv(
        ' /closest_binning/algorithm_SUM_comparison_runtimes_full_experiment.csv'
    )

    df = df[df["k"] == 5].copy()

    df = df.sort_values("Epsilon")
    df['Naive_Time_Width'] = pd.to_numeric(
    df['Naive_Time_Width'],
    errors='coerce')

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['Epsilon'], df['Naive_Time_Width'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['DP_Time_Width'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['Epsilon'], df['Graph_Time_Width'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )
    # from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

    ax.set_yscale('log')

    # Show only 2 and 3 instead of 2×10⁻² and 3×10⁻²
    ax.yaxis.set_major_locator(FixedLocator([0.02, 0.03]))
    ax.yaxis.set_major_formatter(FixedFormatter(['20', '30']))

    # Remove minor tick labels
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Prevent scientific notation offset text
    ax.yaxis.get_offset_text().set_visible(False)
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:g}'))

    ax.set_xlabel('Epsilon', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['Epsilon'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # ax.legend(
    #     loc='best',
    #     fontsize=FONT_SIZE_LEGEND,
    #     framealpha=0.9
    # )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_epsilon_song_sum_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_epsilon_song_sum_plot.png")

    plt.show()


##########
# c- changing vs running time plots for accidents and crimes
##########

def plot_c_accidents_count_width():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_accidents_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")
    df['C_Threshold'] = df['C_Threshold'] / 1000

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Width'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Width'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Width'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')
    ax.yaxis.set_major_locator( FixedLocator([0.014, 0.016, 0.018, 0.020, 0.022]) )

    ax.yaxis.set_major_formatter(FixedFormatter(['14', '16', '18', '20', '22']))

    # Remove minor tick labels

    ax.yaxis.set_minor_formatter(NullFormatter())

    # Remove scientific notation offset

    ax.yaxis.get_offset_text().set_visible(False)

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_c_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_c_accidents_count_plot.png")

    plt.show()


def plot_c_accidents_count_depth():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_accidents_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")
    df['C_Threshold'] = df['C_Threshold'] / 1000

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Depth'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Depth'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Depth'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/depth_c_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as depth_c_accidents_count_plot.png")

    plt.show()


def plot_c_accidents_count_jenks():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_accidents_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")
    df['C_Threshold'] = df['C_Threshold'] / 1000

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Jenks'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Jenks'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Jenks'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/jenks_c_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as jenks_c_accidents_count_plot.png")

    plt.show()


def plot_c_crimes_count_width():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_crimes_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")
    df['C_Threshold'] = df['C_Threshold'] / 1000
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Width'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Width'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Width'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')
    ax.yaxis.set_major_locator( FixedLocator([0.006,0.010]))

    ax.yaxis.set_major_formatter(FixedFormatter(['6','10']))

    # Remove minor tick labels

    ax.yaxis.set_minor_formatter(NullFormatter())

    # Remove scientific notation offset

    ax.yaxis.get_offset_text().set_visible(False)

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_c_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_c_crimes_count_plot.png")

    plt.show()


def plot_c_crimes_count_depth():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_crimes_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")
    df['C_Threshold'] = df['C_Threshold'] / 1000
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Depth'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Depth'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Depth'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/depth_c_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as depth_c_crimes_count_plot.png")

    plt.show()


def plot_c_crimes_count_jenks():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_crimes_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")
    df['C_Threshold'] = df['C_Threshold'] / 1000
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Jenks'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Jenks'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Jenks'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/jenks_c_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as jenks_c_crimes_count_plot.png")

    plt.show()

def plot_c_song_sum_width():

    df = pd.read_csv(
        ' /closest_binning/algorithm_SUM_C_comparison_runtimes_full_experiment.csv'
    )
    # df = df[
    #         (df["k"] == 9) &
    #         (df["Epsilon"] == 3)
    #     ].copy()

    df = df.sort_values("C_Threshold")
    df['Naive_Time_Width'] = pd.to_numeric(
    df['Naive_Time_Width'],
    errors='coerce')

    df['C_Threshold'] = df['C_Threshold'] / 3000

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df['C_Threshold'], df['Naive_Time_Width'],
        'o-', color=COLOR_NAIVE,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Naive', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['DP_Time_Width'],
        's-', color=COLOR_DP,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='DP', zorder=6
    )

    ax.plot(
        df['C_Threshold'], df['Graph_Time_Width'],
        '^-', color=COLOR_GRAPH,
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        label='Graph-Based', zorder=6
    )

    ax.set_yscale('log')
    # ax.yaxis.set_major_locator( FixedLocator([0.014, 0.016, 0.018, 0.020, 0.022]) )

    # ax.yaxis.set_major_formatter(FixedFormatter(['14', '16', '18', '20', '22']))

    # Remove minor tick labels

    ax.yaxis.set_minor_formatter(NullFormatter())

    # Remove scientific notation offset

    ax.yaxis.get_offset_text().set_visible(False)

    ax.set_xlabel('C (×10³)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('time (ms, log scale)', fontsize=FONT_SIZE_LABEL)

    ax.set_xticks(df['C_Threshold'])

    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    ax.grid(
        True, which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-', linewidth=1
    )

    ax.minorticks_on()

    ax.grid(
        True, which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':', linewidth=0.6
    )

    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    plt.savefig(
        ' /closest_binning/images/width_c_song_sum_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as width_c_song_sum_plot.png")

    plt.show()


######################
# changing c vs dest
######################

def plot_c_vs_naive_dist_accidents():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_accidents_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")

    # Scale C: 10000 -> 10, 20000 -> 20, ...
    df['C_Threshold'] = df['C_Threshold'] / 1000

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Width
    ax.plot(
        df['C_Threshold'],
        df['Naive_Dist_Width'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Width',
        zorder=6
    )

    # Depth
    ax.plot(
        df['C_Threshold'],
        df['Naive_Dist_Depth'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Depth',
        zorder=6
    )

    # Jenks
    ax.plot(
        df['C_Threshold'],
        df['Naive_Dist_Jenks'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Jenks',
        zorder=6
    )

    # Axis labels
    ax.set_xlabel(r'C ($\times 10^3$)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('dist', fontsize=FONT_SIZE_LABEL)

    # X ticks
    ax.set_xticks(df['C_Threshold'])

    # Tick sizes
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    # Major grid
    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    # Minor grid
    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # Legend
    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    # Save
    plt.savefig(
        ' /closest_binning/images/c_vs_naive_dist_accidents_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as c_vs_naive_dist_accidents_count_plot.png")

    plt.show()



def plot_c_vs_naive_dist_crimes():

    df = pd.read_csv(
        ' /closest_binning/changing_c/count_crimes_c_experiment.csv'
    )

    df = df.sort_values("C_Threshold")

    # Scale C: 10000 -> 10, 20000 -> 20, ...
    df['C_Threshold'] = df['C_Threshold'] / 1000

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Width
    ax.plot(
        df['C_Threshold'],
        df['Naive_Dist_Width'],
        'o-',
        color=COLOR_NAIVE,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Width',
        zorder=6
    )

    # Depth
    ax.plot(
        df['C_Threshold'],
        df['Naive_Dist_Depth'],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Depth',
        zorder=6
    )

    # Jenks
    ax.plot(
        df['C_Threshold'],
        df['Naive_Dist_Jenks'],
        '^-',
        color=COLOR_GRAPH,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label='Jenks',
        zorder=6
    )

    # Axis labels
    ax.set_xlabel(r'C ($\times 10^3$)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('dist', fontsize=FONT_SIZE_LABEL)

    # X ticks
    ax.set_xticks(df['C_Threshold'])

    # Tick sizes
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=FONT_SIZE_TICKS
    )

    # Major grid
    ax.grid(
        True,
        which='major',
        alpha=GRID_ALPHA_MAJOR,
        linestyle='-',
        linewidth=1
    )

    # Minor grid
    ax.minorticks_on()

    ax.grid(
        True,
        which='minor',
        alpha=GRID_ALPHA_MINOR,
        linestyle=':',
        linewidth=0.6
    )

    # Legend
    ax.legend(
        loc='best',
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.9
    )

    plt.tight_layout()

    # Save
    plt.savefig(
        ' /closest_binning/images/c_vs_naive_dist_crimes_count_plot.png',
        dpi=DPI,
        bbox_inches='tight'
    )

    print("✓ Plot saved as c_vs_naive_dist_crimes_count_plot.png")

    plt.show()



def plot_Avg_Min_max_k():

    # CSV files
    files = {
        "Average": " /closest_binning/average/Average_algorithm_comparison_runtimes_full_experiment.csv",
        "Max": " /closest_binning/MAX_algorithm_comparison_runtimes_full_experiment.csv",
        "Min": " /closest_binning/MIN_algorithm_comparison_runtimes_full_experiment.csv",
    }

    # Same color + marker for the same algorithm
    algorithms = {
        "Naive": {
            "column": "Naive_Time_Width",
            "color": COLOR_NAIVE,
            "marker": "o",
        },
        "DP": {
            "column": "DP_Time_Width",
            "color": COLOR_DP,
            "marker": "s",
        },
        "Graph-Based": {
            "column": "Graph_Time_Width",
            "color": COLOR_GRAPH,
            "marker": "^",
        },
    }

    # Different line style for each aggregate function
    dataset_styles = {
        "Average": "-",
        "Max": "--",
        "Min": ":",
    }

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for dataset_name, file_path in files.items():

        # Read CSV
        df = pd.read_csv(file_path)

        # ----------------------------------
        # KEEP ONLY EPSILON = 3
        # ----------------------------------
        df = df[df["Epsilon"] == 3].copy()

        # Sort by k
        df = df.sort_values("k")

        # Convert runtime columns to numeric
        # TIMEOUT becomes NaN
        for alg_info in algorithms.values():

            column = alg_info["column"]

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Plot each algorithm
        for alg_name, alg_info in algorithms.items():

            ax.plot(
                df["k"],
                df[alg_info["column"]],
                color=alg_info["color"],
                marker=alg_info["marker"],
                linestyle=dataset_styles[dataset_name],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                zorder=6,
            )

    # ----------------------------------
    # Y axis
    # ----------------------------------
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 2e2)

    # ----------------------------------
    # Axis labels
    # ----------------------------------
    ax.set_xlabel(
        "k",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "time (s, log scale)",
        fontsize=FONT_SIZE_LABEL
    )

    # k values
    ax.set_xticks(range(3, 11))

    # ----------------------------------
    # Grid
    # ----------------------------------
    ax.grid(
        True,
        which="major",
        alpha=GRID_ALPHA_MAJOR,
        linestyle="-",
        linewidth=1
    )

    ax.grid(
        True,
        which="minor",
        alpha=GRID_ALPHA_MINOR,
        linestyle=":",
        linewidth=0.6
    )

    # Tick size
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # ----------------------------------
    # Save
    # ----------------------------------
    save_path = (
        " /closest_binning/images/runtime_Avg_Min_Max_k_epsilon_3.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()


def plot_Avg_Min_Max_epsilon():

    # CSV files
    files = {
        "Average": " /closest_binning/average/Average_algorithm_comparison_runtimes_full_experiment.csv",
        "Max": " /closest_binning/MAX_algorithm_comparison_runtimes_full_experiment.csv",
        "Min": " /closest_binning/MIN_algorithm_comparison_runtimes_full_experiment.csv",
    }

    # Same color + marker for the same algorithm
    algorithms = {
        "Naive": {
            "column": "Naive_Time_Width",
            "color": COLOR_NAIVE,
            "marker": "o",
        },
        "DP": {
            "column": "DP_Time_Width",
            "color": COLOR_DP,
            "marker": "s",
        },
        "Graph-Based": {
            "column": "Graph_Time_Width",
            "color": COLOR_GRAPH,
            "marker": "^",
        },
    }

    # Different line style for each aggregate function
    dataset_styles = {
        "Average": "-",
        "Max": "--",
        "Min": ":",
    }

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for dataset_name, file_path in files.items():

        # Read CSV
        df = pd.read_csv(file_path)

        # ----------------------------------
        # KEEP ONLY k = 7
        # ----------------------------------
        df = df[df["k"] == 7].copy()

        # Sort by epsilon
        df = df.sort_values("Epsilon")

        # Convert runtime columns to numeric
        # TIMEOUT becomes NaN
        for alg_info in algorithms.values():

            column = alg_info["column"]

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Plot each algorithm
        for alg_name, alg_info in algorithms.items():

            ax.plot(
                df["Epsilon"],
                df[alg_info["column"]],
                color=alg_info["color"],
                marker=alg_info["marker"],
                linestyle=dataset_styles[dataset_name],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                zorder=6,
            )

    # ----------------------------------
    # Y axis - logarithmic
    # ----------------------------------
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 2e2)

    # ----------------------------------
    # Axis labels
    # ----------------------------------
    ax.set_xlabel(
        r"Epsilon",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "time (s, log scale)",
        fontsize=FONT_SIZE_LABEL
    )

    # Show epsilon = 1, 2, 3, 4, 5
    ax.set_xticks([1, 2, 3, 4, 5])

    # ----------------------------------
    # Grid
    # ----------------------------------
    ax.grid(
        True,
        which="major",
        alpha=GRID_ALPHA_MAJOR,
        linestyle="-",
        linewidth=1
    )

    ax.grid(
        True,
        which="minor",
        alpha=GRID_ALPHA_MINOR,
        linestyle=":",
        linewidth=0.6
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # ----------------------------------
    # Save
    # ----------------------------------
    save_path = (
        " /"
        "closest_binning/images/"
        "runtime_Avg_Min_Max_epsilon_k7.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()

def plot_avg_changing_C_runtime():

    file_path = (
        " /closest_binning/"
        "AGV_changing_C_algorithm_comparison_runtimes_full_experiment.csv"
    )

    df = pd.read_csv(file_path)

    # Keep k = 7 and epsilon = 3
    df = df[
        (df["k"] == 7) &
        (df["Epsilon"] == 3)
    ].copy()

    # Sort by C
    df = df.sort_values("C_Threshold")

    # Algorithms
    algorithms = {
        "Naive": {
            "column": "Naive_Time_Width",
            "color": COLOR_NAIVE,
            "marker": "o",
        },
        "DP": {
            "column": "DP_Time_Width",
            "color": COLOR_DP,
            "marker": "s",
        },
        "Graph-Based": {
            "column": "Graph_Time_Width",
            "color": COLOR_GRAPH,
            "marker": "^",
        },
    }

    # Convert runtime columns to numeric
    # TIMEOUT -> NaN
    for alg_info in algorithms.values():
        column = alg_info["column"]

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot
    for alg_name, alg_info in algorithms.items():

        ax.plot(
            df["C_Threshold"],
            df[alg_info["column"]],
            color=alg_info["color"],
            marker=alg_info["marker"],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=alg_name,
            zorder=6,
        )

    # Log scale
    ax.set_yscale("log")

    # Axis labels
    ax.set_xlabel(
        "C (×10³)",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "time (s, log scale)",
        fontsize=FONT_SIZE_LABEL
    )

    # Show C as 1, 2, 3, 4, 5 with ×10^3
    ax.set_xticks(
        [1000, 2000, 3000, 4000, 5000],
        ["1", "2", "3", "4", "5"]
    )

    # Add ×10^3 beside x axis
    ax.text(
        1.0,
        -0.13,
        "",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TICKS,
        ha="right"
    )

    # Grid
    ax.grid(
        True,
        which="major",
        alpha=GRID_ALPHA_MAJOR,
        linestyle="-",
        linewidth=1
    )

    ax.grid(
        True,
        which="minor",
        alpha=GRID_ALPHA_MINOR,
        linestyle=":",
        linewidth=0.6
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    # Legend
    ax.legend(
        fontsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # Save
    save_path = (
        " /"
        "closest_binning/images/"
        "AVG_changing_C_runtime.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()


def plot_avg_DP_distance_vs_C():

    file_path = (
        " /closest_binning/"
        "AGV_changing_C_algorithm_comparison_runtimes_full_experiment.csv"
    )

    df = pd.read_csv(file_path)

    # Keep k = 7 and epsilon = 3
    df = df[
        (df["k"] == 7) &
        (df["Epsilon"] == 3)
    ].copy()

    # Sort by C
    df = df.sort_values("C_Threshold")

    # Convert DP distance to numeric
    df["DP_Dist_Width"] = pd.to_numeric(
        df["DP_Dist_Width"],
        errors="coerce"
    )

    # ----------------------------------
    # Plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df["C_Threshold"],
        df["DP_Dist_Width"],
        color=COLOR_DP,
        marker="s",
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=6,
    )

    # ----------------------------------
    # Axis labels
    # ----------------------------------
    ax.set_xlabel(
        "C",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "distance",
        fontsize=FONT_SIZE_LABEL
    )

    # C = 1000, 2000, ... shown as 1, 2, ...
    ax.set_xticks(
        [1000, 2000, 3000, 4000, 5000],
        ["1", "2", "3", "4", "5"]
    )

    # Add ×10^3
    ax.text(
        1.0,
        -0.13,
        r"$\times 10^3$",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TICKS,
        ha="right"
    )

    # ----------------------------------
    # Grid
    # ----------------------------------
    ax.grid(
        True,
        which="major",
        alpha=GRID_ALPHA_MAJOR,
        linestyle="-",
        linewidth=1
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # ----------------------------------
    # Save
    # ----------------------------------
    save_path = (
        " /"
        "closest_binning/images/"
        "AVG_DP_distance_vs_C.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()


def plot_epsilon_DP_distance_width():

    # Read CSV
    df = pd.read_csv(
        ' /closest_binning/'
        'changing_eps/count_accidents_eps_experiment.csv'
    )

    # Keep k constant
    df = df[df["k"] == 4].copy()

    # Sort by epsilon
    df = df.sort_values("Epsilon")

    # Convert distance to numeric
    df["DP_Dist_Width"] = pd.to_numeric(
        df["DP_Dist_Width"],
        errors="coerce"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot DP Width Distance
    ax.plot(
        df["Epsilon"],
        df["DP_Dist_Width"],
        's-',
        color=COLOR_DP,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=6
    )

    # Axis labels
    ax.set_xlabel(
        "Epsilon",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "distance",
        fontsize=FONT_SIZE_LABEL
    )

    # X-axis ticks
    ax.set_xticks(sorted(df["Epsilon"].unique()))

    # Grid
    ax.grid(
        True,
        which="major",
        alpha=GRID_ALPHA_MAJOR,
        linestyle="-",
        linewidth=1
    )

    ax.grid(
        True,
        which="minor",
        alpha=GRID_ALPHA_MINOR,
        linestyle=":",
        linewidth=0.6
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # Save figure
    save_path = (
        " /"
        "closest_binning/images/"
        "accidents_epsilon_DP_distance_width.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()



def plot_k_DP_distance_width():

    # Read CSV
    df = pd.read_csv(
        ' /closest_binning/'
        'changing_k/count_Accident_k_experiment.csv'
    )

    # Keep epsilon = 1 and C = 90000
    df = df[
        (df["Epsilon"] == 1) &
        (df["C_Threshold"] == 90000)
    ].copy()

    # Sort by k
    df = df.sort_values("k")

    # Convert distance to numeric
    df["DP_Dist_Width"] = pd.to_numeric(
        df["DP_Dist_Width"],
        errors="coerce"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot DP Width Distance
    ax.plot(
        df["k"],
        df["DP_Dist_Width"],
        's-',
        color=COLOR_DIST,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=6
    )

    # Axis labels
    ax.set_xlabel(
        "k",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "distance",
        fontsize=FONT_SIZE_LABEL
    )

    # X-axis ticks
    ax.set_xticks(sorted(df["k"].unique()))

    # Grid
    ax.grid(
        True,
        which="major",
        alpha=0.3,
        linestyle="-",
        linewidth=LINE_WIDTH
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.15,
        linestyle=":",
        linewidth=LINE_WIDTH
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # Save figure
    save_path = (
        " /"
        "closest_binning/images/"
        "accidents_k_DP_distance_width.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()


def plot_C_DP_distance_width():

    # Read CSV
    df = pd.read_csv(
        ' /closest_binning/'
        'changing_c/count_accidents_c_experiment.csv'
    )

    # Keep k = 4 and epsilon = 1
    df = df[
        (df["k"] == 4) &
        (df["Epsilon"] == 1)
    ].copy()

    # Sort by C
    df = df.sort_values("C_Threshold")

    # Convert distance to numeric
    df["DP_Dist_Width"] = pd.to_numeric(
        df["DP_Dist_Width"],
        errors="coerce"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Plot DP Width Distance
    ax.plot(
        df["C_Threshold"],
        df["DP_Dist_Width"],
        's-',
        color=COLOR_DIST,
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=6
    )

    # ----------------------------------
    # Axis labels
    # ----------------------------------
    ax.set_xlabel(
        r"$C$ ($\times 10^3$)",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "distance",
        fontsize=FONT_SIZE_LABEL
    )

    # ----------------------------------
    # X-axis ticks
    # Display 20, 30, 40, ... 100
    # ----------------------------------
    c_values = sorted(df["C_Threshold"].unique())

    ax.set_xticks(c_values)

    ax.set_xticklabels([
        f"{int(c / 1000)}" for c in c_values
    ])

    # ----------------------------------
    # Grid
    # ----------------------------------
    ax.grid(
        True,
        which="major",
        alpha=0.3,
        linestyle="-",
        linewidth=LINE_WIDTH
    )

    ax.grid(
        True,
        which="minor",
        alpha=0.15,
        linestyle=":",
        linewidth=LINE_WIDTH
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # ----------------------------------
    # Save figure
    # ----------------------------------
    save_path = (
        " /"
        "closest_binning/images/"
        "accidents_C_DP_distance_width.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()


def plot_avg_DP_distance_vs_Epsilon():

    file_path = (
        " /closest_binning/average/Average_algorithm_comparison_runtimes_full_experiment.csv"
    )

    df = pd.read_csv(file_path)

    # Keep k = 7 and epsilon = 3
    df = df[
        (df["k"] == 7) 
    ].copy()

    # Sort by epsilon
    df = df.sort_values("Epsilon")

    # Convert DP distance to numeric
    df["DP_Dist_Width"] = pd.to_numeric(
        df["DP_Dist_Width"],
        errors="coerce"
    )

    # ----------------------------------
    # Plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df["Epsilon"],
        df["DP_Dist_Width"],
        color=COLOR_DP,
        marker="s",
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=6,
    )

    # ----------------------------------
    # Axis labels
    # ----------------------------------
    ax.set_xlabel(
        "Epsilon",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "distance",
        fontsize=FONT_SIZE_LABEL
    )

    # Epsilon = 1000, 2000, ... shown as 1, 2, ...
    # ax.set_xticks(
    #     [1000, 2000, 3000, 4000, 5000],
    #     ["1", "2", "3", "4", "5"]
    # )

    # Add ×10^3
    ax.text(
        1.0,
        -0.13,
        r"$\times 10^3$",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TICKS,
        ha="right"
    )

    # ----------------------------------
    # Grid
    # ----------------------------------
    ax.grid(
        True,
        which="major",
        alpha=0.3,
        linestyle="-",
        linewidth=LINE_WIDTH
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # ----------------------------------
    # Save
    # ----------------------------------
    save_path = (
        " /"
        "closest_binning/images/"
        "AVG_DP_distance_vs_Epsilon.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()


def plot_avg_DP_distance_vs_k():

    file_path = (
        " /closest_binning/average/Average_algorithm_comparison_runtimes_full_experiment.csv"
    )

    df = pd.read_csv(file_path)

    # Keep k = 7 and epsilon = 3
    df = df[
        (df["Epsilon"] == 3) 
    ].copy()

    # Sort by epsilon
    df = df.sort_values("k")

    # Convert DP distance to numeric
    df["DP_Dist_Width"] = pd.to_numeric(
        df["DP_Dist_Width"],
        errors="coerce"
    )

    # ----------------------------------
    # Plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        df["k"],
        df["DP_Dist_Width"],
        color=COLOR_DP,
        marker="s",
        linestyle="-",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=6,
    )

    # ----------------------------------
    # Axis labels
    # ----------------------------------
    ax.set_xlabel(
        "k",
        fontsize=FONT_SIZE_LABEL
    )

    ax.set_ylabel(
        "distance",
        fontsize=FONT_SIZE_LABEL
    )

    # Epsilon = 1000, 2000, ... shown as 1, 2, ...
    # ax.set_xticks(
    #     [1000, 2000, 3000, 4000, 5000],
    #     ["1", "2", "3", "4", "5"]
    # )

    # Add ×10^3
    ax.text(
        1.0,
        -0.13,
        r"$\times 10^3$",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TICKS,
        ha="right"
    )

    # ----------------------------------
    # Grid
    # ----------------------------------
    ax.grid(
        True,
        which="major",
        alpha=0.3,
        linestyle="-",
        linewidth=LINE_WIDTH
    )

    # Tick sizes
    ax.tick_params(
        axis="both",
        labelsize=FONT_SIZE_TICKS
    )

    plt.tight_layout()

    # ----------------------------------
    # Save
    # ----------------------------------
    save_path = (
        " /"
        "closest_binning/images/"
        "AVG_DP_distance_vs_k.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"✓ Plot saved as {save_path}")

    plt.show()

if __name__ == '__main__':
    plot_success_rates()

    # plot_k_accidents_count_width()

    # #plot_k_accidents_count_depth()
    ## plot_k_accidents_count_jenks()

    # plot_k_crimes_count_width()

    ## plot_k_crimes_count_depth()
    ## plot_k_crimes_count_jenks()

    # plot_epsilon_accidents_count_width()

    ## plot_epsilon_accidents_count_depth()
    ## plot_epsilon_accidents_count_jenks()

    # plot_epsilon_crimes_count_width()

    ## plot_epsilon_crimes_count_depth()
    ## plot_epsilon_crimes_count_jenks()

    # plot_c_accidents_count_width()

    ## plot_c_accidents_count_depth()
    ## plot_c_accidents_count_jenks()

    # plot_c_crimes_count_width()

    ## plot_c_crimes_count_depth()
    ## plot_c_crimes_count_jenks()
    # plot_c_vs_naive_dist_accidents()
    # plot_c_vs_naive_dist_crimes()

    # plot_Avg_Min_max_k()
    # plot_Avg_Min_Max_epsilon()
    # plot_avg_changing_C_runtime()

    # #plot_avg_DP_distance_vs_C()
    # plot_C_DP_distance_width()
    # plot_epsilon_DP_distance_width()
    # plot_k_DP_distance_width()
    # plot_k_song_sum_width()
    # plot_epsilon_song_sum_width()
    # plot_c_song_sum_width()
    # plot_avg_DP_distance_vs_Epsilon()
    # plot_avg_DP_distance_vs_k()
