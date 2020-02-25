"""Functions to generate customized plots in ``plots.py``.

"""
import missingno as msno
import numpy as np


def matrix_nan(df):
    """Customize nullity matrix to visualize missing data pattern in dataset.

    Args:
        df (pd.DataFrame): The dataframe to visualize.

    Returns:
        matplotlib Axes: Axes object with the nullity matrix.

    """
    matrix_nan = msno.matrix(df)
    matrix_nan.set_ylabel("INDEX OF OBSERVATIONS", labelpad=0, fontsize=18)
    matrix_nan.get_xticklabels()[19].set_fontweight("bold")
    return matrix_nan


def heatmap_nan(df):
    """Customize nullity matrix to visualize missing data pattern in dataset.

    Args:
        df (pd.DataFrame): The dataframe to visualize.

    Returns:
        matplotlib Axes: Axes object with the heatmap.

    """
    heatmap_nan = msno.heatmap(df, vmin=0, cmap="OrRd")
    heatmap_nan.get_xticklabels()[16].set_fontweight("bold")
    heatmap_nan.get_yticklabels()[16].set_fontweight("bold")
    # Interesting fact:
    # When plotting heatmaps with seaborn (on which the "missingno" library
    # builds), the first and the last row is cut in halve, because of a bug
    # in the matplotlib regression between 3.1.0 and 3.1.1
    # We are correcting it this way:
    bottom, top = heatmap_nan.get_ylim()
    heatmap_nan.set_ylim(bottom + 0.5, top - 0.5)
    positions = np.array([1, 3, 5, 8, 10, 14, 16])
    labels = [
        "BACKGROUND",
        "HOUSEHOLD",
        "FINANCE",
        "HEALTH",
        "EMPLOYMENT",
        "PERSONALITY",
    ]
    heatmap_nan.hlines(positions, xmin=0, xmax=positions, lw=8, color="white")
    for position, label in zip(positions, labels):
        heatmap_nan.text(position + 0.35, position + 0.35, label, fontsize=14)
    return heatmap_nan
