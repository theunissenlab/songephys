import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
        conditions, 
        positions=None,
        offsets=None,
        std=0.5,
        drawlines=False,
        violin=True,
        violin_width=0.4,
        colors=None,
        mean_color="Black",
        violin_bandwith=0.4,
        ax=None
    ):
    """Compare 1+ sets of data using scatter and violin plots
    
    Parameters
        conditions: list of lists, each inner list contains
            values under that condition
        positions: array-like object representing the x pos
            to plot each condition
        offsets: how much to offset the scatter plot from
            violin. defaults to 0 if violin is False
        std: std of the scatter dithering
        drawlines: draw lines between corresponding data
            across columns (in this case, each column of
            conditions must be equal length), default to
            False
        violin: boolean indicator to show violin plot
            defaults to True
        violin_width: width of the violin plot
        violin_bandwith: bandwith of violin
        colors: array-like with color for each datapoint
            (all columns of conditions must be equal)
            # TODO: allow colors to be assigned per
            # column?
        mean_color: color to plot mean on violin plot
        ax: axes object to plot on
    """
    if ax is None:
        ax = plt.gca()
    if positions is None:
        positions = np.arange(len(conditions))
    if offsets is None or violin is False:
        offsets = np.zeros(len(conditions))
    
    x_values = []
    for i, cond in enumerate(conditions):
        x_values.append(np.random.normal(positions[i] - offsets[i], std, size=(len(cond))))
        ax.scatter(
            x_values[i],
            cond, 
            facecolor="none",
            marker="o",
            edgecolor="#1F618D" if colors is None else colors, 
            alpha=0.8, 
            s=140
        )
    x_values = np.array(x_values)

    if drawlines:
        for i, row in enumerate(np.array(conditions).T):
            ax.plot(
                x_values[~np.isnan(row), i],
                row[~np.isnan(row)], 
                color="Black" if colors is None else colors[i],
                alpha=0.4)

    if violin:
        select = [len(x[~np.isnan(x)]) > 0 for x in conditions]
        parts = ax.violinplot(
            [x[~np.isnan(x)] for x in np.array(conditions)[select]],
            positions=np.array(positions)[select] + np.array(offsets)[select],
            showmeans=False,
            showextrema=False,
            widths=violin_width,
            bw_method=violin_bandwith,
        )
        for pc in parts['bodies']:
            pc.set_facecolor("Grey")
            pc.set_alpha(0.2)

        for i, cond in enumerate(conditions):
            ax.hlines(
                np.mean(cond),
                positions[i] + offsets[i] - violin_width / 2,
                positions[i] + offsets[i] + violin_width / 2,
                color=mean_color
            )

    return ax

