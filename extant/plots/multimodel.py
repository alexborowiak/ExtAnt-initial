"""Multi-model summaries: model x experiment tables.

Draws the output of ``significance.area_mean``. The maps of the models' agreement are drawn with ``antarctica``.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotting_modules.utils import plot


@plot("figure")
def model_heatmap(da, title, cbar_label, annot=None, cmap="RdBu_r", col_dim="season"):
    """Model x experiment table of a scalar (e.g. an Antarctic mean), one panel per season.

    A multi-model median row is added at the bottom. ``annot`` (same dims,
    values in 0-1) is written in each cell as a percentage.

    Returns:
        tuple: (fig, axes)
    """
    def with_median(x):
        return xr.concat([x, x.median("model").expand_dims(model=["Multi-model median"])], dim="model")

    da = with_median(da)
    annot = with_median(annot) if annot is not None else None
    limit = float(np.nanmax(np.abs(da.values)))

    cols = da[col_dim].values
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols) + 1.5, 0.45 * da.sizes["model"] + 2),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        d = da.sel({col_dim: col}).transpose("model", "experiment")
        image = ax.imshow(d.values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.axhline(d.sizes["model"] - 1.5, color="k", lw=1)
        ax.set_xticks(range(d.sizes["experiment"]), d["experiment"].values, rotation=45, ha="right")
        ax.set_yticks(range(d.sizes["model"]), d["model"].values)
        ax.set_title(str(col))
        if annot is not None:
            a = annot.sel({col_dim: col}).transpose("model", "experiment").values
            for (i, j), v in np.ndenumerate(a):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=axes, label=cbar_label, shrink=0.8)
    fig.suptitle(title)
    return fig, axes
