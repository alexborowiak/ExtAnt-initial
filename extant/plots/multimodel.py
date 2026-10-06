"""Multi-model summaries: the median across models on maps, hatched where they disagree, and model x experiment tables.

Draws the output of ``significance.model_agreement`` and ``significance.area_mean``.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from plotting_modules import maps
from plotting_modules.utils import plot

from .. import config


@plot("figure")
def model_agreement_maps(agreement, title, cbar_label, cmap="RdBu_r", levels=None, row_dim="experiment",
                         col_dim="season", sign_threshold=config.SIGN_THRESHOLD,
                         significant_threshold=config.SIGNIFICANT_THRESHOLD, **grid_kwargs):
    """Multi-model median on a polar grid, hatched where the models give no robust signal.

    With a significance test (IPCC AR6 advanced approach): ``////`` where fewer
    than ``significant_threshold`` of the models are significant, ``xxxx``
    where enough are but fewer than ``sign_threshold`` agree on the sign.
    Without one, ``////`` where fewer than ``sign_threshold`` agree on the sign.

    Args:
        agreement (xr.Dataset): ``significance.model_agreement`` output.
        title, cbar_label (str): Figure title and colour-bar label.
        cmap: Colormap.
        levels (array-like | None): Contour levels; symmetric about 0 over the 98th percentile of |median| if None.
        row_dim, col_dim (str): Dims mapped to rows and columns.
        sign_threshold, significant_threshold (float): The thresholds ``agreement`` was made with (for the legend).
        **grid_kwargs: Passed to ``maps.polar_grid`` (e.g. norm).

    Returns:
        core.Panels
    """
    median = agreement["median"]
    if levels is None:
        limit = float(np.nanquantile(np.abs(median.values), 0.98))
        levels = np.linspace(-limit, limit, 13)

    panels = maps.polar_grid(median, row_dim=row_dim, col_dim=col_dim, levels=levels, cmap=cmap,
                             title=title, cbar_label=cbar_label, tag=False, **grid_kwargs)

    if "no_signal" in agreement:
        hatches = {
            "////": (agreement["no_signal"], f"No robust signal (< {significant_threshold:.0%} of models significant)"),
            "xxxx": (agreement["conflicting"], f"Conflicting (significant, < {sign_threshold:.0%} agree on sign)"),
        }
    else:
        hatches = {"////": (agreement["sign_agreement"] < sign_threshold,
                            f"< {sign_threshold:.0%} of models agree on sign")}

    rows, cols = median[row_dim].values, median[col_dim].values
    axes = np.asarray(panels.axes, dtype=object).reshape(len(rows), len(cols))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            for hatch, (mask, _) in hatches.items():
                maps.plot_hatch(axes[i, j], mask.sel({row_dim: row, col_dim: col}),
                                hatch=hatch, color=(0, 0, 0, 0.3), linewidth=0.3)

    panels.fig.legend(
        handles=[Patch(facecolor="white", edgecolor="0.25", hatch=hatch, label=label)
                 for hatch, (_, label) in hatches.items()],
        loc="lower center", ncol=len(hatches), frameon=False,
        handlelength=2.2, handleheight=1.2, bbox_to_anchor=(0.5, 0.005),
    )
    return panels


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
