import numpy as np
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import NonUniformImage


def alpha_plot(x, mean, sigma, ax, color, dkl=None, linecolor='C0',
               max_alpha=1.0, **kwargs):
    lower = mean - sigma
    upper = mean + sigma

    if dkl is not None:
        alpha = dkl / np.nanmax(dkl)

        points = np.array([x, mean]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        linecmap = LinearSegmentedColormap.from_list(
            None, [mcolors.to_rgb(linecolor) + (0,), mcolors.to_rgba(linecolor)], N=256)
        cmap = LinearSegmentedColormap.from_list(
            None, [mcolors.to_rgb(color) + (0,), mcolors.to_rgba(color)], N=256)

        lc = LineCollection(segments, array=alpha,
                            cmap=linecmap, norm=plt.Normalize(0, 1),
                            lw=2,
                            )

        # top
        points = np.array([x, mean+sigma]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lct = LineCollection(segments, array=alpha,
                             cmap=cmap, norm=plt.Normalize(0, 1),
                             lw=1.5)
        # bottom
        points = np.array([x, mean-sigma]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lcb = LineCollection(segments, array=alpha,
                             cmap=cmap, norm=plt.Normalize(0, 1),
                             lw=1.5)

        alpha *= max_alpha
        # build the "strip" of alphas – 2 rows so we can fill between
        # lower & upper
        Z = np.vstack([alpha, alpha])    # shape (2, N)

        # build the grid in data‐space
        # X will be repeated x twice, Y will be two rows [lower, upper]
        X, Y = np.broadcast_arrays(x[np.newaxis, :], np.vstack([lower, upper]))

        # draw it – pcolormesh places each quad at the true data‐coordinates
        im = NonUniformImage(ax,
                             cmap=cmap,
                             interpolation='nearest',
                             norm=plt.Normalize(0, 1),
                             interpolation_stage='rgba',
                             )
        im.set_data(x, [lower.min(), upper.max()], Z)
        im.set_extent((x.min(), x.max(),
                       lower.min(), upper.max()))
        im.set_clip_path(ax.patch)
        ax.add_image(im)

        # now clip exactly as before
        verts = np.vstack([
            np.column_stack([x,       upper]),
            np.column_stack([x[::-1], lower[::-1]])
        ])
        clip_path = Path(verts)
        patch = PathPatch(clip_path,
                          facecolor='none',
                          edgecolor='none')
        ax.add_patch(patch)
        im.set_clip_path(patch)

        # finally the dashed line on top
        for _l in [lc, lct, lcb]:
            ax.add_collection(_l)
    else:
        ax.plot(x, mean, color=linecolor, linestyle='--')
        ax.fill_between(x, lower, upper, color=color, alpha=0.5)
