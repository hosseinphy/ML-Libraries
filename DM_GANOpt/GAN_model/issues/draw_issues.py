"""
Draws all detected voxel grid issues using matplotlib.

Displays each slice within the voxel grid, where each issue is coloured differently.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

# Pick backend BEFORE importing pyplot
import matplotlib
# Use TkAgg only if you actually have it; otherwise comment this out or use Qt5Agg
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from  . import overhangs
from . import islands
from . import resin_traps
from . import utils


def draw_issues(fname):
    fname = Path(fname).resolve()

    # Load and frame
    grid = utils.add_frame(np.load(fname))

    # Compute masks
    ils, n_ils = islands.get_island_mask(grid)
    rts, n_rts = resin_traps.get_resin_trap_mask(grid)
    ohs, n_ohs = overhangs.get_overhang_mask(grid)

  
    print("Number of Islands:", n_ils)
    print("Number of Overhangs:", n_ohs)
    print("Number of Resin Traps:", n_rts)

    # Prepare RGB volumes
    base_rgb = np.repeat(grid[:, :, :, np.newaxis] * 255, 3, axis=3).astype(np.uint8)
    grid_ohs = deepcopy(base_rgb)
    grid_ils = deepcopy(base_rgb)
    grid_rts = deepcopy(base_rgb)

    # Color-code masks
    grid_ils[ils] = (255, 0, 0)       # Red
    grid_rts[rts] = (0, 255, 0)       # Green
    grid_ohs[ohs] = (255, 0, 255)     # Purple

    # Slice range
    max_slic = len(grid)              # N
    init_slic = 0
    allowed_slices = list(range(max_slic))  # [0..N-1]

    # Figure & axes
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ils_imshow = ax1.imshow(grid_ils[init_slic])
    rts_imshow = ax2.imshow(grid_rts[init_slic])
    ohs_imshow = ax3.imshow(grid_ohs[init_slic])

    ax1.set_title(f"Islands, Total={n_ils}")
    ax2.set_title(f"Resin Traps, Total={n_rts}")
    ax3.set_title(f"Overhangs, Total={n_ohs}")

    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Slider
    ax_slic = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    slic_slider = Slider(
        ax=ax_slic,
        label="Slice Index",
        valmin=0,
        valmax=max_slic - 1,          # <= index upper bound
        valstep=allowed_slices,       # ensures integer steps
        valinit=init_slic,
        orientation="vertical"
    )

    # Define callbacks INSIDE so they can capture the variables above
    def update(val):
        idx = int(slic_slider.val)
        ils_imshow.set_data(grid_ils[idx])
        rts_imshow.set_data(grid_rts[idx])
        ohs_imshow.set_data(grid_ohs[idx])
        fig.canvas.draw_idle()

    def reset(event):
        slic_slider.reset()

    slic_slider.on_changed(update)

    # Reset button
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    button.on_clicked(reset)

    plt.show()
    # # Show or save depending on backend
    # if "agg" in plt.get_backend().lower():
    #     plt.savefig("issues.png", dpi=200, bbox_inches="tight")
    # else:
    #     plt.show()
