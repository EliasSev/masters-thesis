"""
Utils for plotting and graphics.
"""
from pathlib import Path
import matplotlib.pyplot as plt


def save_plot(fig_name: str) -> None:
    if fig_name is not None:
        path = Path(f"../../figures/{fig_name}.png")
        if not path.exists():
            plt.savefig(path, dpi=300, bbox_inches="tight")
        else:
            print(path, 'already exists')
