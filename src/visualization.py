"""Figure export helpers for analysis notebooks"""

import matplotlib.pyplot as plt


def export_figure(figure, name, *, directory, period):
    """Save the displayed figure and return it with its output path"""

    if figure is None:
        return None, []

    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{name}_{period}.pdf"

    figure.savefig(
        path,

        dpi        =160    ,
        bbox_inches="tight",
        metadata   ={
            "CreationDate" : None,
            "ModDate"      : None,
        },
    )

    plt.close(figure)

    return figure, [path]
