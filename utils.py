
"""
Plotting tools
"""

import numpy as np
import plotly.offline as offline
import plotly.graph_objs as go


def plot_lines(data: dict, xlims: list=None, plot_name: str="untitled") -> None:
    N = 300
    if xlims is not None:
        assert(len(xlims) == 2)
    fig = []
    for name, data in data.items():
        if isinstance(data, list):
            assert(len(data) == 2)
            if xlims is not None:
                mask = (xlims[0] <= data[0]) & (data[0] <= xlims[1])
                xs, ys = data[0][mask], data[1][mask]
            else:
                xs, ys = data[0], data[1]
            fig.append(go.Scatter(x=xs, y=ys, mode="markers", name=name))
        elif callable(data):
            assert(xlims is not None)
            xs = np.linspace(*xlims, N)
            ys = data(xs)
            fig.append(go.Scatter(x=xs, y=ys, mode="lines", name=name))
        else:
            raise RuntimeError("Unrecognized data type")

    offline.plot(fig, filename=plot_name + ".html")

