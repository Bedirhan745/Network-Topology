from __future__ import annotations

from matplotlib.collections import LineCollection


class GraphRenderer:
    """
    Fast drawing:
      - base edges: LineCollection
      - base nodes: scatter
      - path: glow + main
      - path nodes: highlighted scatter + labels
      - start/target: extra emphasis
    """

    def __init__(self, ax):
        self.ax = ax

        self.edge_collection = None
        self.node_scatter = None

        self.path_glow = None
        self.path_main = None

        self.path_node_scatter = None
        self.start_scatter = None
        self.target_scatter = None

        self._path_labels = []

        # theme colors
        self._bg = "#060814"
        self._edge = "#9bb7ff"
        self._node = "#4fd3ff"
        self._path_main = "#7CFFEA"
        self._path_glow = "#3EE6FF"
        self._start = "#FFD166"   # warm yellow
        self._target = "#FF4D6D"  # pink/red

    def apply_theme(self, bg="#060814", edge="#9bb7ff", node="#4fd3ff",
                    path_main="#7CFFEA", path_glow="#3EE6FF",
                    start="#FFD166", target="#FF4D6D"):
        self._bg = bg
        self._edge = edge
        self._node = node
        self._path_main = path_main
        self._path_glow = path_glow
        self._start = start
        self._target = target

        self.ax.set_facecolor(self._bg)

        if self.edge_collection:
            self.edge_collection.set_color(self._edge)
        if self.node_scatter:
            self.node_scatter.set_color(self._node)

        if self.path_main:
            self.path_main.set_color(self._path_main)
        if self.path_glow:
            self.path_glow.set_color(self._path_glow)

        if self.path_node_scatter:
            self.path_node_scatter.set_color(self._path_main)

        if self.start_scatter:
            self.start_scatter.set_color(self._start)
        if self.target_scatter:
            self.target_scatter.set_color(self._target)

        # non-blocking redraw (Tkinter friendly)
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def draw_base(self, G, pos):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_facecolor(self._bg)

        # --- edges (fast) ---
        segs = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            segs.append([(x0, y0), (x1, y1)])

        self.edge_collection = LineCollection(segs, linewidths=0.35, alpha=0.10)
        self.edge_collection.set_color(self._edge)
        self.ax.add_collection(self.edge_collection)

        # --- nodes ---
        xs = [pos[n][0] for n in G.nodes()]
        ys = [pos[n][1] for n in G.nodes()]
        self.node_scatter = self.ax.scatter(xs, ys, s=18, alpha=0.95)
        self.node_scatter.set_color(self._node)

        # --- path lines ---
        self.path_glow, = self.ax.plot([], [], linewidth=10, alpha=0.0, solid_capstyle="round")
        self.path_main, = self.ax.plot([], [], linewidth=3.5, alpha=0.0, solid_capstyle="round")
        self.path_glow.set_color(self._path_glow)
        self.path_main.set_color(self._path_main)

        # --- path nodes / start / target overlays ---
        self.path_node_scatter = self.ax.scatter([], [], s=80, alpha=0.0, zorder=5)
        self.path_node_scatter.set_color(self._path_main)

        self.start_scatter = self.ax.scatter([], [], s=180, alpha=0.0, zorder=6)
        self.start_scatter.set_color(self._start)

        self.target_scatter = self.ax.scatter([], [], s=180, alpha=0.0, zorder=6)
        self.target_scatter.set_color(self._target)

        # clean labels
        self._clear_path_labels()

        # Fit nicely
        self.ax.relim()
        self.ax.autoscale_view()

        # non-blocking redraw
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def _clear_path_labels(self):
        for t in self._path_labels:
            try:
                t.remove()
            except Exception:
                pass
        self._path_labels = []

    def clear_path(self):
        # lines
        if self.path_glow:
            self.path_glow.set_data([], [])
            self.path_glow.set_alpha(0.0)
        if self.path_main:
            self.path_main.set_data([], [])
            self.path_main.set_alpha(0.0)

        # Matplotlib set_offsets expects (N,2). Provide a "hidden" point.
        nan_pt = [(float("nan"), float("nan"))]

        # scatters
        if self.path_node_scatter:
            self.path_node_scatter.set_offsets(nan_pt)
            self.path_node_scatter.set_alpha(0.0)
        if self.start_scatter:
            self.start_scatter.set_offsets(nan_pt)
            self.start_scatter.set_alpha(0.0)
        if self.target_scatter:
            self.target_scatter.set_offsets(nan_pt)
            self.target_scatter.set_alpha(0.0)

        self._clear_path_labels()

        # non-blocking redraw
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()

    def draw_path(self, path, pos):
        # Guard: if base not drawn yet, avoid crashing
        if self.path_glow is None or self.path_main is None or self.path_node_scatter is None:
            return

        if not path or len(path) < 2:
            self.clear_path()
            return

        xs = [pos[n][0] for n in path]
        ys = [pos[n][1] for n in path]

        # lines
        self.path_glow.set_data(xs, ys)
        self.path_main.set_data(xs, ys)
        self.path_glow.set_alpha(0.55)
        self.path_main.set_alpha(0.95)

        # node highlights (all path nodes)
        offsets = list(zip(xs, ys))
        self.path_node_scatter.set_offsets(offsets)
        self.path_node_scatter.set_alpha(0.85)

        # start / target big circles
        if self.start_scatter and self.target_scatter:
            sx, sy = pos[path[0]]
            tx, ty = pos[path[-1]]

            self.start_scatter.set_offsets([(sx, sy)])
            self.start_scatter.set_alpha(0.95)

            self.target_scatter.set_offsets([(tx, ty)])
            self.target_scatter.set_alpha(0.95)

        # labels only for path nodes (to not clutter)
        self._clear_path_labels()
        for n in path:
            x, y = pos[n]
            t = self.ax.text(
                x, y, str(n),
                fontsize=9,
                ha="center", va="center",
                color="#041018",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="#9bf6ff", ec="none", alpha=0.85)
            )
            self._path_labels.append(t)

        # non-blocking redraw
        if self.ax.figure and self.ax.figure.canvas:
            self.ax.figure.canvas.draw_idle()
