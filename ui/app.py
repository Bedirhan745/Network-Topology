from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure 

from core.config import AppConfig
from core.data_loader import load_csvs
from core.graph_builder import build_graph, compute_or_load_layout
from core.solver_registry import SOLVERS
from ui.renderer import GraphRenderer


BG = "#060814"
PANEL_BG = "#0b1022"
TEXT = "#e9ecff"

EDGE = "#9bb7ff"
NODE = "#4fd3ff"
PATH_MAIN = "#7CFFEA"
PATH_GLOW = "#3EE6FF"

START_C = "#FFD166"
TARGET_C = "#FF4D6D"


def run_app():
    App().mainloop()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Network Optimization Desktop (BSM307/317)")
        self.geometry("1360x820")
        self.minsize(1200, 720)

        self.cfg = AppConfig()

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(base, "data")
        self.node_path = os.path.join(self.data_dir, "BSM307_317_Guz2025_TermProject_NodeData.csv")
        self.edge_path = os.path.join(self.data_dir, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
        self.dem_path  = os.path.join(self.data_dir, "BSM307_317_Guz2025_TermProject_DemandData.csv")

        self.G = None
        self.pos = None
        self.demands_df = None

        # navigation state
        self._press_event = None
        self._base_xlim = None
        self._base_ylim = None

        # Q-Learning önceki çözüm state (sadece Q-Learning için)
        self._prev_qlearning_solution = None  # {"path": [...], "metrics": {...}, "mode": "Delay"/"Reliability", "source": int, "target": int}
        
        # ACO önceki çözüm state (sadece ACO için)
        self._prev_aco_solution = None  # {"path": [...], "metrics": {...}, "mode": "Delay"/"Reliability", "source": int, "target": int}

        self._build_ui()
        self._load_everything()
        self._bind_navigation()

    def _build_ui(self):
        self.configure(bg=BG)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ---- Left panel ----
        left = tk.Frame(self, bg=PANEL_BG, padx=14, pady=14)
        left.grid(row=0, column=0, sticky="ns")
        left.configure(width=380)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        def label(txt):
            return tk.Label(left, text=txt, bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11, "bold"))

        r = 0

        # Algorithm
        label("Algorithm").grid(row=r, column=0, sticky="w"); r += 1
        self.alg_var = tk.StringVar(value="Dijkstra")

        # Show ACO & Q-learning in UI even if not implemented yet
        alg_values = ["Dijkstra", "ACO", "Q-Learning"]
        self.alg_cb = ttk.Combobox(left, textvariable=self.alg_var, values=alg_values, state="readonly")
        self.alg_cb.grid(row=r, column=0, sticky="ew", pady=(6, 12)); r += 1

        # Mode (Optimization Metric)
        label("Optimization Metric").grid(row=r, column=0, sticky="w"); r += 1
        # Artık sadece Delay veya Reliability seçeneği var
        self.mode_var = tk.StringVar(value="Delay")
        self.mode_cb = ttk.Combobox(
            left,
            textvariable=self.mode_var,
            values=["Delay", "Reliability"],
            state="readonly",
        )
        self.mode_cb.grid(row=r, column=0, sticky="ew", pady=(6, 14)); r += 1

        # Scenario / Demand
        label("Scenario (DemandData)").grid(row=r, column=0, sticky="w"); r += 1
        self.sc_var = tk.StringVar(value="Manual")
        self.sc_cb = ttk.Combobox(left, textvariable=self.sc_var, values=["Manual"], state="readonly")
        self.sc_cb.grid(row=r, column=0, sticky="ew", pady=(6, 10)); r += 1
        self.sc_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_scenario())

        # Start/Target
        label("Start / Target").grid(row=r, column=0, sticky="w"); r += 1
        self.start_cb = ttk.Combobox(left, state="readonly")
        self.start_cb.grid(row=r, column=0, sticky="ew", pady=(6, 8)); r += 1
        self.target_cb = ttk.Combobox(left, state="readonly")
        self.target_cb.grid(row=r, column=0, sticky="ew", pady=(0, 12)); r += 1

        # Buttons (slightly smaller)
        self.btn_calc = tk.Button(
            left,
            text="Hesapla (Solve)",
            command=self.on_solve,
            bg="#1f6feb",
            fg="white",
            activebackground="#1a5fd0",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=10,
            pady=8,
        )
        self.btn_calc.grid(row=r, column=0, sticky="ew", pady=(4, 6)); r += 1

        self.btn_reset = tk.Button(
            left,
            text="Reset",
            command=self.on_reset,
            bg="#2a2f45",
            fg=TEXT,
            activebackground="#3a4163",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=10,
            pady=7,
        )
        self.btn_reset.grid(row=r, column=0, sticky="ew"); r += 1

        # ---- Metrics area bigger + scroll ----
        label("Result Metrics").grid(row=r, column=0, sticky="w", pady=(14, 6)); r += 1

        metrics_frame = tk.Frame(left, bg=PANEL_BG)
        metrics_frame.grid(row=r, column=0, sticky="nsew")
        left.rowconfigure(r, weight=1)  # make metrics area take remaining space
        r += 1

        self.metrics_txt = tk.Text(
            metrics_frame,
            height=12,
            bg="#0a0f20",
            fg=TEXT,
            relief="flat",
            wrap="word",
            font=("Cascadia Mono", 10),  # crisp text
            padx=10,
            pady=10,
        )
        self.metrics_txt.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metrics_txt.yview)
        sb.pack(side="right", fill="y")
        self.metrics_txt.configure(yscrollcommand=sb.set)

        self.status = tk.Label(
            left,
            text="Hazır.",
            bg=PANEL_BG,
            fg="#b9c2ff",
            font=("Segoe UI", 9),
            wraplength=350,
            justify="left",
        )
        self.status.grid(row=r, column=0, sticky="w", pady=(10, 0)); r += 1

        # ---- Right plot ----
        right = tk.Frame(self, bg=BG, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig = Figure(facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_facecolor(BG)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.renderer = GraphRenderer(self.ax)
        self.renderer.apply_theme(
            bg=BG, edge=EDGE, node=NODE,
            path_main=PATH_MAIN, path_glow=PATH_GLOW,
            start=START_C, target=TARGET_C
        )

    def _load_everything(self):
        try:
            nodes, edges, demands = load_csvs(self.node_path, self.edge_path, self.dem_path, self.cfg)
            self.demands_df = demands

            self.G = build_graph(nodes, edges)
            self.pos = compute_or_load_layout(self.G, self.cfg)

            node_list = sorted(self.G.nodes())
            self.start_cb["values"] = node_list
            self.target_cb["values"] = node_list
            if node_list:
                self.start_cb.set(node_list[0])
                self.target_cb.set(node_list[1] if len(node_list) > 1 else node_list[0])

            # scenarios
            scenario_items = ["Manual"]
            for idx, row in self.demands_df.iterrows():
                scenario_items.append(
                    f"Demand #{idx+1}: S={int(row['src'])} -> D={int(row['dst'])} (bw={float(row['demand_mbps']):g})"
                )
            self.sc_cb["values"] = scenario_items
            self.sc_var.set("Manual")

            # draw
            self.renderer.draw_base(self.G, self.pos)
            self.canvas.draw_idle()

            # capture base view for reset (after draw)
            self.after(200, self._capture_base_view)

            self.status.config(text="Topoloji yüklendi. Manual veya Demand seçip Hesapla diyebilirsin.")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            self.status.config(text="Hata: veriler yüklenemedi.")

    def _apply_scenario(self):
        if self.demands_df is None:
            return
        s = self.sc_var.get()
        if s == "Manual":
            return
        try:
            k = int(s.split("#")[1].split(":")[0]) - 1
            row = self.demands_df.iloc[k]
            self.start_cb.set(int(row["src"]))
            self.target_cb.set(int(row["dst"]))
            self.status.config(text=f"Scenario seçildi: Demand #{k+1}. Start/Target güncellendi.")
        except Exception:
            pass

    def on_reset(self):
        if self.pos is None:
            return
        self.renderer.clear_path()
        self.canvas.draw_idle()
        self.metrics_txt.delete("1.0", tk.END)
        # Q-Learning ve ACO önceki çözümlerini de temizle
        self._prev_qlearning_solution = None
        self._prev_aco_solution = None
        self.status.config(text="Reset: Yol temizlendi.")

    def on_solve(self):
        if self.G is None:
            return

        try:
            s = int(self.start_cb.get())
            t = int(self.target_cb.get())
            mode = self.mode_var.get()
            alg = self.alg_var.get()

            # Show but not implemented yet
            if alg in ("ACO", "Q-Learning") and alg not in SOLVERS:
                messagebox.showinfo("Not implemented", f"{alg} henüz eklenmedi. Şimdilik Dijkstra kullanıyoruz.")
                self.status.config(text=f"{alg} seçildi ama henüz hazır değil.")
                return

            # Ağırlıklar artık optimizasyonda kullanılmıyor; arayüz uyumluluğu için boş dict geçiyoruz
            weights = {}

            demand_mbps = None
            if self.demands_df is not None and self.sc_var.get() != "Manual":
                k = int(self.sc_var.get().split("#")[1].split(":")[0]) - 1
                demand_mbps = float(self.demands_df.iloc[k]["demand_mbps"])

            solver = SOLVERS.get(alg)
            if solver is None:
                raise ValueError(f"Solver not found: {alg}")

            res = solver(self.G, s, t, mode=mode, weights=weights, demand_mbps=demand_mbps)
            path = res.get("path", [])
            m = res.get("metrics", {})

            if not path:
                messagebox.showwarning("No Path", "Bu S-D çifti arasında yol bulunamadı.")
                return

            # Q-Learning için önceki çözüm kontrolü
            if alg == "Q-Learning":
                prev = self._prev_qlearning_solution
                if prev is not None:
                    # Aynı mode, source ve target mı kontrol et
                    if (prev.get("mode") == mode and 
                        prev.get("source") == s and 
                        prev.get("target") == t):
                        
                        prev_path = prev.get("path", [])
                        prev_metrics = prev.get("metrics", {})
                        
                        if prev_path and prev_metrics:
                            # Karşılaştırma yap
                            should_keep_previous = False
                            
                            if mode == "Delay":
                                # Delay modunda: daha düşük delay daha iyi
                                new_delay = float(m.get("TotalDelay_ms", float("inf")))
                                prev_delay = float(prev_metrics.get("TotalDelay_ms", float("inf")))
                                if prev_delay <= new_delay:
                                    should_keep_previous = True
                                    self.status.config(text=f"Q-Learning: Önceki çözüm daha iyi (Delay: {prev_delay:.4f}ms vs {new_delay:.4f}ms). Önceki çözüm gösteriliyor.")
                            
                            elif mode == "Reliability":
                                # Reliability modunda: ReliabilityCost'u minimize etmek = güvenilirliği maksimize etmek
                                # Daha düşük ReliabilityCost daha iyi
                                new_cost = float(m.get("ReliabilityCost", float("inf")))
                                prev_cost = float(prev_metrics.get("ReliabilityCost", float("inf")))
                                if prev_cost <= new_cost:
                                    should_keep_previous = True
                                    self.status.config(text=f"Q-Learning: Önceki çözüm daha iyi (ReliabilityCost: {prev_cost:.6f} vs {new_cost:.6f}). Önceki çözüm gösteriliyor.")
                            
                            if should_keep_previous:
                                # Önceki çözümü kullan
                                path = prev_path
                                m = prev_metrics
                            else:
                                # Yeni çözüm daha iyi, güncelle
                                self._prev_qlearning_solution = {
                                    "path": path,
                                    "metrics": m,
                                    "mode": mode,
                                    "source": s,
                                    "target": t
                                }
                                if mode == "Delay":
                                    self.status.config(text=f"Q-Learning: Yeni çözüm daha iyi (Delay: {float(m.get('TotalDelay_ms')):.4f}ms). Güncellendi.")
                                else:
                                    self.status.config(text=f"Q-Learning: Yeni çözüm daha iyi (ReliabilityCost: {float(m.get('ReliabilityCost')):.6f}). Güncellendi.")
                        else:
                            # Önceki çözüm geçersiz, yeni çözümü kaydet
                            self._prev_qlearning_solution = {
                                "path": path,
                                "metrics": m,
                                "mode": mode,
                                "source": s,
                                "target": t
                            }
                    else:
                        # Farklı mode, source veya target - yeni çözümü kaydet
                        self._prev_qlearning_solution = {
                            "path": path,
                            "metrics": m,
                            "mode": mode,
                            "source": s,
                            "target": t
                        }
                else:
                    # İlk kez Q-Learning çalıştırılıyor - kaydet
                    self._prev_qlearning_solution = {
                        "path": path,
                        "metrics": m,
                        "mode": mode,
                        "source": s,
                        "target": t
                    }
            # ACO için önceki çözüm kontrolü
            elif alg == "ACO":
                prev = self._prev_aco_solution
                if prev is not None:
                    # Aynı mode, source ve target mı kontrol et
                    if (prev.get("mode") == mode and 
                        prev.get("source") == s and 
                        prev.get("target") == t):
                        
                        prev_path = prev.get("path", [])
                        prev_metrics = prev.get("metrics", {})
                        
                        if prev_path and prev_metrics:
                            # Karşılaştırma yap
                            should_keep_previous = False
                            
                            if mode == "Delay":
                                # Delay modunda: daha düşük delay daha iyi
                                new_delay = float(m.get("TotalDelay_ms", float("inf")))
                                prev_delay = float(prev_metrics.get("TotalDelay_ms", float("inf")))
                                if prev_delay <= new_delay:
                                    should_keep_previous = True
                                    self.status.config(text=f"ACO: Önceki çözüm daha iyi (Delay: {prev_delay:.4f}ms vs {new_delay:.4f}ms). Önceki çözüm gösteriliyor.")
                            
                            elif mode == "Reliability":
                                # Reliability modunda: ReliabilityCost'u minimize etmek = güvenilirliği maksimize etmek
                                # Daha düşük ReliabilityCost daha iyi
                                new_cost = float(m.get("ReliabilityCost", float("inf")))
                                prev_cost = float(prev_metrics.get("ReliabilityCost", float("inf")))
                                if prev_cost <= new_cost:
                                    should_keep_previous = True
                                    self.status.config(text=f"ACO: Önceki çözüm daha iyi (ReliabilityCost: {prev_cost:.6f} vs {new_cost:.6f}). Önceki çözüm gösteriliyor.")
                            
                            if should_keep_previous:
                                # Önceki çözümü kullan
                                path = prev_path
                                m = prev_metrics
                            else:
                                # Yeni çözüm daha iyi, güncelle
                                self._prev_aco_solution = {
                                    "path": path,
                                    "metrics": m,
                                    "mode": mode,
                                    "source": s,
                                    "target": t
                                }
                                if mode == "Delay":
                                    self.status.config(text=f"ACO: Yeni çözüm daha iyi (Delay: {float(m.get('TotalDelay_ms')):.4f}ms). Güncellendi.")
                                else:
                                    self.status.config(text=f"ACO: Yeni çözüm daha iyi (ReliabilityCost: {float(m.get('ReliabilityCost')):.6f}). Güncellendi.")
                        else:
                            # Önceki çözüm geçersiz, yeni çözümü kaydet
                            self._prev_aco_solution = {
                                "path": path,
                                "metrics": m,
                                "mode": mode,
                                "source": s,
                                "target": t
                            }
                    else:
                        # Farklı mode, source veya target - yeni çözümü kaydet
                        self._prev_aco_solution = {
                            "path": path,
                            "metrics": m,
                            "mode": mode,
                            "source": s,
                            "target": t
                        }
                else:
                    # İlk kez ACO çalıştırılıyor - kaydet
                    self._prev_aco_solution = {
                        "path": path,
                        "metrics": m,
                        "mode": mode,
                        "source": s,
                        "target": t
                    }
            else:
                # Q-Learning veya ACO değilse önceki çözümleri temizle
                self._prev_qlearning_solution = None
                self._prev_aco_solution = None

            # draw path
            self.renderer.draw_path(path, self.pos)
            self.canvas.draw_idle()

            # metrics text (full, scrollable)
            self.metrics_txt.delete("1.0", tk.END)

            self.metrics_txt.insert(tk.END, f"Algoritma: {alg} | Mod: {mode}\n")
            self.metrics_txt.insert(tk.END, f"Başlangıç: {s}  Hedef: {t}\n")
            self.metrics_txt.insert(tk.END, f"Yol (node list): {path}\n")
            self.metrics_txt.insert(tk.END, f"Yol (okunaklı): " + " \u2192 ".join(map(str, path)) + "\n\n")

            self.metrics_txt.insert(tk.END, f"Path length: {len(path)} nodes\n")
            self.metrics_txt.insert(tk.END, f"TotalDelay_ms: {float(m.get('TotalDelay_ms')):.4f}\n")
            self.metrics_txt.insert(tk.END, f"TotalReliability: {float(m.get('TotalReliability')):.10f}\n")
            self.metrics_txt.insert(tk.END, f"ReliabilityCost (minimize): {float(m.get('ReliabilityCost')):.6f}\n")
            self.metrics_txt.insert(tk.END, f"ResourceCost (bilgi): {float(m.get('ResourceCost')):.6f}\n")
            if "Objective" in m:
                obj = float(m.get("Objective"))
                obj_type = m.get("ObjectiveType", mode)
                if obj_type == "Delay":
                    self.metrics_txt.insert(tk.END, f"Objective (Delay, minimize): {obj:.6f}\n")
                else:
                    self.metrics_txt.insert(tk.END, f"Objective (ReliabilityCost, minimize): {obj:.6f}\n")
            if m.get("Demand_mbps") is not None:
                self.metrics_txt.insert(tk.END, f"Demand_mbps: {m.get('Demand_mbps')}\n")

            self.status.config(text=f"{alg} çalıştı. {s} -> {t} yolu çizildi.")

        except Exception as e:
            messagebox.showerror("Solve error", str(e))
            self.status.config(text="Hata: solve çalışmadı.")

    # ---------------- Zoom / Pan Navigation ----------------
    def _bind_navigation(self):
        """Mouse wheel = zoom, left-drag = pan, double click = reset view."""
        w = self.canvas.get_tk_widget()

        # Windows / macOS
        w.bind("<MouseWheel>", self._on_mousewheel)

        # Linux
        w.bind("<Button-4>", lambda e: self._on_mousewheel(e, linux_delta=+1))
        w.bind("<Button-5>", lambda e: self._on_mousewheel(e, linux_delta=-1))

        # Pan
        w.bind("<ButtonPress-1>", self._on_press)
        w.bind("<B1-Motion>", self._on_drag)
        w.bind("<ButtonRelease-1>", self._on_release)

        # Reset view
        w.bind("<Double-Button-1>", self._reset_view)

    def _capture_base_view(self):
        if self.ax is None:
            return
        self._base_xlim = self.ax.get_xlim()
        self._base_ylim = self.ax.get_ylim()

    def _reset_view(self, _event=None):
        if self._base_xlim is not None and self._base_ylim is not None:
            self.ax.set_xlim(self._base_xlim)
            self.ax.set_ylim(self._base_ylim)
            self.canvas.draw_idle()

    def _on_mousewheel(self, event, linux_delta=None):
        if self.ax is None:
            return

        xdata, ydata = self._event_to_data(event)
        if xdata is None or ydata is None:
            return

        if linux_delta is not None:
            delta = linux_delta
        else:
            delta = 1 if event.delta > 0 else -1

        zoom_in = 0.85
        zoom_out = 1 / zoom_in
        scale = zoom_in if delta > 0 else zoom_out

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        x0 = xdata - new_w * relx
        x1 = x0 + new_w
        y0 = ydata - new_h * rely
        y1 = y0 + new_h

        self.ax.set_xlim((x0, x1))
        self.ax.set_ylim((y0, y1))
        self.canvas.draw_idle()

    def _on_press(self, event):
        xdata, ydata = self._event_to_data(event)
        if xdata is None or ydata is None:
            return
        self._press_event = (xdata, ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_drag(self, event):
        if self._press_event is None:
            return
        xdata, ydata = self._event_to_data(event)
        if xdata is None or ydata is None:
            return

        xpress, ypress, (x0, x1), (y0, y1) = self._press_event
        dx = xdata - xpress
        dy = ydata - ypress

        self.ax.set_xlim((x0 - dx, x1 - dx))
        self.ax.set_ylim((y0 - dy, y1 - dy))
        self.canvas.draw_idle()

    def _on_release(self, _event):
        self._press_event = None

    def _event_to_data(self, event):
        """Convert Tk event coords to Matplotlib data coords."""
        x = event.x
        y = event.y

        width = self.canvas.get_tk_widget().winfo_width()
        height = self.canvas.get_tk_widget().winfo_height()
        if width <= 0 or height <= 0:
            return (None, None)

        fig_x = x / width
        fig_y = 1 - (y / height)  # Tk top-left, MPL bottom-left

        inv = self.ax.transData.inverted()
        disp = self.fig.transFigure.transform((fig_x, fig_y))
        data = inv.transform(disp)
        return float(data[0]), float(data[1])
