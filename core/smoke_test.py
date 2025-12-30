import os
from core.config import AppConfig
from core.data_loader import load_csvs
from core.graph_builder import build_graph, compute_or_load_layout
from core.solver_registry import SOLVERS

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base, "data")

    node_path = os.path.join(data_dir, "BSM307_317_Guz2025_TermProject_NodeData.csv")
    edge_path = os.path.join(data_dir, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
    dem_path  = os.path.join(data_dir, "BSM307_317_Guz2025_TermProject_DemandData.csv")

    cfg = AppConfig()
    nodes, edges, demands = load_csvs(node_path, edge_path, dem_path, cfg)
    G = build_graph(nodes, edges)
    pos = compute_or_load_layout(G, cfg)

    print("OK: nodes =", G.number_of_nodes(), "edges =", G.number_of_edges(), "demands =", len(demands))
    print("OK: layout cached for", len(pos), "nodes")

    # test first demand
    d0 = demands.iloc[0]
    s, t, bw = int(d0["src"]), int(d0["dst"]), float(d0["demand_mbps"])

    weights = {"w_delay": 1/3, "w_rel": 1/3, "w_res": 1/3}
    res = SOLVERS["Dijkstra"](G, s, t, mode="WeightedSum", weights=weights, demand_mbps=bw)

    print("Test demand:", s, "->", t, "bw=", bw)
    print("Path len:", len(res.get("path", [])))
    print("Metrics:", res.get("metrics", {}))

if __name__ == "__main__":
    main()
