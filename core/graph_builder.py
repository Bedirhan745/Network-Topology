import os
import json
import networkx as nx

from .config import AppConfig


def build_graph(nodes_df, edges_df) -> nx.Graph:
    G = nx.Graph()

    # nodes: node_id, s_ms, r_node
    for _, row in nodes_df.iterrows():
        nid = int(row["node_id"])
        G.add_node(
            nid,
            processing_delay_ms=float(row["s_ms"]),
            node_reliability=float(row["r_node"]),
        )

    # edges: src, dst, capacity_mbps, delay_ms, r_link
    for _, row in edges_df.iterrows():
        u = int(row["src"])
        v = int(row["dst"])
        G.add_edge(
            u, v,
            bandwidth_mbps=float(row["capacity_mbps"]),
            link_delay_ms=float(row["delay_ms"]),
            link_reliability=float(row["r_link"]),
        )

    return G


def compute_or_load_layout(G: nx.Graph, cfg: AppConfig) -> dict[int, tuple[float, float]]:
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    cache_path = os.path.join(cfg.outputs_dir, cfg.layout_cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}

    pos = nx.spring_layout(G, seed=cfg.layout_seed, k=cfg.layout_k, iterations=cfg.layout_iterations)
    to_save = {str(k): [float(v[0]), float(v[1])] for k, v in pos.items()}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f)

    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
