import pandas as pd

from .config import AppConfig


def _to_float_decimal_comma(series: pd.Series) -> pd.Series:
    # "0,962" -> 0.962
    return series.astype(str).str.replace(",", ".", regex=False).astype(float)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # trim + lowercase
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)
    return df


def _rename_to_canonical(nodes: pd.DataFrame, edges: pd.DataFrame, demands: pd.DataFrame):
    """
    Canonical names:
      nodes: node_id, s_ms, r_node
      edges: src, dst, capacity_mbps, delay_ms, r_link
      demands: src, dst, demand_mbps
    """
    nodes = _normalize_columns(nodes)
    edges = _normalize_columns(edges)
    demands = _normalize_columns(demands)

    # common alternative headers -> canonical
    nodes_rename = {
        "nodeid": "node_id",
        "id": "node_id",
        "node": "node_id",
        "processingdelay": "s_ms",
        "processing_delay": "s_ms",
        "processing_delay_ms": "s_ms",
        "s": "s_ms",
        "rn": "r_node",
        "nodereliability": "r_node",
        "node_reliability": "r_node",
    }
    edges_rename = {
        "from": "src",
        "to": "dst",
        "u": "src",
        "v": "dst",
        "bandwidth": "capacity_mbps",
        "bandwidth_mbps": "capacity_mbps",
        "capacity": "capacity_mbps",
        "linkdelay": "delay_ms",
        "link_delay": "delay_ms",
        "link_delay_ms": "delay_ms",
        "reliability": "r_link",
        "linkreliability": "r_link",
        "link_reliability": "r_link",
    }
    demands_rename = {
        "bw": "demand_mbps",
        "demand": "demand_mbps",
        "demand_bw": "demand_mbps",
        "demand_bw_mbps": "demand_mbps",
    }

    nodes = nodes.rename(columns={k: v for k, v in nodes_rename.items() if k in nodes.columns})
    edges = edges.rename(columns={k: v for k, v in edges_rename.items() if k in edges.columns})
    demands = demands.rename(columns={k: v for k, v in demands_rename.items() if k in demands.columns})

    return nodes, edges, demands


def load_csvs(node_path: str, edge_path: str, demand_path: str, cfg: AppConfig | None = None):
    cfg = cfg or AppConfig()

    nodes = pd.read_csv(node_path, sep=cfg.sep)
    edges = pd.read_csv(edge_path, sep=cfg.sep)
    demands = pd.read_csv(demand_path, sep=cfg.sep)

    nodes, edges, demands = _rename_to_canonical(nodes, edges, demands)

    # Required columns check (after rename)
    req_nodes = {"node_id", "s_ms", "r_node"}
    req_edges = {"src", "dst", "capacity_mbps", "delay_ms", "r_link"}
    req_dem = {"src", "dst", "demand_mbps"}

    if not req_nodes.issubset(set(nodes.columns)):
        raise KeyError(f"NodeData missing columns. Have={list(nodes.columns)} Need={sorted(req_nodes)}")
    if not req_edges.issubset(set(edges.columns)):
        raise KeyError(f"EdgeData missing columns. Have={list(edges.columns)} Need={sorted(req_edges)}")
    if not req_dem.issubset(set(demands.columns)):
        raise KeyError(f"DemandData missing columns. Have={list(demands.columns)} Need={sorted(req_dem)}")

    # Decimal comma cleanup
    if cfg.decimal_comma:
        nodes["s_ms"] = _to_float_decimal_comma(nodes["s_ms"])
        nodes["r_node"] = _to_float_decimal_comma(nodes["r_node"])
        edges["r_link"] = _to_float_decimal_comma(edges["r_link"])

    # types
    nodes["node_id"] = nodes["node_id"].astype(int)

    edges["src"] = edges["src"].astype(int)
    edges["dst"] = edges["dst"].astype(int)
    edges["capacity_mbps"] = edges["capacity_mbps"].astype(float)
    edges["delay_ms"] = edges["delay_ms"].astype(float)

    demands["src"] = demands["src"].astype(int)
    demands["dst"] = demands["dst"].astype(int)
    demands["demand_mbps"] = demands["demand_mbps"].astype(float)

    return nodes, edges, demands
