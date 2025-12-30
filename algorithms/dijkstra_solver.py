import math
import networkx as nx
from core import metrics


def solve(
    G: nx.Graph,
    source: int,
    target: int,
    mode: str,
    weights: dict,  # UI arayüzü ile uyumluluk için tutuluyor, artık kullanılmıyor
    demand_mbps: float | None = None,
):
    """
    Dijkstra ile en kısa yol hesabı.

    mode:
      - "Delay"       -> Toplam gecikmeyi (TotalDelay_ms) minimize eder.
      - "Reliability" -> Toplam güvenilirliği maksimize eder
                         (eşdeğeri: ReliabilityCost'u minimize etmek).
    """

    def edge_weight(u, v, data):
        # Delay metriği için: link_delay + (hedef olmayan düğümün processing_delay)
        if mode == "Delay":
            w = float(data["link_delay_ms"])
            if v != target:
                w += float(G.nodes[v]["processing_delay_ms"])
            return w

        if mode == "Reliability":
            # Güvenilirlik çarpımsal bir metrik => -log(rel) toplamını minimize et
            r_link = float(data["link_reliability"])
            r_link = min(max(r_link, 1e-12), 1.0)
            w = -math.log(r_link)

            r_node = float(G.nodes[v]["node_reliability"])
            r_node = min(max(r_node, 1e-12), 1.0)
            w += -math.log(r_node)
            return w

        # Varsayılan güvenli davranış: Delay
        w = float(data["link_delay_ms"])
        if v != target:
            w += float(G.nodes[v]["processing_delay_ms"])
        return w

    try:
        path = nx.dijkstra_path(G, source, target, weight=edge_weight)
    except nx.NetworkXNoPath:
        path = []

    out = {"path": path, "metrics": {}}
    if not path:
        return out

    out["metrics"] = {
        "TotalDelay_ms": metrics.total_delay_ms(G, path),
        "TotalReliability": metrics.total_reliability(G, path),
        "ReliabilityCost": metrics.reliability_cost(G, path),
        "ResourceCost": metrics.resource_cost(G, path),
        # Artık WeightedSum yok; amaç metrikleri doğrudan yukarıdaki alanlardan okunuyor
        "Objective": (
            metrics.total_delay_ms(G, path)
            if mode == "Delay"
            else metrics.reliability_cost(G, path)
        ),
        "ObjectiveType": mode,
        "Demand_mbps": demand_mbps if demand_mbps is not None else None,
    }
    return out
