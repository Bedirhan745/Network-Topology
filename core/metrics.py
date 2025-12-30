import math
import random
import networkx as nx
from core import metrics


# ---------------------------
# Edge incremental cost (Dijkstra ile tutarlı)
# ---------------------------
def _edge_cost_for_mode(G: nx.Graph, u, v, data, target, mode: str) -> float:
    if mode == "Delay":
        w = float(data["link_delay_ms"])
        if v != target:
            w += float(G.nodes[v]["processing_delay_ms"])
        return w

    if mode == "Reliability":
        r_link = float(data["link_reliability"])
        r_link = min(max(r_link, 1e-12), 1.0)
        w = -math.log(r_link)

        r_node = float(G.nodes[v]["node_reliability"])
        r_node = min(max(r_node, 1e-12), 1.0)
        w += -math.log(r_node)
        return w

    # fallback
    w = float(data["link_delay_ms"])
    if v != target:
        w += float(G.nodes[v]["processing_delay_ms"])
    return w


# ---------------------------
# Bandwidth lookup (Demand constraint)
# ---------------------------
def _bandwidth_mbps(edge_data) -> float:
    # Sizin projede ana alan: bandwidth_mbps
    for k in ("bandwidth_mbps", "capacity_mbps", "bw", "bandwidth", "capacity", "cap"):
        if k in edge_data:
            try:
                return float(edge_data[k])
            except Exception:
                pass
    # alan yoksa kısıt uygulamayalım (ama sizde var)
    return float("inf")


# ---------------------------
# Objective (mode bazlı)
# ---------------------------
def _path_objective(G, path, mode: str) -> float:
    if mode == "Delay":
        return metrics.total_delay_ms(G, path)
    # ReliabilityCost minimize edilir (daha güvenilir = daha düşük -log toplamı)
    return metrics.reliability_cost(G, path)


# ---------------------------
# MMAS helpers: tau bounds
# ---------------------------
def _compute_tau_min(tau_max: float, p_best: float, n: int) -> float:
    """
    Klasik MMAS tau_min tahmini.
    n burada tipik olarak problem boyutu; biz basitçe path uzunluğunu kullanacağız.
    """
    p_best = max(1e-6, min(0.999999, p_best))
    pb = p_best ** (1.0 / max(1, n))
    denom = ((n / 2.0) - 1.0) * pb
    if denom <= 0:
        return max(1e-16, tau_max * 0.001)
    tau_min = tau_max * (1.0 - pb) / denom
    return max(1e-16, min(tau_max, tau_min))


def solve(
    G: nx.Graph,
    source,
    target,
    mode,
    weights,            # UI signature bozulmasın diye duruyor (şimdilik kullanılmıyor)
    demand_mbps=None,
    seed=42,
    # MMAS params
    ants=60,
    iterations=200,
    alpha=1.0,
    beta=2.5,
    rho=0.2,            # evaporation (MMAS'ta rho kullanılır)
    p_best=0.05,
    update_mode="ib",   # "ib" (iteration-best) veya "gb" (global-best)
    max_steps=120,
):
    """
    MMAS (Max-Min Ant System) ACO Solver.
    - Deterministic: seed sabit.
    - Demand constraint: bandwidth_mbps >= demand_mbps.
    - True MMAS: tau_max/tau_min + clamp + deposit only best.
    """

    if seed is not None:
        random.seed(seed)

    # 1) Pheromone storage for UNDIRECTED edge: store both directions
    tau = {}
    for (u, v) in G.edges():
        tau[(u, v)] = 1.0
        tau[(v, u)] = 1.0

    best_path = []
    best_cost = float("inf")

    # 2) quick init: find an initial feasible best (a few random constructions)
    for _ in range(300):
        p = _construct_path(G, source, target, tau, alpha, beta, mode, demand_mbps, max_steps)
        if not p:
            continue
        c = _path_objective(G, p, mode)
        if c < best_cost:
            best_cost, best_path = c, p

    if not best_path:
        return {"path": [], "metrics": {}}

    # 3) MMAS tau bounds
    tau_max = 1.0 / (rho * best_cost + 1e-12)
    tau_min = _compute_tau_min(tau_max, p_best, n=len(best_path))

    # init all pheromones to tau_max
    for k in list(tau.keys()):
        tau[k] = tau_max

    # 4) Main loop
    for _it in range(iterations):
        iter_best_path = None
        iter_best_cost = float("inf")

        for _a in range(ants):
            p = _construct_path(G, source, target, tau, alpha, beta, mode, demand_mbps, max_steps)
            if not p:
                continue
            c = _path_objective(G, p, mode)

            if c < iter_best_cost:
                iter_best_cost, iter_best_path = c, p

        # update global best
        if iter_best_path and iter_best_cost < best_cost:
            best_cost, best_path = iter_best_cost, iter_best_path
            tau_max = 1.0 / (rho * best_cost + 1e-12)
            tau_min = _compute_tau_min(tau_max, p_best, n=len(best_path))

        # Evaporation
        for k in list(tau.keys()):
            tau[k] *= (1.0 - rho)

        # Deposit ONLY best (MMAS)
        dep_path, dep_cost = (best_path, best_cost) if update_mode.lower() == "gb" else (iter_best_path, iter_best_cost)
        if dep_path:
            delta = 1.0 / (dep_cost + 1e-12)
            for i in range(len(dep_path) - 1):
                u, v = dep_path[i], dep_path[i + 1]
                tau[(u, v)] = tau.get((u, v), tau_min) + delta
                tau[(v, u)] = tau.get((v, u), tau_min) + delta

        # Clamp to [tau_min, tau_max]
        for k in list(tau.keys()):
            if tau[k] < tau_min:
                tau[k] = tau_min
            elif tau[k] > tau_max:
                tau[k] = tau_max

    # Output for UI
    out = {"path": best_path, "metrics": {}}
    if best_path:
        out["metrics"] = {
            "TotalDelay_ms": metrics.total_delay_ms(G, best_path),
            "TotalReliability": metrics.total_reliability(G, best_path),
            "ReliabilityCost": metrics.reliability_cost(G, best_path),
            "ResourceCost": metrics.resource_cost(G, best_path),
            "Objective": best_cost,
            "ObjectiveType": mode,
            "Demand_mbps": demand_mbps,
            "Seed": seed,
            "ACOType": "MMAS",
            "UpdateMode": update_mode,
            "TauMax": tau_max,
            "TauMin": tau_min,
        }
    return out


def _construct_path(G, src, dst, tau, alpha, beta, mode, demand_mbps, max_steps):
    cur = src
    path = [cur]
    visited = {cur}

    for _ in range(max_steps):
        if cur == dst:
            return path

        # candidates: not visited + demand constraint
        candidates = []
        for n in G.neighbors(cur):
            if n in visited:
                continue
            edge_data = G[cur][n]
            bw = _bandwidth_mbps(edge_data)
            if demand_mbps is not None and bw < float(demand_mbps):
                continue
            candidates.append(n)

        if not candidates:
            return None

        # roulette: tau^alpha * eta^beta
        weights = []
        for n in candidates:
            edge_data = G[cur][n]
            c = _edge_cost_for_mode(G, cur, n, edge_data, dst, mode)
            eta = 1.0 / (c + 1e-9)
            w = (tau.get((cur, n), 1.0) ** alpha) * (eta ** beta)
            weights.append(w)

        s = sum(weights)
        if s <= 0:
            nxt = random.choice(candidates)
        else:
            r = random.random() * s
            acc = 0.0
            nxt = candidates[-1]
            for n, w in zip(candidates, weights):
                acc += w
                if acc >= r:
                    nxt = n
                    break

        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return None

# ==========================================================
# FINAL COMPATIBILITY LAYER (UI + ALL SOLVERS)
# ==========================================================

import math


def total_delay_ms(G, path):
    if not path or len(path) < 2:
        return float("inf")

    link_sum = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        link_sum += float(G.edges[u, v]["link_delay_ms"])

    proc_sum = 0.0
    for k in path[1:-1]:
        proc_sum += float(G.nodes[k]["processing_delay_ms"])

    return link_sum + proc_sum


def total_reliability(G, path):
    if not path:
        return 0.0

    rel = 1.0
    for n in path:
        rel *= float(G.nodes[n]["node_reliability"])

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rel *= float(G.edges[u, v]["link_reliability"])

    return rel


def reliability_cost(G, path):
    """
    ReliabilityCost(P) = Σ -log(LinkReliability) + Σ -log(NodeReliability)
    (minimize edilir)
    """
    if not path:
        return float("inf")

    cost = 0.0

    for n in path:
        r = float(G.nodes[n]["node_reliability"])
        r = min(max(r, 1e-12), 1.0)
        cost += -math.log(r)

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        r = float(G.edges[u, v]["link_reliability"])
        r = min(max(r, 1e-12), 1.0)
        cost += -math.log(r)

    return cost


def resource_cost(G, path):
    """
    ResourceCost(P) = Σ (1000 / bandwidth_mbps)
    """
    if not path or len(path) < 2:
        return float("inf")

    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        bw = float(G.edges[u, v].get("bandwidth_mbps", 1e-9))
        bw = max(bw, 1e-9)
        cost += (1000.0 / bw)

    return cost
