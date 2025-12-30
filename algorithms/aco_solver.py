import math
import random
import networkx as nx
from core import metrics

DEFAULT_SEED = 42

# (Aynı S-D-mode-demand tekrar tekrar istenirse anında dönsün)
_CACHE = {}


# ---------------------------
# SAFE metric calls (ACO dışını etkilemez)
# ---------------------------
def _safe_total_delay(G, path):
    if hasattr(metrics, "total_delay_ms"):
        return metrics.total_delay_ms(G, path)
    if hasattr(metrics, "total_delay"):
        return metrics.total_delay(G, path)
    # local fallback
    link_sum = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        link_sum += float(G.edges[u, v]["link_delay_ms"])
    proc_sum = 0.0
    for k in path[1:-1]:
        proc_sum += float(G.nodes[k]["processing_delay_ms"])
    return link_sum + proc_sum


def _safe_total_reliability(G, path):
    if hasattr(metrics, "total_reliability"):
        return metrics.total_reliability(G, path)
    rel = 1.0
    for n in path:
        rel *= float(G.nodes[n]["node_reliability"])
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rel *= float(G.edges[u, v]["link_reliability"])
    return rel


def _safe_reliability_cost(G, path):
    if hasattr(metrics, "reliability_cost"):
        return metrics.reliability_cost(G, path)
    cost = 0.0
    for n in path:
        r = min(max(float(G.nodes[n]["node_reliability"]), 1e-12), 1.0)
        cost += -math.log(r)
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        r = min(max(float(G.edges[u, v]["link_reliability"]), 1e-12), 1.0)
        cost += -math.log(r)
    return cost


def _safe_resource_cost(G, path):
    if hasattr(metrics, "resource_cost"):
        return metrics.resource_cost(G, path)
    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        bw = float(G.edges[u, v]["bandwidth_mbps"])
        bw = max(bw, 1e-9)
        cost += (1000.0 / bw)
    return cost


# ---------------------------
# Edge incremental cost (Dijkstra ile tutarlı)
# ---------------------------
def _edge_cost_for_mode(G: nx.Graph, u, v, data, target, mode: str) -> float:
    if mode == "Delay":
        w = float(data["link_delay_ms"])
        # intermediate node processing delay (target hariç)
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

    # fallback Delay
    w = float(data["link_delay_ms"])
    if v != target:
        w += float(G.nodes[v]["processing_delay_ms"])
    return w


def _bandwidth_mbps(edge_data) -> float:
    if "bandwidth_mbps" in edge_data:
        return float(edge_data["bandwidth_mbps"])
    for k in ("capacity_mbps", "bw", "bandwidth", "capacity", "cap"):
        if k in edge_data:
            return float(edge_data[k])
    return float("inf")


# ---------------------------
# MMAS tau_min helper
# ---------------------------
def _compute_tau_min(tau_max: float, p_best: float, n: int) -> float:
    p_best = max(1e-6, min(0.999999, p_best))
    n = max(2, n)
    pb = p_best ** (1.0 / n)
    denom = ((n / 2.0) - 1.0) * pb
    if denom <= 0:
        return max(1e-16, tau_max * 0.001)
    tau_min = tau_max * (1.0 - pb) / denom
    return max(1e-16, min(tau_max, tau_min))


# ---------------------------
# Construct path + incremental objective cost (HIZ İÇİN)
# returns (path or None, cost)
# ---------------------------
def _construct_path(G, src, dst, tau, alpha, beta, mode, demand_mbps, max_steps):
    cur = src
    path = [cur]
    visited = {cur}
    total_cost = 0.0

    for _ in range(max_steps):
        if cur == dst:
            return path, total_cost

        candidates = []
        cand_weights = []

        # candidate list + roulette weights
        for n in G.neighbors(cur):
            if n in visited:
                continue
            edge_data = G[cur][n]

            # demand constraint
            bw = _bandwidth_mbps(edge_data)
            if demand_mbps is not None and bw < float(demand_mbps):
                continue

            inc = _edge_cost_for_mode(G, cur, n, edge_data, dst, mode)
            eta = 1.0 / (inc + 1e-9)
            w = (tau.get((cur, n), 1.0) ** alpha) * (eta ** beta)

            candidates.append((n, inc))
            cand_weights.append(w)

        if not candidates:
            return None, float("inf")

        s = sum(cand_weights)
        if s <= 0:
            nxt, inc = random.choice(candidates)
        else:
            r = random.random() * s
            acc = 0.0
            nxt, inc = candidates[-1]
            for (n, inc_i), w in zip(candidates, cand_weights):
                acc += w
                if acc >= r:
                    nxt, inc = n, inc_i
                    break

        path.append(nxt)
        visited.add(nxt)
        total_cost += inc
        cur = nxt

    return None, float("inf")


# ---------------------------
# FINAL SOLVER: Fast->Refine + EarlyStop + 1 Restart + MMAS clamp
# ---------------------------
def solve(G, source, target, mode, weights, demand_mbps=None):
    # Cache (aynı query tekrar gelirse anında dön)
    cache_key = (source, target, mode, demand_mbps)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    random.seed(DEFAULT_SEED)

    # --- "Hız + kalite" parametreleri ---
    # Phase-1 (FAST)
    fast_ants = 12
    fast_iters = 20

    # Phase-2 (REFINE)
    ants = 18
    iterations = 60  # total refine iterations

    # Common
    alpha = 1.0
    beta = 2.0
    rho = 0.25
    p_best = 0.05
    max_steps = 70

    # Early stopping / restart
    early_stop_patience = 12     # 12 iter iyileşme yoksa dur
    restart_no_improve = 20      # 20 iter iyileşme yoksa 1 kere restart
    max_restarts = 1
    improve_eps = 1e-9

    # pheromone (undirected -> both directions)
    tau = {}
    for (u, v) in G.edges():
        tau[(u, v)] = 1.0
        tau[(v, u)] = 1.0

    # ---- initial best (az deneme: hızlı) ----
    best_path = []
    best_cost = float("inf")
    for _ in range(60):
        p, c = _construct_path(G, source, target, tau, alpha, beta, mode, demand_mbps, max_steps)
        if p and c < best_cost:
            best_cost, best_path = c, p

    if not best_path:
        out = {"path": [], "metrics": {}}
        _CACHE[cache_key] = out
        return out

    # ---- init MMAS bounds ----
    tau_max = 1.0 / (rho * best_cost + 1e-12)
    tau_min = _compute_tau_min(tau_max, p_best, n=len(best_path))
    for k in list(tau.keys()):
        tau[k] = tau_max

    def _mmas_step(step_ants: int):
        nonlocal best_cost, best_path, tau_max, tau_min

        iter_best_path = None
        iter_best_cost = float("inf")

        for _a in range(step_ants):
            p, c = _construct_path(G, source, target, tau, alpha, beta, mode, demand_mbps, max_steps)
            if not p:
                continue
            if c < iter_best_cost:
                iter_best_cost, iter_best_path = c, p

        # update global best
        improved = False
        if iter_best_path and iter_best_cost + improve_eps < best_cost:
            best_cost, best_path = iter_best_cost, iter_best_path
            tau_max = 1.0 / (rho * best_cost + 1e-12)
            tau_min = _compute_tau_min(tau_max, p_best, n=len(best_path))
            improved = True

        # evaporation
        for kk in list(tau.keys()):
            tau[kk] *= (1.0 - rho)

        # deposit ONLY iteration-best
        if iter_best_path:
            delta = 1.0 / (iter_best_cost + 1e-12)
            for i in range(len(iter_best_path) - 1):
                u, v = iter_best_path[i], iter_best_path[i + 1]
                tau[(u, v)] = tau.get((u, v), tau_min) + delta
                tau[(v, u)] = tau.get((v, u), tau_min) + delta

        # clamp
        for kk in list(tau.keys()):
            if tau[kk] < tau_min:
                tau[kk] = tau_min
            elif tau[kk] > tau_max:
                tau[kk] = tau_max

        return improved

    # ---- Phase 1: FAST ----
    no_improve = 0
    for _ in range(fast_iters):
        improved = _mmas_step(fast_ants)
        no_improve = 0 if improved else (no_improve + 1)
        if no_improve >= early_stop_patience:
            break

    # ---- Phase 2: REFINE (with 1 restart) ----
    no_improve = 0
    restarts_done = 0
    for _ in range(iterations):
        improved = _mmas_step(ants)
        no_improve = 0 if improved else (no_improve + 1)

        # restart if stuck
        if no_improve >= restart_no_improve and restarts_done < max_restarts:
            restarts_done += 1
            no_improve = 0
            # reset tau closer to tau_max (escape local optimum)
            for kk in list(tau.keys()):
                # small randomness but deterministic due to seed
                tau[kk] = tau_max * (0.5 + 0.5 * random.random())
            # clamp after reset
            for kk in list(tau.keys()):
                if tau[kk] < tau_min:
                    tau[kk] = tau_min
                elif tau[kk] > tau_max:
                    tau[kk] = tau_max

        if no_improve >= early_stop_patience:
            break

    # ---- final metrics (sadece 1 kez) ----
    out = {"path": best_path, "metrics": {}}
    if best_path:
        out["metrics"] = {
            "TotalDelay_ms": _safe_total_delay(G, best_path),
            "TotalReliability": _safe_total_reliability(G, best_path),
            "ReliabilityCost": _safe_reliability_cost(G, best_path),
            "ResourceCost": _safe_resource_cost(G, best_path),
            "Objective": best_cost,          # mode'a göre incremental objective
            "ObjectiveType": mode,
            "Demand_mbps": demand_mbps,
            "Seed": DEFAULT_SEED,
            "ACOType": "MMAS-FastRefine",
            "Restarts": restarts_done,
        }

    _CACHE[cache_key] = out
    return out
