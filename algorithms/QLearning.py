import math
import random
from typing import Dict, Tuple, List, Any

import networkx as nx
from core import metrics

# =========================
# RANDOM SEED (REPRODUCIBILITY)
# =========================
DEFAULT_SEED = 42


def _edge_cost_for_mode(G: nx.Graph, u, v, data, target, mode: str) -> float:
    """Dijkstra ile aynı maliyet tanımı (Delay / Reliability)."""
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

    # Varsayılan: Delay
    w = float(data["link_delay_ms"])
    if v != target:
        w += float(G.nodes[v]["processing_delay_ms"])
    return w


def _build_q_table(G: nx.Graph) -> Dict[int, Dict[int, float]]:
    """Her düğüm için komşu aksiyonları tutan Q tablosu oluştur."""
    Q: Dict[int, Dict[int, float]] = {}
    for u in G.nodes():
        Q[u] = {v: 0.0 for v in G.neighbors(u)}
    return Q


def _choose_action(state: int, Q: Dict[int, Dict[int, float]], epsilon: float) -> int:
    """Epsilon-greedy aksiyon seçimi."""
    actions = list(Q[state].keys())
    if not actions:
        return state

    if random.random() < epsilon:
        return random.choice(actions)

    max_q = max(Q[state].values())
    best_actions = [a for a, q in Q[state].items() if q == max_q]
    return random.choice(best_actions)


def _q_learning_episode(
    G: nx.Graph,
    source: int,
    target: int,
    mode: str,
    Q: Dict[int, Dict[int, float]],
    alpha: float,
    gamma: float,
    epsilon: float,
) -> Tuple[List[int], float]:
    """
    Tek bir Q-learning episodu çalıştır.
    Geri dönen: (izlenen yol, kümülatif maliyet)
    """
    state = source
    path = [state]
    total_cost = 0.0
    visited = {state}

    for _ in range(200):  # güvenlik için maksimum adım sayısı
        if state == target:
            break

        if state not in Q or not Q[state]:
            break

        action = _choose_action(state, Q, epsilon)
        if action in visited:  # döngüleri engelle
            break

        data = G[state][action]
        step_cost = _edge_cost_for_mode(G, state, action, data, target, mode)

        next_state = action
        visited.add(next_state)
        path.append(next_state)
        total_cost += step_cost

        # Q-güncellemesi
        old_q = Q[state][action]
        if next_state in Q and Q[next_state]:
            max_next_q = max(Q[next_state].values())
        else:
            max_next_q = 0.0

        # Amaç: maliyeti minimize etmek => ödül = -step_cost
        reward = -step_cost
        new_q = (1 - alpha) * old_q + alpha * (reward + gamma * max_next_q)
        Q[state][action] = new_q

        state = next_state

    return path, total_cost


def solve(
    G: nx.Graph,
    source: int,
    target: int,
    mode: str,
    weights: dict,  # kullanılmıyor, arayüz uyumu için
    demand_mbps: float | None = None,
) -> Dict[str, Any]:
    """
    Q-Learning tabanlı yol bulma.

    - Durum: aktif düğüm
    - Aksiyon: komşu düğüme gitmek
    - Ödül: - maliyet (Delay için gecikme, Reliability için ReliabilityCost)
    """

    # =========================
    # SET RANDOM SEED
    # =========================
    random.seed(DEFAULT_SEED)

    if source not in G or target not in G:
        return {"path": [], "metrics": {}}

    # Q-learning parametreleri
    episodes = 10000
    alpha = 0.7
    gamma = 0.8
    epsilon_start = 0.9
    epsilon_end = 0.05

    Q = _build_q_table(G)

    best_path: List[int] = []
    best_cost = float("inf")

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        epsilon = epsilon_start * (1 - frac) + epsilon_end * frac

        path, cost = _q_learning_episode(
            G, source, target, mode, Q, alpha, gamma, epsilon
        )

        if path and path[-1] == target and cost < best_cost:
            best_cost = cost
            best_path = path

    out: Dict[str, Any] = {"path": best_path, "metrics": {}}
    if not best_path:
        return out

    out["metrics"] = {
        "TotalDelay_ms": metrics.total_delay_ms(G, best_path),
        "TotalReliability": metrics.total_reliability(G, best_path),
        "ReliabilityCost": metrics.reliability_cost(G, best_path),
        "ResourceCost": metrics.resource_cost(G, best_path),
        "Objective": (
            metrics.total_delay_ms(G, best_path)
            if mode == "Delay"
            else metrics.reliability_cost(G, best_path)
        ),
        "ObjectiveType": mode,
        "Demand_mbps": demand_mbps,
        "Seed": DEFAULT_SEED,   # <-- raporlanıyor
    }

    return out
