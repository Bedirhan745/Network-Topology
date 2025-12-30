from algorithms import dijkstra_solver, aco_solver, QLearning

def _not_implemented_solver(name: str):
    def _solve(*args, **kwargs):
        raise NotImplementedError(f"{name} henüz entegre edilmedi. (Şimdilik sadece Dijkstra çalışıyor)")
    return _solve

SOLVERS = {
    "Dijkstra": dijkstra_solver.solve,
    "ACO": aco_solver.solve,
    "Q-Learning": QLearning.solve,
}