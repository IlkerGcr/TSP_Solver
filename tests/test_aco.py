from tsp.instance import TSPInstance
from solvers.aco import ACOConfig, aco_solve

SQUARE = TSPInstance.from_coords([(0, 0), (0, 1), (1, 1), (1, 0)])
OPTIMAL_SQUARE_COST = 4.0


def make_config(seed):
    return ACOConfig(
        num_ants=4,
        alpha=1.0,
        beta=3.0,
        rho=0.3,
        q=1.0,
        iterations=50,
        use_local_search=True,
        local_search_passes=1,
        log_interval=10,
        seed=seed,
    )


def test_aco_finds_optimal_on_trivial_square():
    result = aco_solve(SQUARE, make_config(seed=1))
    assert abs(result["best_cost"] - OPTIMAL_SQUARE_COST) < 1e-6


def test_aco_is_deterministic_given_same_seed():
    result_a = aco_solve(SQUARE, make_config(seed=99))
    result_b = aco_solve(SQUARE, make_config(seed=99))
    assert result_a["best_cost"] == result_b["best_cost"]
    assert result_a["best_tour"] == result_b["best_tour"]
