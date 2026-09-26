import numpy as np


MU = 0.15


class HeterogeneitySimulator:

    def __init__(self, num_clients, heterogeneity="homo", seed=None):
        if num_clients <= 0:
            raise ValueError("num_clients must be greater than 0")

        if heterogeneity not in {"homo", "normal", "exp"}:
            raise ValueError(
                "heterogeneity must be 'homo', 'normal', or 'exp'"
            )

        self.num_clients = num_clients
        self.heterogeneity = heterogeneity
        self.rng = np.random.default_rng(seed)

        self.base_times = self._generate_base_times()

    def _generate_base_times(self):
        if self.heterogeneity == "homo":
            return np.full(self.num_clients, MU)

        if self.heterogeneity == "normal":
            std = 0.3 * MU
            times = self.rng.normal(MU, std, self.num_clients)
            return np.maximum(times, 0.001)

        return self.rng.exponential(MU, self.num_clients)

    def get_round_duration(self, round_k, Q):
        if Q <= 0:
            raise ValueError("Q must be greater than 0")

        round_step_times = self.rng.normal(
            self.base_times,
            0.05 * self.base_times
        )

        round_step_times = np.maximum(round_step_times, 0.001)

        completion_times = Q * round_step_times
        round_duration = np.max(completion_times)

        return completion_times, round_duration