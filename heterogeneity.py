import numpy as np


MU = 0.15


class HeterogeneitySimulator:

    def __init__(self, num_clients, heterogeneity="homo", seed=None, client_ids=None):
        if num_clients <= 0:
            raise ValueError("num_clients must be greater than 0")

        if heterogeneity not in {"homo", "normal", "exp"}:
            raise ValueError(
                "heterogeneity must be 'homo', 'normal', or 'exp'"
            )

        if client_ids is None:
            client_ids = list(range(1, num_clients + 1))

        if len(client_ids) != num_clients:
            raise ValueError("client_ids must have length num_clients")

        self.num_clients = num_clients
        self.client_ids = list(client_ids)
        self.heterogeneity = heterogeneity
        self.rng = np.random.default_rng(seed)

        base_times = self._generate_base_times()
        self.base_times = dict(zip(self.client_ids, base_times))

    def _generate_base_times(self):
        if self.heterogeneity == "homo":
            return np.full(self.num_clients, MU)

        if self.heterogeneity == "normal":
            std = 0.3 * MU
            times = self.rng.normal(MU, std, self.num_clients)
            return np.maximum(times, 0.001)

        return self.rng.exponential(MU, self.num_clients)

    def sample_step_times(self, client_ids=None):
        """
        Returns {client_id: seconds for ONE local step} for this draw, with
        5% per-round jitter around each client's base speed.

        Split out of get_round_duration because FedCompass assigns every
        client a different number of local steps Q_i, so a single shared
        round duration is meaningless there -- the scheduler needs the
        per-step cost and multiplies by that client's own Q_i.
        """
        if client_ids is None:
            client_ids = self.client_ids

        base_times = np.array([self.base_times[cid] for cid in client_ids])

        step_times = self.rng.normal(base_times, 0.05 * base_times)
        step_times = np.maximum(step_times, 0.001)

        return dict(zip(client_ids, step_times))

    def get_round_duration(self, round_k, Q, client_ids=None):
        """
        Returns ({client_id: completion_time}, round_duration) for client_ids
        (default: every client this simulator was built with). round_duration
        is the straggler max over exactly that set, so a partial
        client-selection round simulates the timing of only the clients that
        actually trained that round.
        """
        if Q <= 0:
            raise ValueError("Q must be greater than 0")

        if client_ids is None:
            client_ids = self.client_ids

        step_times = self.sample_step_times(client_ids)

        completion_times = {
            cid: Q * step_time for cid, step_time in step_times.items()
        }
        round_duration = max(completion_times.values())

        return completion_times, round_duration