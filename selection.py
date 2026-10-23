"""
Milestone 2: synchronous client-selection strategies.

Two of the four required comparison arms live here:

    random_selection    uniformly random subset S* of size K  (naive baseline)
    adaptive_selection  the team's proposed score from the project brief:

        Score_i = alpha * (1 / T_i) + beta * D_i - gamma * C_i

    where T_i is the client's estimated compute time for the round, D_i is
    how useful its local data is, and C_i is its communication cost.

The other two arms are FedAvg (select everyone, `fedavg.py`) and FedCompass
(semi-asynchronous, `compass.py`).

A note on the three terms, because two of them are degenerate on this
benchmark and pretending otherwise would be dishonest:

  * C_i is identical for every client here -- the same CNN is shipped both
    ways regardless of who trains it -- so after normalisation C_i == 1.0 and
    the `- gamma * C_i` term shifts every score by the same constant. It
    therefore cannot change a top-K ranking. It is kept because the brief
    specifies it and because it *does* bite in the `--budget` variant, where
    K is grown until the communication budget is spent rather than fixed.
  * T_i is unknown before a client has ever reported, so round 1 is a
    profiling round that selects everybody. From round 2 on, T_i comes from
    an EMA of observed completion times -- the server is never handed the
    simulator's ground-truth speeds.

`delta * age_i` is ours, not the brief's. Pure top-K by speed permanently
starves the slow clients, which both destroys the participation-fairness
number the brief asks us to report and (worse) throws away their data under
a non-IID split. The age term is the cheapest fix that keeps the selector
greedy while guaranteeing no client is starved forever.
"""

import math

import numpy as np


def client_data_profile(client_loaders, num_classes=10):
    """
    D_i: "useful local data", normalised to (0, 1].

    Useful is taken to mean *both* plentiful and diverse: a client holding
    30% of the data but only one digit class is far less useful to a global
    MNIST model than one holding 10% spread over all ten. So

        D_i_raw = (n_i / n_total) * (H(labels_i) / log(num_classes))

    the data share scaled by the normalised label entropy. Computed once from
    the label vectors -- no model, no gradients, so it costs nothing per round.
    """
    raw = {}

    for client_id, loader in client_loaders.items():
        subset = loader.dataset
        labels = np.asarray(subset.dataset.targets)[np.asarray(subset.indices)]

        counts = np.bincount(labels, minlength=num_classes).astype(float)
        n_i = counts.sum()

        if n_i == 0:
            raw[client_id] = 0.0
            continue

        probabilities = counts[counts > 0] / n_i
        entropy = float(-(probabilities * np.log(probabilities)).sum())

        raw[client_id] = n_i * (entropy / math.log(num_classes))

    total = sum(
        len(loader.dataset) for loader in client_loaders.values()
    )
    if total == 0:
        raise ValueError("client_loaders contain no samples")

    share = {cid: value / total for cid, value in raw.items()}

    peak = max(share.values())
    if peak <= 0:
        # Every client holds a single class: entropy is 0 everywhere, so the
        # diversity signal carries no information. Fall back to data share.
        return {
            cid: len(client_loaders[cid].dataset) / total
            for cid in client_loaders
        }

    return {cid: value / peak for cid, value in share.items()}


def random_selection(client_ids, K, rng):
    """Uniformly random subset of size K. The naive baseline arm."""
    K = _check_k(K, len(client_ids))
    return sorted(rng.choice(list(client_ids), size=K, replace=False).tolist())


def adaptive_selection(
    client_ids,
    K,
    est_time,
    data_profile,
    age,
    comm_cost=None,
    alpha=1.0,
    beta=1.0,
    gamma=0.1,
    delta=0.15,
):
    """
    Top-K by Score_i = alpha/T_i + beta*D_i - gamma*C_i + delta*age_i.

    est_time:     {client_id: estimated seconds for this client's round}.
                  Must be positive; a client that has never reported should
                  not be scored (round 1 selects everyone instead).
    data_profile: {client_id: D_i} from client_data_profile().
    age:          {client_id: rounds since this client was last selected}.
    comm_cost:    {client_id: C_i}; defaults to 1.0 for every client, which
                  is correct here -- see the module docstring.

    1/T_i is normalised against the fastest client so that alpha, beta, gamma
    and delta are all comparable on a 0-1 scale and the defaults mean
    something. Returns the selected ids, sorted.
    """
    K = _check_k(K, len(client_ids))

    if comm_cost is None:
        comm_cost = {cid: 1.0 for cid in client_ids}

    for cid in client_ids:
        if est_time.get(cid, 0.0) <= 0:
            raise ValueError(
                f"est_time[{cid}] must be positive; client has not been "
                f"profiled yet (round 1 should select all clients)"
            )

    speed_score = {cid: 1.0 / est_time[cid] for cid in client_ids}
    fastest = max(speed_score.values())
    speed_score = {cid: v / fastest for cid, v in speed_score.items()}

    oldest = max(max(age.get(cid, 0) for cid in client_ids), 1)

    scores = {
        cid: (
            alpha * speed_score[cid]
            + beta * data_profile.get(cid, 0.0)
            - gamma * comm_cost.get(cid, 1.0)
            + delta * age.get(cid, 0)
        )
        for cid in client_ids
    }

    ranked = sorted(client_ids, key=lambda cid: (-scores[cid], cid))
    return sorted(ranked[:K])


def jain_fairness(participation):
    """
    Jain's fairness index over per-client participation counts.

    1.0 means every client trained equally often; 1/N means one client did
    all the work. This is the "client participation fairness" number the
    brief asks for.
    """
    counts = np.asarray(list(participation.values()), dtype=float)

    if counts.size == 0 or counts.sum() == 0:
        return 0.0

    return float(counts.sum() ** 2 / (counts.size * (counts ** 2).sum()))


def _check_k(K, num_clients):
    if num_clients == 0:
        raise ValueError("no clients to select from")
    if K is None:
        return num_clients
    if K <= 0:
        raise ValueError("K must be greater than 0")
    return min(K, num_clients)


def _demo():
    rng = np.random.default_rng(0)
    ids = [1, 2, 3, 4, 5]

    # random_selection respects K, stays in range, and never repeats.
    for _ in range(50):
        chosen = random_selection(ids, 3, rng)
        assert len(chosen) == 3, chosen
        assert len(set(chosen)) == 3, chosen
        assert set(chosen) <= set(ids), chosen

    assert random_selection(ids, 99, rng) == ids, "K clamps to client count"

    # Speed dominates when it is the only signal that varies.
    flat_d = {cid: 0.5 for cid in ids}
    no_age = {cid: 0 for cid in ids}
    fast_first = adaptive_selection(
        ids, 2,
        est_time={1: 10.0, 2: 1.0, 3: 20.0, 4: 2.0, 5: 30.0},
        data_profile=flat_d, age=no_age, beta=0.0, delta=0.0,
    )
    assert fast_first == [2, 4], fast_first

    # Data usefulness outranks speed once beta is large enough.
    data_first = adaptive_selection(
        ids, 2,
        est_time={1: 10.0, 2: 1.0, 3: 20.0, 4: 2.0, 5: 30.0},
        data_profile={1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0},
        age=no_age, alpha=1.0, beta=5.0, delta=0.0,
    )
    assert data_first == [1, 3], data_first

    # C_i cannot reorder a top-K when it is uniform -- the documented
    # degeneracy. If this ever fails, C_i stopped being constant.
    with_gamma = adaptive_selection(
        ids, 2,
        est_time={1: 10.0, 2: 1.0, 3: 20.0, 4: 2.0, 5: 30.0},
        data_profile=flat_d, age=no_age, beta=0.0, delta=0.0, gamma=99.0,
    )
    assert with_gamma == fast_first, with_gamma

    # The age term eventually rescues a client that speed alone would starve.
    starved = adaptive_selection(
        ids, 1,
        est_time={1: 100.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
        data_profile=flat_d,
        age={1: 50, 2: 0, 3: 0, 4: 0, 5: 0},
        beta=0.0, delta=10.0,
    )
    assert starved == [1], starved

    # An unprofiled client is refused rather than silently scored as infinite.
    try:
        adaptive_selection(
            ids, 2, est_time={1: 1.0}, data_profile=flat_d, age=no_age
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unprofiled client")

    # Jain's index: equal participation is 1.0, single-client is 1/N.
    assert abs(jain_fairness({1: 7, 2: 7, 3: 7}) - 1.0) < 1e-12
    assert abs(jain_fairness({1: 9, 2: 0, 3: 0}) - 1 / 3) < 1e-12
    assert jain_fairness({}) == 0.0

    print("selection.py self-check passed")


if __name__ == "__main__":
    _demo()
