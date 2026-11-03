"""
Milestone 2: synchronous client-selection strategies.

Two of the four required comparison arms live here:

    random_selection    uniformly random subset S* of size K  (naive baseline)
    adaptive_selection  the team's proposed score from the project brief:

        Score_i = alpha * (1 / T_i) + beta * D_i - gamma * C_i

    where T_i is the client's estimated compute time for the round, D_i is
    how useful its local data is, and C_i is its communication cost.

    We evaluate it *greedily over the set*, not per client: the speed term is
    the round time the pick would force (a synchronous round waits for its
    slowest member), and D_i is the per-class data the pick adds that the
    already-chosen clients do not have. Scoring clients independently made
    the selector pick the biggest and the fastest client every round, which
    under a non-IID split left whole digit classes out of most rounds. See
    adaptive_selection().

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
    D_i: per-class data mass, {client_id: array[num_classes]} where entry k
    is the fraction of the *whole* federation's class-k samples that client i
    holds. Rows of a (client x class) coverage matrix, so the selector can
    score what a client adds to an already-chosen set rather than scoring
    clients in isolation. Computed once from the label vectors -- no model,
    no gradients.
    """
    counts = {}
    for client_id, loader in client_loaders.items():
        subset = loader.dataset
        labels = np.asarray(subset.dataset.targets)[np.asarray(subset.indices)]
        counts[client_id] = np.bincount(labels, minlength=num_classes).astype(float)

    total = sum(counts.values())
    if total.sum() == 0:
        raise ValueError("client_loaders contain no samples")

    return {cid: c / np.maximum(total, 1.0) for cid, c in counts.items()}


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
    delta=0.5,
    cover_floor=None,
):
    """
    Greedy set-aware top-K.

    Score_i = alpha * speed(S + i) + beta * gain(S + i) - gamma * C_i + delta * age_i

    where S is the set chosen so far and both set terms are marginal:

      speed(S + i) = T_fastest / max(T_j for j in S + i)   -- the round time
                     the pick would *force*, since a synchronous round waits
                     for its slowest member; a slow client is scored by the
                     round it costs, not by its own 1/T.
      gain(S + i)  = coverage(S + i) - coverage(S), with
                     coverage(S) = sum_k sqrt(sum_{j in S} D_j[k]). Concave in
                     class mass, so the second client holding the same digits
                     is worth far less than the first. Normalised so the best
                     single first pick scores 1.0.

    est_time:     {client_id: estimated seconds for this client's round}.
                  Must be positive; a client that has never reported should
                  not be scored (round 1 selects everyone instead).
    data_profile: {client_id: D_i} from client_data_profile(). A per-class
                  vector, or a scalar (as the distributed server sends) which
                  is treated as a one-class vector.
    age:          {client_id: rounds since this client was last selected}.
    comm_cost:    {client_id: C_i}; defaults to 1.0 for every client, which
                  is correct here -- see the module docstring.
    cover_floor:  None -> exactly K clients (fixed-K top-K). A fraction in
                  (0, 1] -> K becomes a *cap* and the greedy loop stops as
                  soon as every class has at least cover_floor of its mass
                  in the chosen set. Under a non-IID split a fixed K can be
                  structurally unable to cover every class (seed 42, 5
                  clients: four digits each live on a single client, so K=3
                  drops a digit every round and the model forgets it). The
                  selector picks as many clients as the data needs, no more.

    Returns the selected ids, sorted.
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

    mass = {
        cid: np.atleast_1d(np.asarray(data_profile.get(cid, 0.0), dtype=float))
        for cid in client_ids
    }
    fastest = min(est_time[cid] for cid in client_ids)
    gain_scale = max(max(np.sqrt(m).sum() for m in mass.values()), 1e-12)

    chosen = []
    covered = np.zeros_like(next(iter(mass.values())))
    slowest = 0.0
    base_cov = 0.0
    while len(chosen) < K:
        if cover_floor is not None and chosen and (covered >= cover_floor).all():
            break
        best, best_score = None, -math.inf
        for cid in client_ids:
            if cid in chosen:
                continue
            speed = fastest / max(slowest, est_time[cid])
            gain = (np.sqrt(covered + mass[cid]).sum() - base_cov) / gain_scale
            score = (
                alpha * speed
                + beta * gain
                - gamma * comm_cost.get(cid, 1.0)
                + delta * age.get(cid, 0)
            )
            if score > best_score or (score == best_score and cid < best):
                best, best_score = cid, score
        chosen.append(best)
        covered = covered + mass[best]
        base_cov = np.sqrt(covered).sum()
        slowest = max(slowest, est_time[best])

    return sorted(chosen)


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
    flat_d = {cid: np.full(10, 0.1) for cid in ids}
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
        data_profile={1: np.full(10, 0.5), 2: np.zeros(10), 3: np.full(10, 0.5),
                      4: np.zeros(10), 5: np.zeros(10)},
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

    # Set-aware: two clients holding the same digits should not both win
    # over a client that brings new ones, even if it is smaller.
    same_t = {cid: 1.0 for cid in ids}
    cov = np.zeros((5, 10))
    cov[0, :5] = 0.5   # client 1: digits 0-4
    cov[1, :5] = 0.5   # client 2: the same digits 0-4
    cov[2, 5:] = 0.2   # client 3: digits 5-9, less data
    coverage_first = adaptive_selection(
        ids, 2, est_time=same_t,
        data_profile={cid: cov[i] for i, cid in enumerate(ids)},
        age=no_age, alpha=0.0, delta=0.0,
    )
    assert coverage_first == [1, 3], coverage_first

    # Speed is the round the pick forces: once a slow client is in the set,
    # a slightly-slower one costs nothing extra and its data decides.
    forced = adaptive_selection(
        ids, 2,
        est_time={1: 10.0, 2: 1.0, 3: 11.0, 4: 2.0, 5: 30.0},
        data_profile={1: np.full(10, 0.5), 2: np.zeros(10), 3: np.full(10, 0.5),
                      4: np.zeros(10), 5: np.zeros(10)},
        age=no_age, alpha=1.0, beta=5.0, delta=0.0,
    )
    assert forced == [1, 3], forced

    # A scalar profile (what the distributed server sends) still works.
    scalar = adaptive_selection(
        ids, 2, est_time=same_t,
        data_profile={1: 0.9, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
        age=no_age, alpha=0.0, delta=0.0,
    )
    assert scalar[0] == 1, scalar

    # An unprofiled client is refused rather than silently scored as infinite.
    try:
        adaptive_selection(
            ids, 2, est_time={1: 1.0}, data_profile=flat_d, age=no_age
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unprofiled client")

    # cover_floor grows K until every class is held: digits 0-4 need client
    # 1 or 2, digits 5-9 need client 3, so coverage stops at two clients even
    # though the cap allows five.
    grown = adaptive_selection(
        ids, 5, est_time=same_t,
        data_profile={cid: cov[i] for i, cid in enumerate(ids)},
        age=no_age, alpha=0.0, delta=0.0, cover_floor=0.1,
    )
    assert grown == [1, 3], grown
    # ...and a floor nobody can reach just runs to the cap.
    assert len(adaptive_selection(
        ids, 4, est_time=same_t,
        data_profile={cid: cov[i] for i, cid in enumerate(ids)},
        age=no_age, cover_floor=0.99,
    )) == 4

    # Jain's index: equal participation is 1.0, single-client is 1/N.
    assert abs(jain_fairness({1: 7, 2: 7, 3: 7}) - 1.0) < 1e-12
    assert abs(jain_fairness({1: 9, 2: 0, 3: 0}) - 1 / 3) < 1e-12
    assert jain_fairness({}) == 0.0

    print("selection.py self-check passed")


if __name__ == "__main__":
    _demo()
