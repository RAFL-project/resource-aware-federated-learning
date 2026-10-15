"""
Milestone 2: FedCompass / COMPASS (COMputing Power-Aware Scheduler).

Paper:     FedCompass, ICLR 2024 -- https://arxiv.org/abs/2309.14675
Reference: github.com/APPFL/FedCompass
           src/appfl/scheduler/compass_scheduler.py
           src/appfl/aggregator/fedcompass_aggregator.py

Unlike the other three arms, this one is *semi-asynchronous*: there are no
global rounds. Each client is assigned its own number of local steps Q_i,
chosen so that clients of differing speed land at the server at roughly the
same moment and can be aggregated together as an "arrival group". Fast
clients do more steps instead of sitting idle waiting for stragglers, which
is where the wall-clock saving over FedAvg comes from.

    Q_i = floor((T_a(g) - t_now) / speed_i)    subject to Q_min <= Q_i <= Q_max

Two deliberate divergences from the reference implementation, both
load-bearing enough to call out:

1. SIMULATED CLOCK, NO THREADS. The reference drives groups with
   `time.time()` and `threading.Timer`. This class takes `now` as an
   argument and reports when its next timeout is due, so the whole thing is
   a deterministic discrete-event state machine: same seed, same trace,
   every run. Benchmark numbers that move when the laptop is busy are not
   benchmark numbers. It also keeps the scheduler free of torch, so the
   self-check at the bottom runs in milliseconds without training anything.

2. STALENESS SIGN. The reference's "polynomial" staleness function is
   `(u + 1) ** a` with a default of a = +0.5, which *grows* with staleness --
   it upweights the most out-of-date updates. The paper says st(.) is there
   to penalise stale models, and FedAsync (where this family of functions
   comes from) defines the polynomial form as `(u + 1) ** -a`. We use the
   decaying form. Pass `staleness_a=-0.5` to reproduce the reference's
   behaviour exactly.

The aggregation rule follows the paper's Algorithm 1 rather than the
reference code: the paper weights each update by the client's data share
p_i, while the reference hardcodes an equal 1/N. p_i is both what the paper
writes and what our FedAvg arm already uses, so it keeps the comparison
honest.
"""

import copy
import heapq
import math
import time

import torch


class CompassScheduler:
    """
    Pure bookkeeping over a simulated clock. Holds no models and no torch
    state -- it decides *who trains for how long* and *which updates get
    aggregated together*; the caller owns the tensors.

    Every entry point takes the current simulated time `now` and returns an
    action dict:

        {
          "aggregate": None | {"client_ids": [...], "staleness": {cid: int}},
          "dispatch":  [(client_id, local_steps), ...],
        }

    "aggregate" means: combine these clients' stored deltas into the global
    model now. "dispatch" means: start these clients training, with these
    step counts, from the current global model.
    """

    def __init__(
        self,
        client_ids,
        max_local_steps=200,
        min_local_steps=40,
        speed_momentum=0.9,
        latest_time_factor=1.5,
    ):
        if max_local_steps <= 0 or min_local_steps <= 0:
            raise ValueError("local step bounds must be positive")
        if min_local_steps > max_local_steps:
            raise ValueError("min_local_steps must not exceed max_local_steps")
        if not 0.0 <= speed_momentum <= 1.0:
            raise ValueError("speed_momentum must be in [0, 1]")
        if latest_time_factor < 1.0:
            raise ValueError("latest_time_factor must be at least 1.0")

        self.client_ids = list(client_ids)
        self.max_local_steps = max_local_steps
        self.min_local_steps = min_local_steps
        self.speed_momentum = speed_momentum
        self.latest_time_factor = latest_time_factor

        self.client_info = {}
        self.arrival_group = {}
        self.group_buffer = {}
        self.general_buffer = set()
        self.timers = {}
        self.group_counter = 0
        self.global_timestamp = 0

        # Diagnostics for the report: how many steps each client was asked
        # for, and how big each aggregation batch was.
        self.assigned_steps_log = []
        self.group_sizes = []
        # Updates discarded because the same client delivered a fresher one
        # before its buffered copy could be aggregated.
        self.superseded_count = 0

    # -- public API ------------------------------------------------------

    def initial_dispatch(self):
        """
        Everyone starts together on max_local_steps. The reference gets this
        implicitly (an unseen client defaults to max_local_steps when its
        first response is scored); here it is explicit.
        """
        return [(cid, self.max_local_steps) for cid in self.client_ids]

    def next_timeout(self):
        """(fire_time, group_idx) of the soonest group deadline, or None."""
        if not self.timers:
            return None
        group_idx = min(self.timers, key=lambda g: (self.timers[g], g))
        return self.timers[group_idx], group_idx

    def on_arrival(self, client_id, now):
        """A client has finished its local training at simulated time `now`."""
        self._record_info(client_id, now)

        group_idx = self.client_info[client_id].get("goa", -1)

        if group_idx == -1:
            # First time we have ever heard from this client: apply its
            # update on its own and immediately put it back to work.
            return self._single_update(client_id, now, buffer=False)

        return self._group_update(client_id, group_idx, now)

    def on_timeout(self, group_idx, now):
        """A group hit its latest arrival time with clients still missing."""
        self.timers.pop(group_idx, None)
        return self._group_aggregation(group_idx, now)

    # -- internals -------------------------------------------------------

    def _record_info(self, client_id, now):
        """Update the EMA estimate of this client's seconds-per-local-step."""
        info = self.client_info.get(client_id)

        start_time = info["start_time"] if info else 0.0
        steps = info["local_steps"] if info else self.max_local_steps

        elapsed = now - start_time
        observed_speed = elapsed / steps

        if observed_speed <= 0:
            raise ValueError(
                f"client {client_id} reported a non-positive step time "
                f"({elapsed}s over {steps} steps)"
            )

        if info is None:
            self.client_info[client_id] = {
                "timestamp": 0,
                "speed": observed_speed,
                "local_steps": self.max_local_steps,
                "start_time": 0.0,
            }
        else:
            # Matches the reference exactly, including the surprising naming:
            # `speed_momentum` is the weight on the NEW sample, not the old
            # one, so the default 0.9 tracks recent speed closely.
            info["speed"] = (
                (1 - self.speed_momentum) * info["speed"]
                + self.speed_momentum * observed_speed
            )

    def _single_update(self, client_id, now, buffer):
        """
        Apply (or buffer) one lone client's update, then re-dispatch it.

        buffer=False -> aggregate it immediately, bumping the global clock.
        buffer=True  -> park it in the general buffer; it will ride along
                        with the next group aggregation. This is the path for
                        a straggler that missed its group's deadline.
        """
        info = self.client_info[client_id]
        aggregate = None

        if buffer:
            self.general_buffer.add(client_id)
        else:
            aggregate = {
                "client_ids": [client_id],
                "staleness": {
                    client_id: self.global_timestamp - info["timestamp"]
                },
                "fresh": [client_id],
                "from_buffer": [],
                "superseded": [],
            }
            self.global_timestamp += 1
            self.group_sizes.append(1)

        info["timestamp"] = self.global_timestamp

        dispatch = self._assign_group(client_id, now)

        # `buffered` matters to the caller: this client is being put straight
        # back to work, so the delta it just handed over has to be moved out
        # of the way before the next one overwrites it.
        return {
            "aggregate": aggregate,
            "dispatch": dispatch,
            "buffered": client_id if buffer else None,
        }

    def _group_update(self, client_id, group_idx, now):
        group = self.arrival_group[group_idx]

        if now > group["latest_arrival_time"]:
            # Too late -- the group has already aggregated without us.
            group["clients"].remove(client_id)
            if not group["clients"] and not group["arrived_clients"]:
                del self.arrival_group[group_idx]
                self.timers.pop(group_idx, None)
            return self._single_update(client_id, now, buffer=True)

        group["clients"].remove(client_id)
        group["arrived_clients"].append(client_id)
        self.group_buffer.setdefault(group_idx, set()).add(client_id)

        if group["clients"]:
            # Still waiting on groupmates. This client idles -- no dispatch.
            return {"aggregate": None, "dispatch": []}

        return self._group_aggregation(group_idx, now)

    def _group_aggregation(self, group_idx, now):
        """
        Aggregate a group's buffered updates together with whatever is
        sitting in the general buffer, then put the arrived clients back to
        work fastest-first.
        """
        group = self.arrival_group.get(group_idx)
        buffered = self.group_buffer.get(group_idx)

        if group is None or not buffered:
            # Deadline fired but nobody arrived; nothing to aggregate. The
            # reference no-ops here too.
            return {"aggregate": None, "dispatch": []}

        # A client can be in BOTH buffers: it missed an earlier group (so its
        # update went to the general buffer), got re-dispatched, and has now
        # arrived with a second, fresher update. Both buffers are keyed by
        # client, so only one update per client can be aggregated. The fresh
        # one wins and the general-buffer copy is superseded -- the reference
        # does the same thing, implicitly, by merging the group buffer over
        # the general buffer in a dict.
        superseded = sorted(self.general_buffer & buffered)
        from_buffer = sorted(self.general_buffer - buffered)
        fresh = sorted(buffered)
        participants = sorted(set(fresh) | set(from_buffer))

        staleness = {
            cid: self.global_timestamp - self.client_info[cid]["timestamp"]
            for cid in participants
        }

        aggregate = {
            "client_ids": participants,
            "staleness": staleness,
            # Which store the caller should take each delta from.
            "fresh": fresh,
            "from_buffer": from_buffer,
            "superseded": superseded,
        }

        self.superseded_count += len(superseded)
        self.general_buffer = set()
        del self.group_buffer[group_idx]
        self.timers.pop(group_idx, None)

        self.global_timestamp += 1
        self.group_sizes.append(len(participants))

        arrived = list(group["arrived_clients"])
        for cid in arrived:
            self.client_info[cid]["timestamp"] = self.global_timestamp

        # Zeroing the times retires this group: _join_group now computes a
        # negative step count for it and _create_group skips it, so the
        # clients below cannot re-join the group they just left. Any
        # straggler still in group["clients"] will take the late path.
        group["expected_arrival_time"] = 0.0
        group["latest_arrival_time"] = 0.0
        group["arrived_clients"] = []

        # Fastest first: the quickest client founds the next group and the
        # slower ones size their step counts to land alongside it.
        arrived.sort(key=lambda cid: self.client_info[cid]["speed"])

        dispatch = []
        for cid in arrived:
            dispatch.extend(self._assign_group(cid, now))

        if not group["clients"]:
            self.arrival_group.pop(group_idx, None)

        return {"aggregate": aggregate, "dispatch": dispatch}

    def _assign_group(self, client_id, now):
        if not self.arrival_group:
            self._open_group(client_id, now, self.max_local_steps)
        elif not self._join_group(client_id, now):
            self._create_group(client_id, now)

        steps = self.client_info[client_id]["local_steps"]
        self.assigned_steps_log.append((client_id, steps))
        return [(client_id, steps)]

    def _join_group(self, client_id, now):
        """Land in an existing group if some Q_i in bounds gets us there."""
        speed = self.client_info[client_id]["speed"]

        best_group = -1
        best_steps = -1

        for group_idx, group in self.arrival_group.items():
            remaining = group["expected_arrival_time"] - now
            steps = math.floor(remaining / speed)

            if (
                steps < self.min_local_steps
                or steps > self.max_local_steps
                or steps < best_steps
            ):
                continue

            best_group = group_idx
            best_steps = steps

        if best_group == -1:
            return False

        self.arrival_group[best_group]["clients"].append(client_id)
        self.client_info[client_id]["goa"] = best_group
        self.client_info[client_id]["local_steps"] = best_steps
        self.client_info[client_id]["start_time"] = now
        return True

    def _create_group(self, client_id, now):
        """
        No group is reachable, so start one -- but size it so that this
        client is likely to arrive just as some existing group's *next*
        cycle completes, which lets the two merge later instead of
        fragmenting the schedule further.
        """
        speed = self.client_info[client_id]["speed"]
        assigned = -1

        for group in self.arrival_group.values():
            if now >= group["latest_arrival_time"]:
                continue

            members = group["clients"] + group["arrived_clients"]
            if not members:
                continue

            fastest = min(self.client_info[cid]["speed"] for cid in members)

            est_arrival = (
                group["latest_arrival_time"] + fastest * self.max_local_steps
            )
            steps = math.floor((est_arrival - now) / speed)

            if steps <= self.max_local_steps:
                assigned = max(assigned, steps)

        if assigned < 0:
            assigned = self.max_local_steps
        elif assigned < self.min_local_steps:
            assigned = self.min_local_steps

        self._open_group(client_id, now, assigned)

    def _open_group(self, client_id, now, steps):
        speed = self.client_info[client_id]["speed"]
        group_idx = self.group_counter
        self.group_counter += 1

        expected = now + steps * speed

        self.arrival_group[group_idx] = {
            "clients": [client_id],
            "arrived_clients": [],
            "expected_arrival_time": expected,
            "latest_arrival_time": now
            + steps * speed * self.latest_time_factor,
        }
        self.timers[group_idx] = self.arrival_group[group_idx][
            "latest_arrival_time"
        ]

        self.client_info[client_id]["goa"] = group_idx
        self.client_info[client_id]["local_steps"] = steps
        self.client_info[client_id]["start_time"] = now


def polynomial_staleness(a=0.5):
    """
    FedAsync's polynomial staleness weight, (u + 1) ** -a: an update that is
    u global steps out of date is discounted. See the module docstring for
    why the sign differs from the reference implementation -- pass a negative
    `a` to get the reference's behaviour back.
    """
    return lambda u: float((u + 1) ** -a)


def apply_update(global_model, deltas, weights, staleness, staleness_fn, alpha):
    """
    Paper Algorithm 1/3:  w := w - sum_i alpha * st(u_i) * p_i * Delta_i

    deltas[i] is `w_before - w_after` from model.local_train, i.e. already a
    descent direction, so it is subtracted.

    ponytail: no normalisation by Q_i. Clients run different step counts, so
    a 200-step delta is mechanically larger than a 40-step one and carries
    proportionally more weight here. The reference implementation has the
    same property (it threads `local_steps` into the aggregator and then
    never reads it). If the Q_min/Q_max spread is widened much beyond the
    5:1 default, divide each delta by its step count here.
    """
    global_state = global_model.state_dict()

    with torch.no_grad():
        for name in global_state:
            if not global_state[name].is_floating_point():
                # Integer buffers (e.g. num_batches_tracked) cannot take a
                # weighted update; leave them on the server's copy.
                continue

            accumulated = torch.zeros_like(global_state[name])

            for cid in deltas:
                coefficient = (
                    alpha * staleness_fn(staleness[cid]) * weights[cid]
                )
                accumulated += coefficient * deltas[cid][name]

            global_state[name] -= accumulated

    global_model.load_state_dict(global_state)
    return global_model


def run_fedcompass(
    global_model,
    client_loaders,
    client_weights,
    local_train,
    test_loader,
    heterogeneity_simulator,
    time_budget,
    evaluate_model,
    evaluate_train_loss,
    max_local_steps=200,
    min_local_steps=40,
    speed_momentum=0.9,
    latest_time_factor=1.5,
    staleness_a=0.5,
    alpha=0.9,
    lr=0.003,
    device="cpu",
    eval_every=1,
    verbose=True,
):
    """
    Discrete-event FedCompass over a simulated clock, run until
    `time_budget` simulated seconds have elapsed.

    The budget replaces FedAvg's round count: a semi-asynchronous algorithm
    has no rounds to count, and equal simulated wall-clock is the only axis
    on which the four arms can be compared fairly.

    eval_every: evaluate once every N global aggregations (the final one is
    always evaluated). FedCompass aggregates far more often than FedAvg
    completes rounds -- with a small Q it can be hundreds of times -- and
    scoring the full 60k train set on each one costs more than the training
    does. Raise this when Q is small.

    Returns (global_model, history, stats). Each history row is one
    *evaluated* global aggregation.
    """
    if time_budget <= 0:
        raise ValueError("time_budget must be greater than 0")
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1")
    if client_loaders.keys() != client_weights.keys():
        raise ValueError(
            "client_loaders and client_weights must have the same client ids"
        )

    global_model = global_model.to(device)
    client_ids = list(client_loaders.keys())

    scheduler = CompassScheduler(
        client_ids,
        max_local_steps=max_local_steps,
        min_local_steps=min_local_steps,
        speed_momentum=speed_momentum,
        latest_time_factor=latest_time_factor,
    )
    staleness_fn = polynomial_staleness(staleness_a)

    pending_deltas = {}
    # Deltas already handed to the scheduler's general buffer. They must live
    # apart from pending_deltas: a buffered client is re-dispatched
    # immediately, and that new training would otherwise clobber the update
    # still waiting to be aggregated.
    held_deltas = {}
    participation = {cid: 0 for cid in client_ids}
    events = []
    counter = 0
    comm_transfers = 0
    busy_time = {cid: 0.0 for cid in client_ids}
    selection_overhead = 0.0
    history = []
    num_updates = 0
    last_update = None

    def record(update):
        """Evaluate the global model and append one history row."""
        train_loss = evaluate_train_loss(
            global_model, client_loaders, device=device
        )
        test_loss, accuracy = evaluate_model(
            global_model, test_loader, device=device
        )

        history.append({
            "round": num_updates,
            "total_wall_clock_sec": update["sim_time"],
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "num_selected": len(update["participants"]),
            "comm_transfers": comm_transfers,
            "mean_staleness": (
                sum(update["staleness"].values()) / len(update["staleness"])
            ),
        })

        if verbose:
            print(
                f"Update {num_updates} | "
                f"Time: {update['sim_time']:.2f}s | "
                f"Clients: {len(update['participants'])} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Accuracy: {accuracy:.2f}%"
            )

    def dispatch(batch, now):
        """Train each client from the CURRENT global model, queue its arrival."""
        nonlocal counter, comm_transfers

        for client_id, steps in batch:
            local_model = copy.deepcopy(global_model)
            pending_deltas[client_id] = local_train(
                local_model,
                client_loaders[client_id],
                Q=steps,
                lr=lr,
                device=device,
            )

            step_time = heterogeneity_simulator.sample_step_times(
                [client_id]
            )[client_id]
            duration = steps * step_time

            busy_time[client_id] += duration
            participation[client_id] += 1
            # One model down to the client, one update back up.
            comm_transfers += 2

            counter += 1
            heapq.heappush(events, (now + duration, counter, client_id))

    dispatch(scheduler.initial_dispatch(), 0.0)

    sim_time = 0.0

    while True:
        timeout = scheduler.next_timeout()
        arrival = events[0][0] if events else None

        if timeout is None and arrival is None:
            break

        # Ties go to the arrival: a client that lands exactly on the deadline
        # made it. `now > latest_arrival_time` in _group_update is strict.
        if arrival is not None and (timeout is None or arrival <= timeout[0]):
            sim_time = arrival
            if sim_time > time_budget:
                break
            _, _, client_id = heapq.heappop(events)
            scheduler_start = time.perf_counter()
            action = scheduler.on_arrival(client_id, sim_time)
            selection_overhead += time.perf_counter() - scheduler_start
        else:
            sim_time = timeout[0]
            if sim_time > time_budget:
                break
            scheduler_start = time.perf_counter()
            action = scheduler.on_timeout(timeout[1], sim_time)
            selection_overhead += time.perf_counter() - scheduler_start

        if action["aggregate"]:
            participants = action["aggregate"]["client_ids"]
            staleness = action["aggregate"]["staleness"]

            # A superseded buffered delta is dropped: its client has since
            # delivered a fresher one, which is the one that gets aggregated.
            for cid in action["aggregate"]["superseded"]:
                held_deltas.pop(cid, None)

            batch_deltas = {
                cid: pending_deltas.pop(cid)
                for cid in action["aggregate"]["fresh"]
            }
            batch_deltas.update({
                cid: held_deltas.pop(cid)
                for cid in action["aggregate"]["from_buffer"]
            })
            batch_weights = {cid: client_weights[cid] for cid in participants}

            global_model = apply_update(
                global_model,
                batch_deltas,
                batch_weights,
                staleness,
                staleness_fn,
                alpha,
            )

            num_updates += 1
            last_update = {
                "sim_time": sim_time,
                "participants": participants,
                "staleness": staleness,
            }

            if num_updates % eval_every == 0:
                record(last_update)
                last_update = None

        buffered_id = action.get("buffered")
        if buffered_id is not None:
            held_deltas[buffered_id] = pending_deltas.pop(buffered_id)

        dispatch(action["dispatch"], sim_time)

    if last_update is not None:
        record(last_update)

    # Idle time is the headline claim: fast clients should barely wait.
    total_idle = sum(
        max(sim_time - busy, 0.0) for busy in busy_time.values()
    )

    stats = {
        "participation": participation,
        "comm_transfers": comm_transfers,
        "simulated_time": sim_time,
        "total_idle_sec": total_idle,
        "selection_overhead_sec": selection_overhead,
        "global_updates": num_updates,
        "mean_group_size": (
            sum(scheduler.group_sizes) / len(scheduler.group_sizes)
            if scheduler.group_sizes else 0.0
        ),
        "assigned_steps": scheduler.assigned_steps_log,
    }

    return global_model, history, stats


def _demo():
    """
    Scheduler self-check on a fake clock -- no torch, no training.

    Clients have fixed, known step times, so every assignment is checkable
    by hand. This is the thing that fails if the group logic breaks.
    """
    speeds = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.50, 5: 1.00}
    ids = list(speeds)

    def simulate(seed_speeds, budget=4000.0, **kwargs):
        """
        Mirrors run_fedcompass's event loop exactly, but each "delta" is a
        unique integer token instead of a state dict. That makes the buffer
        bookkeeping checkable: every token aggregated must be one a client
        actually produced, and none may be aggregated twice or lost.
        """
        sched = CompassScheduler(list(seed_speeds), **kwargs)
        events = []
        counter = 0
        now = 0.0
        aggregations = []
        aggregated_tokens = []
        participation = {cid: 0 for cid in seed_speeds}
        busy = {cid: 0.0 for cid in seed_speeds}

        pending = {}
        held = {}
        issued = set()
        in_flight = set()
        delivered = set()
        dropped = set()

        def go(batch, t):
            nonlocal counter
            for cid, steps in batch:
                assert steps >= 1, (cid, steps)
                counter += 1
                # Stand-in for the delta this dispatch will produce. Written
                # into `pending` at DISPATCH time, exactly as run_fedcompass
                # does -- training there happens eagerly when the client is
                # handed the model, so a second dispatch overwrites the slot.
                token = (cid, counter)
                issued.add(token)
                in_flight.add(token)
                pending[cid] = token
                duration = steps * seed_speeds[cid]
                busy[cid] += duration
                participation[cid] += 1
                heapq.heappush(events, (t + duration, counter, cid, token))

        go(sched.initial_dispatch(), 0.0)

        while True:
            timeout = sched.next_timeout()
            arrival = events[0][0] if events else None
            if timeout is None and arrival is None:
                break

            if arrival is not None and (timeout is None or arrival <= timeout[0]):
                now = arrival
                if now > budget:
                    break
                _, _, cid, arrived_token = heapq.heappop(events)
                in_flight.discard(arrived_token)
                delivered.add(arrived_token)
                action = sched.on_arrival(cid, now)
            else:
                now = timeout[0]
                if now > budget:
                    break
                action = sched.on_timeout(timeout[1], now)

            if action["aggregate"]:
                participants = action["aggregate"]["client_ids"]
                for c in action["aggregate"]["superseded"]:
                    dropped.add(held.pop(c))
                batch = {
                    c: pending.pop(c) for c in action["aggregate"]["fresh"]
                }
                batch.update({
                    c: held.pop(c)
                    for c in action["aggregate"]["from_buffer"]
                })
                for c, token in batch.items():
                    assert token[0] == c, (c, token)
                    # The invariant that the general-buffer bug violates:
                    # you cannot aggregate an update whose training has not
                    # finished yet.
                    assert token not in in_flight, (
                        f"aggregated delta {token} for client {c} while it "
                        f"was still training -- a buffered update was "
                        f"clobbered by a re-dispatch"
                    )
                aggregations.append(participants)
                aggregated_tokens.extend(batch.values())

            buffered_id = action.get("buffered")
            if buffered_id is not None:
                held[buffered_id] = pending.pop(buffered_id)

            go(action["dispatch"], now)

        # No delta aggregated twice, and every one came from a real dispatch.
        assert len(aggregated_tokens) == len(set(aggregated_tokens)), (
            "a delta was aggregated more than once"
        )
        assert set(aggregated_tokens) <= issued, "aggregated an unissued delta"

        # CONSERVATION. Every update a client actually finished must either
        # have been aggregated or still be sitting in a buffer when the clock
        # ran out. A delivered update that is in neither place was silently
        # dropped -- which is exactly what happens if a buffered delta gets
        # overwritten by its client's next dispatch. This is the check that
        # fails if the general-buffer handling regresses.
        lost = (
            delivered
            - set(aggregated_tokens)
            - set(pending.values())
            - set(held.values())
            - dropped
        )
        assert not lost, (
            f"{len(lost)} delivered update(s) were neither aggregated nor "
            f"buffered -- a client update was silently dropped: "
            f"{sorted(lost)[:5]}"
        )

        return sched, aggregations, participation, busy, now

    sched, aggregations, participation, busy, end = simulate(speeds)

    # Every assigned step count honours the configured bounds. The only
    # exception the algorithm permits is the max_local_steps bootstrap.
    for cid, steps in sched.assigned_steps_log:
        assert sched.min_local_steps <= steps <= sched.max_local_steps, (
            cid, steps
        )

    # Aggregation actually happened, and grouped clients together at least
    # some of the time -- if every batch were size 1 this would just be
    # asynchronous FL with extra steps.
    assert len(aggregations) > 5, len(aggregations)
    assert max(len(batch) for batch in aggregations) > 1, aggregations

    # No starvation: the 10x-slower client still participates.
    for cid in ids:
        assert participation[cid] > 0, (cid, participation)

    # No client is aggregated twice in one batch, and no unknown ids appear.
    for batch in aggregations:
        assert len(batch) == len(set(batch)), batch
        assert set(batch) <= set(ids), batch

    # The point of the whole exercise: the fastest client should spend most
    # of its time training, not idling behind the 10x-slower straggler.
    assert busy[1] / end > 0.8, (busy[1], end)

    # Deterministic: identical inputs, identical trace.
    sched2, aggregations2, participation2, _, _ = simulate(speeds)
    assert aggregations == aggregations2
    assert participation == participation2
    assert sched.assigned_steps_log == sched2.assigned_steps_log

    # A single client degenerates to plain sequential training, not a crash.
    _, solo, solo_part, _, _ = simulate({1: 0.1}, budget=100.0)
    assert len(solo) > 0 and solo_part[1] > 0

    # Homogeneous clients should land in one group together.
    _, uniform, _, _, _ = simulate({c: 0.2 for c in range(1, 6)}, budget=500.0)
    assert max(len(b) for b in uniform) >= 4, uniform

    # Force the general-buffer path: a 100x speed spread with the tightest
    # legal deadline guarantees stragglers miss their arrival group, which is
    # the route that parks a delta in the general buffer while re-dispatching
    # its client. The token checks inside simulate() are what fail if the
    # buffered delta gets clobbered by the re-dispatch.
    straggler_speeds = {1: 0.001, 2: 0.002, 3: 0.01, 4: 2.0, 5: 10.0}
    _, late, late_part, _, _ = simulate(
        straggler_speeds,
        budget=400.0,
        latest_time_factor=1.0,
        min_local_steps=5,
        max_local_steps=200,
    )
    assert len(late) > 5, late
    for cid in straggler_speeds:
        assert late_part[cid] > 0, (cid, late_part)

    # Bad configuration is rejected rather than silently misbehaving.
    for bad in (
        {"min_local_steps": 300, "max_local_steps": 200},
        {"latest_time_factor": 0.5},
        {"speed_momentum": 2.0},
        {"max_local_steps": 0},
    ):
        try:
            CompassScheduler([1, 2], **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Staleness weighting decays, and is 1.0 at zero staleness.
    st = polynomial_staleness(0.5)
    assert st(0) == 1.0
    assert st(1) < st(0) and st(5) < st(1)

    print(
        f"compass.py self-check passed "
        f"({len(aggregations)} aggregations, "
        f"mean group {sched.group_sizes and sum(sched.group_sizes)/len(sched.group_sizes):.2f}, "
        f"fastest-client utilisation {busy[1]/end:.1%})"
    )


if __name__ == "__main__":
    _demo()
