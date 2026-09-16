"""Trace-driven simulation of a GPU pool serving multi-step coding agents: mixed engines, prefill/decode
disaggregation (PD), or the PPD layout, at any pool size, on measured or scaled hardware.

Why a simulator: the PD verdicts in Milestone 3 come from two GPUs, where one role bounds the system. The
questions "what at 8 or 32 GPUs", "what if part of the traffic is disaggregated", and "what on a 141 GB GPU"
cannot be run on the hardware at hand. The simulator replays the mixed56 workload (per task: prompt and
completion tokens of every step, the gap between steps, the closed-loop admission of the harness) through a
per-iteration model of vLLM's scheduler and must reproduce the three measured two-GPU runs before any
prediction is quoted (gate in the task record).

Engine model (one vLLM engine, per iteration; parameters fitted from the 2xL40S runs, Qwen3-4B FP8, eager):
  * running set <= max_num_seqs; a request that is still prefilling counts as running;
  * chunked prefill under max_num_batched_tokens, decodes take one token each first;
  * KV capacity in tokens: running requests are pinned, finished contexts stay as an LRU prefix cache
    keyed by task (a task's next prompt extends its previous context), evicted when space is needed;
    a decode step with no free space preempts the newest running request (recomputed later);
  * iteration time = a + b * n_decode + c * (sum of decode context tokens) + prefill cost, where a prefill chunk
    at context position n costs p * chunk * (1 + n / attn_ktok): the linear term is the weight FLOPs, the
    position term is attention over the context (for Qwen3-4B the two are equal near n = 20K tokens).
Roles: mixed (prefill + decode), P (prefill only; the finished KV is pushed to a D engine, transfer cost
22 ms + 0.0102 ms per token on PCIe), D (receives KV, decodes). Routing: task-sticky on mixed engines and on
P engines (turn 1 to the engine with the fewest outstanding requests); D by fewest outstanding.

Three layouts, no variants of our own:
  mixed  - colocated replicas, task-sticky.
  pd     - classic disaggregation: every step prefills on a P engine and decodes on a D engine.
  ppd    - arXiv 2603.13358: P engines plus prefill-capable decode engines (pD). A task's turn one
           prefills on a P engine and its KV is pushed to the task's home pD; every later step stays
           on that pD and append-prefills against its own prefix cache. That is what the published
           decision engine did on this workload: in results/mixed56-vast-ppd-pcie-20260907-r1 it sent
           all 113 first turns to PD and all 3,902 later steps local (1,547 by its 512-token
           short-input bypass, 2,355 by its offline lookup table). The table is measured hardware data
           the simulator does not carry, so the simulator claims nothing beyond those recorded decisions.

Usage:
  pd_pool_simulation.py extract results/<run> --out workload.json
  pd_pool_simulation.py run workload.json --pool mixed:2 --hw l40s [--agents-per-gpu 16] [--out JSON]
  pd_pool_simulation.py run workload.json --pool pd:1,1                          (1 P, 1 D)
  pd_pool_simulation.py run workload.json --pool ppd:6,2                         (6 pD, 2 P)
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import statistics
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

# Fitted on results/mixed56-vast-pd-pcie-20260907-r3 (D side, 3,997 requests: TPOT = 34.6 + 0.38*n + 0.054 per
# thousand tokens of summed batch context, r2 0.42; the context term is the KV read of every decode step) and
# on the P side's aggregate throughput (101.7M prompt tokens, no cache hits, P saturated for the 2.9 h run:
# 9.7K tokens/s, i.e. 211 ms per 2048-token iteration, minus the fixed term = 0.085 ms per token at the
# workload's mean context position; the position factor over this workload is 2.11, so p = 0.040). A mixed engine
# pays 1.4x that prefill cost per chunk (fitted on the FCFS-sticky run's TPOT 142 ms and JCT; the single-engine PPD
# run was held out and lands within 4%): eager-mode mixed batches launch prefill and decode kernels separately and
# hash prefix-cache blocks on the CPU, which a prefill-only engine amortises. KV capacity is the
# engine's own log line (275,008 tokens). Transfer constants are Milestone 3's measured PCIe values.
HW = {
    "l40s": dict(a_ms=34.6, b_ms=0.38, c_ms_per_ktok=0.054, p_ms_per_tok=0.040, attn_ktok=20.0, mixed_prefill_factor=1.4,
                 kv_tokens=275_008, xfer_fixed_ms=22.0, xfer_ms_per_tok=0.0102),
}
# H200 141 GB relative to L40S 48 GB: KV capacity from free memory after 4.5 GB of weights at 0.95 utilisation,
# (141*0.95-4.5)/(46*0.95-4.5) = 3.3x; decode per-sequence and KV-read terms by HBM bandwidth (4.8/0.864 TB/s);
# prefill by dense FP8 compute (1979/733 TFLOPS); the fixed per-iteration term is launch overhead (eager
# mode) and is kept. NVLink transfer at 900 GB/s versus the 14 GB/s measured on PCIe. These are assumptions
# to be read as a sensitivity band, not measurements.
HW["h200"] = dict(HW["l40s"], b_ms=0.38 / 5.6, c_ms_per_ktok=0.054 / 5.6, p_ms_per_tok=0.040 / 2.7,
                  kv_tokens=int(275_008 * 3.3), xfer_ms_per_tok=0.0102 * 14 / 900)


# ------------------------------------------------------------------------------------------------ workload
def extract(run: Path) -> dict:
    """Per original task: steps (prompt, completion tokens) and the gap after each step, from the replay logs."""
    summary = json.loads((run / "output/throughput_summary.json").read_text())
    order = {t["run_instance_id"]: t["manifest_index"] for t in summary["tasks"]}
    tasks = []
    for d in sorted((run / "output").iterdir(), key=lambda d: order.get(d.name, 1 << 30)):
        if not d.is_dir() or "__replacement-" in d.name:
            continue
        ends, starts = [], []
        for log in sorted(d.glob("attempt_*/openclaw_host_replay.jsonl")):
            for line in log.open():
                if '"llm_call_end"' not in line and '"llm_call_start"' not in line:
                    continue
                r = json.loads(line)
                if r.get("event") == "llm_call_start":
                    starts.append(r["ts"])
                elif r.get("event") == "llm_call_end" and r["data"].get("finish_reason") != "error":
                    sg = r["data"]["shadow_generation"]
                    ends.append((r["ts"], sg["prompt_tokens"], r["data"]["completion_tokens"]))
        steps = [{"prompt": p, "gen": g} for _, p, g in ends]
        for i, (ts_end, _, _) in enumerate(ends[:-1]):
            nxt = [s for s in starts if s > ts_end]
            steps[i]["gap_s"] = round(min(nxt) - ts_end, 3) if nxt else 0.0
        steps[-1]["gap_s"] = 0.0
        tasks.append({"task": d.name, "steps": steps})
    return {"source_run": run.name, "concurrency": summary["concurrency"],
            "replacement_delay_mean_s": summary["replacement_load"]["delay_mean_s"], "tasks": tasks}


# ------------------------------------------------------------------------------------------------ engine
@dataclass
class Request:
    task: "Task"
    step: int
    prompt: int
    gen: int
    arrival: float
    cached: int = 0
    computed: int = 0            # prompt tokens with KV present (cached + prefilled)
    generated: int = 0
    first_token_t: float | None = None
    finish_t: float | None = None
    preempted: int = 0
    p_engine: "Engine | None" = None
    p_cached: int = 0
    d_target: "Engine | None" = None
    p_queue_s: float = 0.0
    d_arrival: float | None = None
    xfer_tokens: int = 0


@dataclass
class Task:
    tid: int
    profile: list[dict]
    measured: bool
    admitted: float = 0.0
    step: int = 0
    pool: str = "mixed"
    sticky: dict = field(default_factory=dict)   # role -> Engine
    done_t: float | None = None


class Engine:
    def __init__(self, eid: str, role: str, hw: dict, max_seqs: int, budget: int, shared_prefix: int):
        self.eid, self.role, self.hw = eid, role, hw
        self.max_seqs, self.budget, self.shared = max_seqs, budget, shared_prefix
        self.cap = hw["kv_tokens"]
        self.running: list[Request] = []
        self.waiting: deque[Request] = deque()
        self.cache: OrderedDict[int, int] = OrderedDict()   # task id -> resident context tokens (LRU, oldest first)
        self.pinned = 0
        self.busy_s = 0.0
        self.iterations = 0
        self.scheduled = False

    # KV space -------------------------------------------------------------------------------------------
    def cached_free(self) -> int:
        return sum(self.cache.values())

    def make_room(self, need: int) -> bool:
        """Evict LRU finished contexts until `need` tokens fit beside the pinned ones."""
        while self.pinned + self.cached_free() + need > self.cap and self.cache:
            self.cache.popitem(last=False)
        return self.pinned + self.cached_free() + need <= self.cap

    def outstanding(self) -> int:
        return len(self.running) + len(self.waiting)

    def pending_prefill_tokens(self) -> int:
        return sum(max(0, r.prompt - r.computed) for r in self.running) + sum(
            r.prompt - min(self.cache.get(r.task.tid, 0) or self.shared, r.prompt) for r in self.waiting)

    def prefill_ms_per_token(self) -> float:
        """Mean cost of one prefill token at this workload's context positions (position factor 2.1)."""
        return self.hw["p_ms_per_tok"] * 2.1 * (self.hw["mixed_prefill_factor"] if self.role == "mixed" else 1.0)

    # one scheduler iteration ---------------------------------------------------------------------------
    def iterate(self, now: float, sim: "Sim") -> float:
        hw = self.hw
        budget = self.budget
        decodes = [r for r in self.running if r.computed >= r.prompt and r.generated > 0]
        # decode steps need one more KV token each; preempt the newest running request when there is no room
        while decodes and not self.make_room(len(decodes)):
            victim = self.running.pop()
            self.pinned -= victim.computed + victim.generated
            victim.computed, victim.cached, victim.generated = 0, 0, 0
            victim.preempted += 1
            self.waiting.appendleft(victim)
            decodes = [r for r in self.running if r.computed >= r.prompt and r.generated > 0]
        budget -= len(decodes)
        prefill_cost = 0.0   # token-equivalents: chunk * (1 + position / attn_ktok)
        pos = lambda r, chunk: chunk * (1 + (r.computed + chunk / 2) / (hw["attn_ktok"] * 1e3))
        admitted: list[Request] = []
        # admit waiting requests while there are slots and budget
        while self.waiting and len(self.running) < self.max_seqs and budget > 0:
            r = self.waiting[0]
            if r.d_arrival is not None:
                resident = self.cache.get(r.task.tid, 0)   # blocks this engine still holds from the previous step
                keep = min(resident, r.prompt)
                if resident:
                    self.cache.pop(r.task.tid)
                    self.pinned += resident
                if not self.make_room(r.prompt - keep):   # only the pushed blocks have to fit
                    if resident:
                        self.cache[r.task.tid] = resident
                        self.cache.move_to_end(r.task.tid, last=False)
                        self.pinned -= resident
                    break
                self.pinned += r.prompt - keep
                r.cached = r.computed = r.prompt
            else:
                resident = self.cache.get(r.task.tid, 0)
                cached = min(resident, r.prompt) if resident else min(self.shared, r.prompt)
                chunk = min(budget, r.prompt - cached)
                if resident:
                    self.cache.pop(r.task.tid)   # the resident context becomes part of the pinned request
                    self.pinned += resident
                    need = max(0, cached + chunk - resident)
                else:
                    need = cached + chunk
                if not self.make_room(need):
                    if resident:
                        self.cache[r.task.tid] = resident
                        self.cache.move_to_end(r.task.tid, last=False)
                        self.pinned -= resident
                    break
                self.pinned += need
                r.cached, r.computed = cached, cached
                if self.role == "P":
                    r.p_cached = cached
                prefill_cost += pos(r, chunk)
                r.computed += chunk
                budget -= chunk
            self.waiting.popleft()
            self.running.append(r)
            admitted.append(r)
        # continue chunked prefills of requests admitted earlier
        for r in self.running:
            if r in admitted or r.computed >= r.prompt or budget <= 0:
                continue
            chunk = min(budget, r.prompt - r.computed)
            if not self.make_room(chunk):
                break
            self.pinned += chunk
            prefill_cost += pos(r, chunk)
            r.computed += chunk
            budget -= chunk
        ctx = sum(r.computed + r.generated for r in decodes)
        if self.role == "mixed":
            prefill_cost *= hw["mixed_prefill_factor"]
        dt = (hw["a_ms"] + hw["b_ms"] * len(decodes) + hw["c_ms_per_ktok"] * ctx / 1e3 + hw["p_ms_per_tok"] * prefill_cost) / 1e3
        t = now + dt
        self.busy_s += dt
        self.iterations += 1
        for r in decodes:
            r.generated += 1
            self.pinned += 1
        for r in list(self.running):
            if r.computed >= r.prompt and r.generated == 0:
                # prefill (or D's one-token step) completed this iteration: the first token is sampled now
                if self.role == "P":
                    self.finish(r, t, sim)
                    continue
                r.generated = 1
                self.pinned += 1
                r.first_token_t = t
            if r.generated >= r.gen:
                self.finish(r, t, sim)
        return t

    def finish(self, r: Request, t: float, sim: "Sim") -> None:
        self.running.remove(r)
        tokens = r.computed + r.generated
        self.pinned -= tokens
        self.cache[r.task.tid] = tokens
        self.cache.move_to_end(r.task.tid)
        r.finish_t = t
        if self.role == "P":
            sim.transfer(r, t)
        else:
            sim.request_done(r, t)


# ------------------------------------------------------------------------------------------------ pool
class Sim:
    def __init__(self, workload: dict, pool: str, hw: dict, agents: int, max_seqs: int, budget: int, shared_prefix: int,
                 seed: int, tasks_per_gpu: float, d_max_seqs: int | None):
        self.hw, self.rng = hw, random.Random(seed)
        kind, _, spec = pool.partition(":")
        assert kind in ("mixed", "pd", "ppd"), f"pool must be mixed:N, pd:P,D or ppd:pD,P: {pool}"
        n = [int(x) for x in spec.split(",")]
        # ppd's decode side is a pD: it decodes, keeps its prefix cache and prefills later turns itself,
        # which is what the mixed engine already models, so it reuses that role.
        self.mixed = [Engine(f"M{i}", "mixed", hw, max_seqs, budget, shared_prefix) for i in range(n[0] if kind != "pd" else 0)]
        p_count, d_count = (n[0], n[1]) if kind == "pd" else (n[1], 0) if kind == "ppd" else (0, 0)
        self.P = [Engine(f"P{i}", "P", hw, max_seqs, budget, shared_prefix) for i in range(p_count)]
        self.D = [Engine(f"D{i}", "D", hw, d_max_seqs or max_seqs, budget, shared_prefix) for i in range(d_count)]
        self.engines = self.mixed + self.P + self.D
        self.kind = kind
        self.gpus = len(self.engines)
        self.agents = agents
        profiles = [t["steps"] for t in workload["tasks"]]
        n_measured = round(tasks_per_gpu * self.gpus)
        self.measured = [Task(i, profiles[i % len(profiles)], True) for i in range(n_measured)]
        self.profiles = profiles
        self.pending = deque(self.measured)
        self.delay_mean = workload["replacement_delay_mean_s"]
        self.events: list = []   # (time, seq, kind, payload)
        self.seq = 0
        self.requests: list[Request] = []
        self.active = 0
        self.next_tid = n_measured
        self.now = 0.0

    def push(self, t: float, kind: str, payload) -> None:
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, payload))

    # admission (harness semantics: `agents` slots, measured tasks first, then background copies keep the load)
    def admit(self, t: float) -> None:
        while self.active < self.agents:
            if self.pending:
                task = self.pending.popleft()
            else:
                task = Task(self.next_tid, self.profiles[self.next_tid % len(self.profiles)], False)
                self.next_tid += 1
            task.admitted = t
            task.pool = "pd" if self.kind == "pd" else "mixed"
            self.active += 1
            self.start_step(task, t)

    def start_step(self, task: Task, t: float) -> None:
        s = task.profile[task.step]
        r = Request(task, task.step, s["prompt"], s["gen"], t)
        self.requests.append(r)
        if self.kind == "ppd":
            # Published rule as it behaved on this workload: turn one disaggregates, every later step
            # stays on the task's home pD and append-prefills against the context it already holds.
            home = task.sticky.get("mixed") or min(self.mixed, key=Engine.outstanding)
            task.sticky["mixed"] = home
            if task.step == 0:
                eng = task.sticky.get("P") or min(self.P, key=Engine.outstanding)
                task.sticky["P"] = eng
                r.p_engine, r.d_target = eng, home
            else:
                eng = home
        elif task.pool == "mixed":
            eng = task.sticky.get("mixed") or min(self.mixed, key=Engine.outstanding)
            task.sticky["mixed"] = eng
        else:
            eng = task.sticky.get("P") or min(self.P, key=Engine.outstanding)
            task.sticky["P"] = eng
            r.p_engine = eng
        eng.waiting.append(r)
        self.kick(eng, t)

    def kick(self, eng: Engine, t: float) -> None:
        if not eng.scheduled:
            eng.scheduled = True
            self.push(t, "iter", eng)

    def transfer(self, r: Request, t: float) -> None:
        r.p_queue_s = (r.first_token_t or t) - r.arrival
        resident = r.d_target.cache.get(r.task.tid, 0) if r.d_target is not None else 0
        r.xfer_tokens = max(0, r.prompt - min(resident, r.prompt))
        done = t + (self.hw["xfer_fixed_ms"] + self.hw["xfer_ms_per_tok"] * r.xfer_tokens) / 1e3
        self.push(done, "arrive_d", r)

    def request_done(self, r: Request, t: float) -> None:
        task = r.task
        task.step += 1
        if task.step >= len(task.profile):
            task.done_t = t
            self.active -= 1
            delay = self.rng.expovariate(1 / self.delay_mean) if self.delay_mean else 0.0
            self.push(t + delay, "admit", None)
        else:
            gap = task.profile[r.step]["gap_s"]
            self.push(t + gap, "step", task)

    def run(self) -> dict:
        self.admit(0.0)
        while self.events:
            t, _, kind, payload = heapq.heappop(self.events)
            self.now = t
            if kind == "iter":
                eng: Engine = payload
                if eng.running or eng.waiting:
                    nxt = eng.iterate(t, self)
                    self.push(nxt, "iter", eng)
                else:
                    eng.scheduled = False
            elif kind == "step":
                self.start_step(payload, t)
            elif kind == "admit":
                if any(task.done_t is None for task in self.measured):
                    self.admit(t)
            elif kind == "arrive_d":
                r: Request = payload
                r.d_arrival = t
                r.first_token_t = None
                eng = r.d_target or min(self.D, key=Engine.outstanding)
                eng.waiting.append(r)
                self.kick(eng, t)
            if all(task.done_t is not None for task in self.measured):
                break
        return self.report()

    def report(self) -> dict:
        end = max(task.done_t for task in self.measured)
        jct = sorted(task.done_t - task.admitted for task in self.measured)
        r2t = sorted(task.done_t for task in self.measured)
        done = [r for r in self.requests if r.finish_t is not None and r.task.measured and (r.p_engine is None or r.d_arrival is not None)]
        tpots = [(r.finish_t - r.first_token_t) / (r.gen - 1) for r in done if r.gen > 1 and r.first_token_t]
        weights = [r.gen - 1 for r in done if r.gen > 1 and r.first_token_t]
        tw = sum(t * w for t, w in zip(tpots, weights)) / sum(weights)
        pd_done = [r for r in done if r.p_engine is not None]
        out = {
            "gpus": self.gpus, "agents": self.agents, "measured_tasks": len(self.measured), "requests": len(done),
            "makespan_min": round(end / 60, 1),
            "mean_jct_elapsed_min": round(statistics.fmean(jct) / 60, 1), "p95_jct_elapsed_min": round(jct[int(0.95 * len(jct)) - 1] / 60, 1),
            "mean_jct_r2t_min": round(statistics.fmean(r2t) / 60, 1), "p95_jct_r2t_min": round(r2t[int(0.95 * len(r2t)) - 1] / 60, 1),
            "tpot_ms_token_weighted": round(tw * 1e3, 1), "tpot_ms_median": round(statistics.median(tpots) * 1e3, 1),
            "share_tpot_over_50ms": round(sum(1 for t in tpots if t > 0.05) / len(tpots), 3),
            "ttft_s_mean": round(statistics.fmean((r.first_token_t - r.arrival) for r in done if r.first_token_t), 1),
            "preempted_share": round(sum(1 for r in done if r.preempted) / len(done), 3),
            "engine_busy": {e.eid: round(e.busy_s / end, 2) for e in self.engines},
        }
        local = [r for r in done if r.p_engine is None]
        if local:
            out["cached_share_local"] = round(sum(min(r.cached, r.prompt) for r in local) / sum(r.prompt for r in local), 3)
        if pd_done:
            out["pd_requests"] = len(pd_done)
            out["pd_request_share"] = round(len(pd_done) / len(done), 3)
            out["p_wait_s_mean"] = round(statistics.fmean(r.p_queue_s for r in pd_done), 1)
            out["cached_share_on_p"] = round(sum(r.p_cached for r in pd_done) / sum(r.prompt for r in pd_done), 3)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract"); ex.add_argument("run", type=Path); ex.add_argument("--out", type=Path, required=True)
    rn = sub.add_parser("run"); rn.add_argument("workload", type=Path)
    rn.add_argument("--pool", required=True, help="mixed:N | pd:P,D | ppd:pD,P")
    rn.add_argument("--hw", default="l40s", choices=sorted(HW))
    rn.add_argument("--agents-per-gpu", type=float, default=16)
    rn.add_argument("--tasks-per-gpu", type=float, default=28)
    rn.add_argument("--max-num-seqs", type=int, default=8)
    rn.add_argument("--d-max-num-seqs", type=int)
    rn.add_argument("--max-num-batched-tokens", type=int, default=2048)
    rn.add_argument("--shared-prefix", type=int, default=272, help="tokens of the common system prompt (turn-1 cache hits)")
    rn.add_argument("--kv-scale", type=float, default=1.0); rn.add_argument("--p-scale", type=float, default=1.0)
    rn.add_argument("--a-scale", type=float, default=1.0)
    rn.add_argument("--mixed-prefill-factor", type=float, help="override the mixed-engine prefill cost factor")
    rn.add_argument("--seed", type=int, default=0)
    rn.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "extract":
        w = extract(a.run)
        a.out.write_text(json.dumps(w) + "\n")
        n = sum(len(t["steps"]) for t in w["tasks"])
        print(f"{a.run.name}: {len(w['tasks'])} tasks, {n} steps, concurrency {w['concurrency']}")
        return
    hw = dict(HW[a.hw])
    hw["kv_tokens"] = int(hw["kv_tokens"] * a.kv_scale)
    hw["p_ms_per_tok"] *= a.p_scale
    hw["a_ms"] *= a.a_scale
    if a.mixed_prefill_factor is not None:
        hw["mixed_prefill_factor"] = a.mixed_prefill_factor
    workload = json.loads(a.workload.read_text())
    sim = Sim(workload, a.pool, hw, agents=0, max_seqs=a.max_num_seqs, budget=a.max_num_batched_tokens, shared_prefix=a.shared_prefix,
              seed=a.seed, tasks_per_gpu=a.tasks_per_gpu, d_max_seqs=a.d_max_num_seqs)
    sim.agents = round(a.agents_per_gpu * sim.gpus)
    out = {"pool": a.pool, "hw": a.hw, "params": hw, "max_num_seqs": a.max_num_seqs, "seed": a.seed, **sim.run()}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
