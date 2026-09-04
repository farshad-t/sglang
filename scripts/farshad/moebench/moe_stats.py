"""Read archbench MoE expert-stats CSVs and turn them into per-expert token counts.

This module is a faithful port of the reader in archbench
`networks/MLP/Qwen/Qwen3/qwen3_moe_expert_dist_utils.py`. The projection and this
benchmark MUST derive their expert distribution from the same numbers, or the
comparison measures the reader difference instead of the model.

The two phases carry different schemas, on purpose:

  DECODE   Layer,Activations_Per_Expert,Num_Experts,Batch_Size,TopK
           A full per-layer histogram: one row per distinct per-expert token
           count. Mass is exact: sum(act * nexp) == batch * topk.

  PREFILL  Layer,Bucket,Expert_IDs,Avg_Activations,Batch_Size,TopK,Num_Experts
           Quantized to 5 load buckets, each with ONE averaged activation count.
           `Num_Experts` is empty in the real-prompt schema; the count is
           len(Expert_IDs). Mass legitimately OVERSHOOTS batch*seq*topk because
           the stats were collected on prompts longer than the modelled seq len.

Both readers reduce to the same internal form: Dict[tokens_per_expert -> num_experts].
"""

import csv
from typing import Dict, List, Optional, Tuple

# How far a PREFILL histogram's mass may sit from the routed-token count the config
# implies before it is treated as the wrong data rather than the same data collected
# on slightly different prompts. Mirrors EXPERT_DIST_PREFILL_MASS_TOLERANCE in
# archbench. Decode is exact and gets no band.
PREFILL_MASS_TOLERANCE = 0.20

Histogram = Dict[int, int]


def read_decode(csv_file: str, batch: int, layer: int) -> Optional[Histogram]:
    hist: Histogram = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            if int(row["Layer"]) != layer or int(row["Batch_Size"]) != batch:
                continue
            act = int(row["Activations_Per_Expert"])
            hist[act] = hist.get(act, 0) + int(row["Num_Experts"])
    return hist or None


def read_prefill(csv_file: str, batch: int, layer: int) -> Optional[Histogram]:
    hist: Histogram = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            if int(row["Layer"]) != layer:
                continue
            # The real-prompt files concatenate every batch into one CSV.
            if row.get("Batch_Size") not in (None, ""):
                if int(row["Batch_Size"]) != batch:
                    continue
            act = int(row["Avg_Activations"])
            if row.get("Num_Experts") not in (None, ""):
                nexp = int(row["Num_Experts"])
            else:
                ids = row.get("Expert_IDs") or ""
                nexp = len([x for x in ids.split(",") if x.strip()])
            hist[act] = hist.get(act, 0) + nexp
    return hist or None


def read_decode_layers_averaged(csv_file: str, batch: int) -> Optional[Histogram]:
    hist: Histogram = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            if int(row["Batch_Size"]) != batch:
                continue
            act = int(row["Activations_Per_Expert"])
            hist[act] = hist.get(act, 0) + int(row["Num_Experts"])
    return hist or None


def read_prefill_layers_averaged(csv_file: str) -> Optional[Histogram]:
    hist: Histogram = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            act = int(row["Avg_Activations"])
            if row.get("Num_Experts") not in (None, ""):
                nexp = int(row["Num_Experts"])
            else:
                ids = row.get("Expert_IDs") or ""
                nexp = len([x for x in ids.split(",") if x.strip()])
            hist[act] = hist.get(act, 0) + nexp
    return hist or None


def read_histogram(csv_file: str, phase: str, batch: int, layer: int,
                   mode: str = "per_layer") -> Optional[Histogram]:
    if phase == "decode":
        return (read_decode_layers_averaged(csv_file, batch)
                if mode == "layers_averaged" else read_decode(csv_file, batch, layer))
    return (read_prefill_layers_averaged(csv_file)
            if mode == "layers_averaged" else read_prefill(csv_file, batch, layer))


def validate(hist: Histogram, num_experts: int, routed_tokens: Optional[int],
             phase: str, where: str = "<histogram>") -> None:
    """Reject a histogram that is not physically possible. Port of archbench's
    validate_expert_token_dist. Two properties are physics, not convention:

      (A) sum(num_experts) <= E. A layer cannot activate more experts than it has.
          This is the check that catches an analyzer emitting its activations==1
          bucket as a token REMAINDER instead of an expert count.
      (B) sum(act * nexp) == routed_tokens. Every routed token lands on exactly
          topk experts, so the mass is fixed. Prefill is checked against a band
          instead of exact equality (see PREFILL_MASS_TOLERANCE).
    """
    if not hist:
        raise ValueError(f"{where}: empty histogram")
    for act, nexp in sorted(hist.items()):
        if act < 1 or nexp < 1:
            raise ValueError(f"{where}: bucket tokens_per_expert={act} -> num_experts={nexp}; "
                             f"both must be >= 1 (each group's input batch must be >= its "
                             f"weight batch, i.e. avg >= 1)")
    active = sum(hist.values())
    if active > num_experts:
        biggest = max(hist.items(), key=lambda kv: kv[1])
        raise ValueError(
            f"{where}: reports {active} activated experts but the layer only has {num_experts}. "
            f"Largest bucket is tokens_per_expert={biggest[0]} -> num_experts={biggest[1]}. "
            f"A num_experts count that tracks the TOKEN count is the signature of an analyzer "
            f"emitting its activations==1 bucket as a remainder -- re-collect this "
            f"(model, dtype, batch).")
    if routed_tokens is None or routed_tokens <= 0:
        return
    mass = sum(act * nexp for act, nexp in hist.items())
    gap = (mass - routed_tokens) / routed_tokens
    if phase == "decode":
        if mass != routed_tokens:
            raise ValueError(f"{where}: mass {mass} != topk*tokens {routed_tokens} "
                             f"({gap:+.1%}); decode is exact. The histogram covers a "
                             f"different token count than this cell routes.")
    elif abs(gap) > PREFILL_MASS_TOLERANCE:
        raise ValueError(f"{where}: mass {mass} vs topk*tokens {routed_tokens} is {gap:+.1%}, "
                         f"outside the +/-{PREFILL_MASS_TOLERANCE:.0%} band. The stats were "
                         f"collected at a different sequence length than this cell models.")


def histogram_mass(hist: Histogram) -> int:
    """Total routed-token slots the histogram accounts for."""
    return sum(act * nexp for act, nexp in hist.items())


def groups(hist: Histogram) -> List[Tuple[int, int]]:
    """The projection's MoE decomposition: one (num_experts, tokens_per_expert) group
    per histogram bucket. Each group is ONE batched GEMM in the model, of shape
    [nexp, tokens, K] x [nexp, K, N] -- mirrors archbench returning
    [expert_dist(nexp, avg) for avg, nexp in token_dist.items()].
    Sorted by descending tokens_per_expert so output ordering is stable."""
    return [(nexp, act) for act, nexp in sorted(hist.items(), reverse=True)]


def expert_token_counts(hist: Histogram, num_experts: int, topk: int) -> Tuple[List[int], int, int]:
    """Expand a histogram into per-expert token counts.

    Returns (counts, num_tokens, slack) where `counts` has length num_experts
    (zero-padded for idle experts), `num_tokens` is the token count the benchmark
    must feed, and `slack` is how many routed-token slots were ADDED to make the
    count divisible by topk.

    Why slack exists: a [num_tokens, topk] routing table holds exactly
    num_tokens*topk assignments, so the histogram mass must be a multiple of topk.
    Decode mass is always B*topk, so slack is 0. Prefill mass is an overshooting
    average over 5 buckets and is usually NOT divisible by topk, so we round the
    token count UP and add the few leftover slots to the most-loaded experts.
    Bounded by topk-1 slots, i.e. <= 7 of ~500k for a bs64 prefill layer.
    """
    counts = [0] * num_experts
    # Busiest experts get the lowest ids. The kernel sorts tokens by expert, so the
    # id assignment cannot change the measured time -- fixing it keeps runs reproducible.
    idx = 0
    for act, nexp in sorted(hist.items(), reverse=True):
        for _ in range(nexp):
            if idx >= num_experts:
                raise ValueError(f"histogram activates more than {num_experts} experts")
            counts[idx] = act
            idx += 1

    mass = sum(counts)
    slack = (-mass) % topk
    for i in range(slack):
        counts[i % max(1, idx)] += 1
    num_tokens = (mass + slack) // topk

    active = idx
    if active < topk:
        raise ValueError(f"only {active} experts are active but topk={topk}; a token cannot "
                         f"be routed to topk distinct experts")
    if max(counts) > num_tokens:
        raise ValueError(f"expert with {max(counts)} tokens exceeds the {num_tokens}-token "
                         f"budget; no routing table can realise this histogram")
    return counts, num_tokens, slack
