"""CPU row/exposure/cost and frame/pair controls; never select a method by HR."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sched_common import *
from schedule_new import scalar_overlap


def distribution(values):
    values = sorted(float(v) for v in values)
    require(values and all(math.isfinite(v) for v in values), "Empty/nonfinite cost distribution")

    def quantile(q):
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)

    return dict(n=len(values), min=values[0], max=values[-1], mean=statistics.fmean(values),
                std=statistics.pstdev(values),
                quantiles={str(q): quantile(q) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)})


def semantic_rows(table, name="rows"):
    return [tuple(tuple(table["record_keys"][i]) for i in row) for row in table[name]]


def cost_rows(table, overlap):
    result, edges = [], []
    for row in semantic_rows(table):
        require(len({k[1] for k in row}) == 1, "Cross-cost needs same-time triples")
        frame = row[0][1]
        values = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            q = overlap["frames"][frame][tuple(sorted((row[a][0], row[b][0])))]
            value = (q["camera_center_distance"] / overlap["baseline_scale"]
                     + q["optical_axis_angle_radians"] / math.pi
                     + (1 - q["ratio"] if q["ratio"] is not None else 1.))
            values.append(value)
        result.append(sum(values))
        edges.extend(values)
    return result, edges


def compare_schedules(baseline, candidate, left_overlap, right_overlap):
    a, b = semantic_rows(baseline), semantic_rows(candidate)
    require(len(a) == len(b) == 6000 and set(map(tuple, baseline["record_keys"])) ==
            set(map(tuple, candidate["record_keys"])), "Different schedule population")
    cameras = sorted({key[0] for key in baseline["record_keys"]})
    exposures = []
    for camera in cameras:
        left = [sum(row[col][0] == camera for row in a) for col in range(3)]
        right = [sum(row[col][0] == camera for row in b) for col in range(3)]
        exposures.append(dict(camera=camera, baseline=left, candidate=right,
                              delta=[r - l for l, r in zip(left, right)]))
    per_key = []
    for col in range(3):
        ca, cb = Counter(row[col] for row in a), Counter(row[col] for row in b)
        per_key.append(dict(column=col, differing_keys=sum(ca[k] != cb[k] for k in ca),
                            max_abs_delta=max(abs(cb[k] - ca[k]) for k in ca)))
    costs = {}
    vectors = {}
    for name, table, overlap in (
        ("baseline_rows_baseline_parent", baseline, left_overlap),
        ("candidate_rows_candidate_parent", candidate, right_overlap),
        ("baseline_rows_candidate_parent", baseline, right_overlap),
        ("candidate_rows_baseline_parent", candidate, left_overlap),
    ):
        rows, edges = cost_rows(table, overlap)
        vectors[name] = rows
        costs[name] = dict(triplet=distribution(rows), selected_edge=distribution(edges))
    delta = [y - x for x, y in zip(vectors["baseline_rows_baseline_parent"],
                                   vectors["candidate_rows_candidate_parent"])]
    fa = Counter(tuple(sorted(row)) for row in a)
    fb = Counter(tuple(sorted(row)) for row in b)
    normalized = []
    for table in (baseline, candidate):
        value = copy.deepcopy(table)
        value.pop("identity")
        value["assignment_protocol"].pop("parent_overlap_inputs")
        normalized.append(value)
    return dict(exact_ordered_rows=sum(x == y for x, y in zip(a, b)),
                row_agreement=sum(x == y for x, y in zip(a, b)) / len(a),
                same_position_unordered_triplet_agreement=sum(set(x) == set(y) for x, y in zip(a, b)) / len(a),
                same_frame_triplet_multiset_intersection=sum((fa & fb).values()),
                corresponding_column_agreement=[
                    sum(x[c] == y[c] for x, y in zip(a, b)) / len(a) for c in range(3)],
                random_rows_equal=semantic_rows(baseline, "random_rows") == semantic_rows(candidate, "random_rows"),
                calibration_equal=semantic_rows(baseline, "calibration_triplets") ==
                                  semantic_rows(candidate, "calibration_triplets"),
                per_camera_exposure=exposures, per_key_exposure=per_key,
                all_schedule_fields_except_input_identity_equal=normalized[0] == normalized[1],
                identity_fields_excluded=["identity", "assignment_protocol.parent_overlap_inputs"],
                cost_distributions=costs, paired_position_triplet_cost_delta=distribution(delta),
                cost_formula="Original unit baseline/max + angle/pi + (1-overlap); null penalty1; no tie noise",
                note="Paired position costs compare shuffled row positions; cross-scoring separates input and selection effects")


def compare_overlaps(left, right):
    require(set(left["frames"]) == set(right["frames"]) == set(range(300)), "Frame populations differ")
    changes = []
    ratios = []
    equal_pairs = 0
    for frame in range(300):
        a, b = left["frames"][frame], right["frames"][frame]
        require(a.keys() == b.keys(), "Pair populations differ")
        for pair in sorted(a):
            diffs = differences(a[pair], b[pair], f"/frames/{frame}/{pair[0]}--{pair[1]}")
            equal_pairs += not diffs
            changes.extend(diffs)
            if a[pair]["ratio"] is not None and b[pair]["ratio"] is not None:
                ratios.append(abs(a[pair]["ratio"] - b[pair]["ratio"]))
    count = sum(len(x) for x in left["frames"].values())
    return dict(pairs=count, equal_pairs=equal_pairs, pair_agreement=equal_pairs / count,
                differing_fields=len(changes),
                conservative_ratio_absolute_delta=distribution(ratios) if ratios else None), changes


def compare_raw(left_index, right_index):
    import numpy as np
    left = {(x["camera"], x["frame"]): x for x in left_index["raw_moments"]}
    right = {(x["camera"], x["frame"]): x for x in right_index["raw_moments"]}
    require(left.keys() == right.keys(), "Raw camera/frame population differs")
    changes = []
    for key in sorted(left):
        with np.load(bound(left[key]["raw"]), allow_pickle=False) as x:
            a = x["raw_moments"]
        with np.load(bound(right[key]["raw"]), allow_pickle=False) as x:
            b = x["raw_moments"]
        require(a.shape == b.shape and a.dtype == b.dtype == np.float32, "Raw grid/dtype differs")
        if not np.array_equal(a, b, equal_nan=True):
            finite = np.isfinite(a) & np.isfinite(b)
            changes.append(dict(camera=key[0], frame=key[1],
                finite_max_abs=float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None,
                finite_pattern_equal=bool(np.array_equal(np.isfinite(a), np.isfinite(b)))))
    return dict(packets=len(left), exact_equal=len(left) - len(changes),
                changed_packets=changes,
                rule="Decoded float32 arrays exactly equal (NaNs in same positions); ZIP timestamps ignored")


def run(a):
    _, _, native = science_modules()
    ltable, rtable = read(a.baseline), read(a.candidate)
    lm, _, _, _ = manifest_metadata(bound(ltable["identity"]["manifest"]))
    rm, _, _, _ = manifest_metadata(bound(rtable["identity"]["manifest"]))
    for table, m in ((ltable, lm), (rtable, rm)):
        native.validate_schedule(table, m, table["identity"], SEED)
    lo = scalar_overlap(a.baseline_index, lm)
    ro = scalar_overlap(a.candidate_index, rm)
    require(lo["context"] == ltable["identity"] and ro["context"] == rtable["identity"],
            "Schedule and overlap contexts differ")
    report = dict(baseline=entry(a.baseline), candidate=entry(a.candidate),
                  baseline_index=entry(a.baseline_index), candidate_index=entry(a.candidate_index),
                  schedule=compare_schedules(ltable, rtable, lo, ro),
                  parent_identity_equal=lo["context"]["parent"] == ro["context"]["parent"],
                  source=entry(__file__), GPU_calls=0, HR_quality_used=False)
    report["overlap"], changes = compare_overlaps(lo, ro)
    if a.compare_raw:
        report["raw"] = compare_raw(read(a.baseline_index), read(a.candidate_index))
    exact = (report["schedule"]["row_agreement"] == 1
             and report["schedule"]["all_schedule_fields_except_input_identity_equal"]
             and report["schedule"]["random_rows_equal"] and report["schedule"]["calibration_equal"]
             and not changes and (not a.compare_raw or not report["raw"]["changed_packets"]))
    report.update(exact_scientific_replay=exact, expect_exact=a.expect_exact,
                  status="passed" if not a.expect_exact or exact else "failed_exact_comparison")
    write_new(local(a.out) / "pair_field_differences.json", changes)
    write_new(local(a.out) / "comparison.json", report)
    require(not a.expect_exact or exact, "Native replay differs; preserve comparison and investigate")
    write_new(local(a.out) / "complete.json", dict(status=report["status"],
              comparison=entry(local(a.out) / "comparison.json"), exact_scientific_replay=exact))
    no_torch()
    return dict(status=report["status"], row_agreement=report["schedule"]["row_agreement"],
                pair_agreement=report["overlap"]["pair_agreement"],
                report=entry(local(a.out) / "comparison.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "baseline-index", "candidate-index", "out"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--compare-raw", action="store_true")
    p.add_argument("--expect-exact", action="store_true")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(run(a), ensure_ascii=False))


if __name__ == "__main__":
    main()
