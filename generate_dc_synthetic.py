"""
Synthetic DC parametric test data generator.

Produces data that exercises every technique in the methodology:
  - correlated / redundant tests        (IIL_A vs IIL_B, IDDQ vectors)
  - never-fail tests                    (VIH, CONT)
  - predictable expensive tests         (IDDQ_V2..V5 from IDDQ_V1, IIL, IDD)
  - lot-to-lot process drift            (two "hot" lots with higher leakage)
  - wafer-edge spatial effect           (leakage rises with die radius)
  - tester/site offset                  (one site reads IIL high)
  - latent defects                      (in-spec multivariate outliers that fail burn-in / RMA)
  - settling-time behaviour             (exponential settling + noise vs averaging)

Outputs (CSV):
  test_catalog.csv      one row per test: units, limits, typical test time
  dc_test_results.csv   one row per device: location, tester, all DC values, bins, downstream labels
  settling_sweep.csv    repeated measurements vs wait time and sample count
"""
import argparse
import os

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
p.add_argument("--lots", type=int, default=12)
p.add_argument("--wafers", type=int, default=5)
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()
rng = np.random.default_rng(args.seed)
os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- test catalog
catalog = pd.DataFrame([
    # test,            group,        unit, lo,     hi,     time_ms
    ("CONT_VDIODE",    "Continuity", "V",  -0.90,  -0.40,  18.0),
    ("IIL_A",          "Leakage",    "nA", -1000,  1000,   22.0),
    ("IIL_B",          "Leakage",    "nA", -1000,  1000,   22.0),
    ("IIH_A",          "Leakage",    "nA", -1000,  1000,   22.0),
    ("IIH_B",          "Leakage",    "nA", -1000,  1000,   22.0),
    ("VOH",            "Output",     "V",   2.40,  None,   15.0),
    ("VOL",            "Output",     "V",   None,  0.40,   15.0),
    ("IDDQ_V1",        "IDDQ",       "uA",  None,  5.00,   30.0),
    ("IDDQ_V2",        "IDDQ",       "uA",  None,  5.00,   30.0),
    ("IDDQ_V3",        "IDDQ",       "uA",  None,  5.00,   30.0),
    ("IDDQ_V4",        "IDDQ",       "uA",  None,  5.00,   30.0),
    ("IDDQ_V5",        "IDDQ",       "uA",  None,  5.00,   30.0),
    ("IDD_STATIC",     "Supply",     "mA",  None,  12.0,   25.0),
    ("VIL",            "Threshold",  "V",   0.80,  None,   35.0),
    ("VIH",            "Threshold",  "V",   None,  2.00,   35.0),
    ("IOS",            "Output",     "mA",  20.0,  120.0,  24.0),
], columns=["test", "group", "unit", "lo_limit", "hi_limit", "test_time_ms"])
catalog.to_csv(f"{args.out}/test_catalog.csv", index=False)

# ---------------------------------------------------------------- die map
R = 12  # die radius in grid units
xs, ys = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1))
mask = xs**2 + ys**2 <= R**2
die_x, die_y = xs[mask], ys[mask]
n_die = len(die_x)

hot_lots = {3, 8}                     # process drift: higher leakage
testers = ["T01", "T02", "T03"]
rows = []
for lot in range(1, args.lots + 1):
    lot_shift = rng.normal(0, 0.08) + (0.45 if lot in hot_lots else 0.0)
    for w in range(1, args.wafers + 1):
        wafer_shift = rng.normal(0, 0.05)
        tester = testers[(lot + w) % 3]
        temp = 25 if lot % 4 else 90  # every 4th lot tested hot
        rows.append(pd.DataFrame({
            "lot": f"L{lot:03d}", "wafer": w, "die_x": die_x, "die_y": die_y,
            "tester": tester, "site": rng.integers(1, 5, n_die), "temp_C": temp,
            "_lot_shift": lot_shift, "_wafer_shift": wafer_shift,
        }))
df = pd.concat(rows, ignore_index=True)
n = len(df)
df.insert(0, "ecid", [f"E{i:07d}" for i in range(n)])

# ---------------------------------------------------------------- latent process variables
radius = np.sqrt(df.die_x**2 + df.die_y**2) / R                 # 0 centre .. 1 edge
leak = (df._lot_shift + df._wafer_shift + 0.6 * radius**3
        + rng.normal(0, 0.25, n))                                # leakage factor
leak += np.where(df.temp_C == 90, 0.7, 0.0)
vt = rng.normal(0, 1, n)                                         # threshold factor
drive = rng.normal(0, 1, n) - 0.3 * vt                           # drive-strength factor

# defects: hard fails (out of spec) and latent (in spec but abnormal)
hard = rng.random(n) < 0.012
latent = rng.random(n) < 0.0015

# ---------------------------------------------------------------- measurements
def noise(s): return rng.normal(0, s, n)

site_offset = np.where((df.tester == "T02") & (df.site == 3), 60.0, 0.0)   # equipment artefact

df["CONT_VDIODE"] = -0.65 + noise(0.03)
iil = 120 * np.exp(leak) + noise(15)
df["IIL_A"] = -(iil + site_offset)
df["IIL_B"] = -(iil * 1.02 + noise(12))                         # near-duplicate of IIL_A
df["IIH_A"] = iil * 0.9 + noise(18)
df["IIH_B"] = iil * 0.92 + noise(18)
df["VOH"] = 2.95 + 0.08 * drive + noise(0.03)
df["VOL"] = 0.22 - 0.03 * drive + noise(0.02)
iddq = 0.9 * np.exp(0.9 * leak) + noise(0.05)
for k in range(1, 6):                                           # vectors track each other
    df[f"IDDQ_V{k}"] = iddq * (1 + 0.04 * (k - 3)) + noise(0.06)
df["IDD_STATIC"] = 7.5 + 1.1 * leak + 0.3 * drive + noise(0.25)
df["VIL"] = 1.25 + 0.06 * vt + noise(0.03)
df["VIH"] = 1.55 + 0.06 * vt + noise(0.03)                      # never fails
df["IOS"] = 65 + 9 * drive + noise(3)

# hard defects: push one or two test families out of spec
fam = rng.choice(["leak", "iddq", "out", "ios", "cont"], n, p=[.3, .35, .15, .15, .05])
hi = hard
df.loc[hi & (fam == "leak"), ["IIL_A", "IIL_B"]] *= rng.uniform(9, 20, (hi & (fam == "leak")).sum())[:, None]
df.loc[hi & (fam == "iddq"), [f"IDDQ_V{k}" for k in range(1, 6)]] *= rng.uniform(5, 12, (hi & (fam == "iddq")).sum())[:, None]
df.loc[hi & (fam == "out"), "VOH"] -= rng.uniform(0.6, 1.0, (hi & (fam == "out")).sum())
df.loc[hi & (fam == "ios"), "IOS"] = rng.uniform(5, 18, (hi & (fam == "ios")).sum())
df.loc[hi & (fam == "cont"), "CONT_VDIODE"] = rng.choice([-0.05, -1.4], (hi & (fam == "cont")).sum())
# a few IDDQ fails only on vector 4 (unique-fail test: removing it causes escapes)
v4only = (~hard) & (rng.random(n) < 0.0008)
df.loc[v4only, "IDDQ_V4"] = rng.uniform(5.5, 9.0, v4only.sum())

# latent defects: in spec, but IDDQ high relative to leakage (multivariate outlier)
df.loc[latent, [f"IDDQ_V{k}" for k in range(1, 6)]] += rng.uniform(1.5, 2.5, latent.sum())[:, None]
df.loc[latent, "IDD_STATIC"] += rng.uniform(1.0, 2.0, latent.sum())

# ---------------------------------------------------------------- limits, bins, test time
tests = catalog.test.tolist()
fail_first = np.full(n, "", dtype=object)
for _, t in catalog.iterrows():
    v = df[t.test]
    f = np.zeros(n, bool)
    if pd.notna(t.lo_limit): f |= v < t.lo_limit
    if pd.notna(t.hi_limit): f |= v > t.hi_limit
    df[f"{t.test}_PF"] = np.where(f, "F", "P")
    fail_first = np.where((fail_first == "") & f, t.test, fail_first)
df["first_fail_test"] = fail_first
df["sort_pass"] = (fail_first == "").astype(int)
bin_map = {"Continuity": 2, "Leakage": 3, "Output": 4, "IDDQ": 5, "Supply": 6, "Threshold": 7}
grp = dict(zip(catalog.test, catalog.group))
df["hard_bin"] = [1 if t == "" else bin_map[grp[t]] for t in fail_first]

# full-flow DC test time per device (stop on first fail), with per-site jitter
tt = catalog.set_index("test").test_time_ms
cum = np.cumsum([tt[t] for t in tests])
idx = {t: i for i, t in enumerate(tests)}
df["dc_test_time_ms"] = [round(cum[idx[t]] if t else cum[-1], 1) for t in fail_first] + rng.normal(0, 2, n).round(1)

# ---------------------------------------------------------------- downstream labels (only for sort-pass parts)
sp = df.sort_pass == 1
p_ft = 0.002 + 0.25 * latent
df["final_test_fail"] = np.where(sp, rng.random(n) < p_ft, np.nan)
p_bi = 0.0008 + 0.45 * latent + 0.002 * (df._lot_shift > 0.3)
df["burnin_fail"] = np.where(sp, rng.random(n) < p_bi, np.nan)
p_rma = 0.00005 + 0.15 * latent
df["rma"] = np.where(sp, rng.random(n) < p_rma, np.nan)
for c in ["final_test_fail", "burnin_fail", "rma"]:
    df[c] = df[c].astype("Int64")

df["is_latent_defect_truth"] = latent.astype(int)  # ground truth, for evaluating PAT only
df = df.drop(columns=["_lot_shift", "_wafer_shift"])
num = df.select_dtypes("float").columns
df[num] = df[num].round(4)
df.to_csv(f"{args.out}/dc_test_results.csv", index=False)

# ---------------------------------------------------------------- settling-time sweep
sweep_tests = {  # true settling constant tau (ms), initial error, noise per sample
    "IIL_A":   (0.80, 400.0, 25.0),
    "IDDQ_V1": (1.50, 3.0, 0.20),
    "VOH":     (0.15, 0.30, 0.012),
    "IDD_STATIC": (0.60, 2.5, 0.10),
}
waits = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
samples = [1, 2, 4, 8, 16]
recs = []
dev_true = {t: rng.normal(1, 0.1, 30) for t in sweep_tests}
for t, (tau, e0, sig) in sweep_tests.items():
    base = {"IIL_A": -150, "IDDQ_V1": 0.9, "VOH": 2.95, "IDD_STATIC": 7.5}[t]
    for d in range(30):
        true = base * dev_true[t][d]
        for temp in (25, 90):
            tau_t = tau * (1.4 if temp == 90 else 1.0)
            for wt in waits:
                for s in samples:
                    for r in range(10):
                        meas = true + e0 * np.exp(-wt / tau_t) + rng.normal(0, sig / np.sqrt(s))
                        recs.append((t, d + 1, temp, wt, s, r + 1, round(true, 5), round(meas, 5)))
sw = pd.DataFrame(recs, columns=["test", "device", "temp_C", "wait_ms", "n_samples", "repeat",
                                 "reference_value", "measured_value"])
sw.to_csv(f"{args.out}/settling_sweep.csv", index=False)

print(f"devices: {n:,}  sort yield: {df.sort_pass.mean():.2%}  latent: {latent.sum()}  v4-only fails: {v4only.sum()}")
print(f"settling sweep rows: {len(sw):,}")
