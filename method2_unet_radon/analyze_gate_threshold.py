import pandas as pd
import numpy as np

CSV_PATH = r"eval_full_outputs\gate_stats.csv"

df = pd.read_csv(CSV_PATH)
print("N =", len(df))
print("Columns:", list(df.columns))

# 1) refine 是否真的跑过（是否有有效值）
leak = pd.to_numeric(df.get("leak_ratio", np.nan), errors="coerce")
rmse = pd.to_numeric(df.get("boundary_rmse", np.nan), errors="coerce")
inlr = pd.to_numeric(df.get("inlier_ratio", np.nan), errors="coerce")

attempted = leak.notna() | rmse.notna() | inlr.notna()
print("\n---- Attempted refine ----")
print("attempted =", int(attempted.sum()), "/", len(df))
if "refine_ok" in df.columns:
    print("refine_ok=1:", int((df["refine_ok"] == 1).sum()))
if "used_ref" in df.columns:
    print("used_ref=1:", int((df["used_ref"] == 1).sum()))

def q(series, name):
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if s.size == 0:
        print(name, ": EMPTY")
        return
    for p in [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0]:
        print(f"{name} p{int(p*100):02d} = {np.quantile(s,p):.6f}")
    print(f"{name} mean = {s.mean():.6f}")

print("\n---- conf distribution ----")
q(df["conf"], "conf")

# 2) gate 指标分布（只统计 attempted 子集）
print("\n---- gate metric distribution (attempted only) ----")
q(leak[attempted], "leak_ratio")
q(rmse[attempted], "boundary_rmse")
q(inlr[attempted], "inlier_ratio")

# 3) 给一个“按分位数定阈值”的建议（你可按目标覆盖率调整）
print("\n---- Suggested (quantile-based) ----")
conf = pd.to_numeric(df["conf"], errors="coerce").dropna().to_numpy()
for cover in [0.10, 0.20, 0.30, 0.40, 0.50]:
    thr = float(np.quantile(conf, cover))
    print(f"CONF_REFINE_THRESH (cover lowest {int(cover*100)}% conf) ~= {thr:.6f}")

# gate 建议：leak/rmse 用 p95 上限；inlier 用 p05 下限（或固定 0.30~0.40）
leak_s = leak[attempted].dropna().to_numpy()
rmse_s = rmse[attempted].dropna().to_numpy()
inlr_s = inlr[attempted].dropna().to_numpy()
if leak_s.size: print("GATE_LEAK_RATIO_MAX ~= p95 =", float(np.quantile(leak_s, 0.95)))
if rmse_s.size: print("GATE_BOUNDARY_RMSE_MAX ~= p95 =", float(np.quantile(rmse_s, 0.95)))
if inlr_s.size: print("RANSAC_MIN_INLIER_RATIO ~= p05 =", float(np.quantile(inlr_s, 0.05)))
