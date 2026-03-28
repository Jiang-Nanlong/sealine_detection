import os, shutil
import pandas as pd

CSV_PATH = r"eval_outputs/eval_test.csv"
OUT_DIR = r"eval_outputs/failures_dump"
OUTLIER_VIS_DIR = r"eval_outputs/outliers_test"  # 你脚本生成的 GT/Pred 叠加图

# 你可以先沿用脚本里的阈值
THRESH = {
    "edge_y_orig": 20.0,
    "line_dist_orig": 20.0,
    "rho_err_orig": 20.0,
    "theta_err_deg": 2.0,
}

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns.tolist())
print("N =", len(df))

# 选坏样本（任何一个指标超阈就算）
bad = df[
    (df["edge_y_orig"] >= THRESH["edge_y_orig"]) |
    (df["line_dist_orig"] >= THRESH["line_dist_orig"]) |
    (df["rho_err_orig"] >= THRESH["rho_err_orig"]) |
    (df["theta_err_deg"] >= THRESH["theta_err_deg"])
].copy()

print("Bad N =", len(bad))

# 按你最关心的指标排序：通常 edge_y_orig / line_dist_orig 更直观
bad = bad.sort_values("edge_y_orig", ascending=False)

# 导出清单
bad.to_csv(os.path.join(OUT_DIR, "bad_list_sorted.csv"), index=False)

# 把可视化结果复制出来（优先用你已经画好的 outliers png）
cnt = 0
for _, r in bad.head(300).iterrows():  # 先拷前300个，够你分析一轮
    stem = os.path.splitext(os.path.basename(r["img_name"]))[0]
    vis = os.path.join(OUTLIER_VIS_DIR, stem + ".png")
    if os.path.exists(vis):
        shutil.copy(vis, os.path.join(OUT_DIR, f"{cnt:04d}_{stem}.png"))
        cnt += 1

print("Copied", cnt, "visualizations to", OUT_DIR)
