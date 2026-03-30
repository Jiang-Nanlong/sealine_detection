"""
batch_test_ablation.py — 批量测试消融实验（全部 7 组，统一指标重跑）

用法：
  在 PyCharm 中直接运行本文件，会依次测试 7 个模型并将结果分别保存。

输出目录：method1_linea_entropy/test_outputs_v5/
"""

import method1_linea_entropy.test_linea_stage1 as T

# 改为 v4 避免覆盖旧结果
T.SAVE_ROOT = "method1_linea_entropy/test_outputs_v5"

# ============================================================
# 定义 7 组消融实验配置
# ============================================================
TEST_CONFIGS = [
    {
        "name": "1. Baseline (LINEA-L, no entropy)",
        "MODE": "baseline",
        "CONFIG_FILE": "method1_linea_entropy/configs/linea_baseline_musid.py",
        "WEIGHTS_PATH": "output/linea_baseline_musid_e100/best_checkpoint.pth",
    },
    {
        "name": "2. +单尺度熵注入 (LINEA_ENTROPY)",
        "MODE": "entropy",
        "CONFIG_FILE": "method1_linea_entropy/configs/linea_entropy_musid.py",
        "WEIGHTS_PATH": "output/linea_entropy_musid_e100/best_checkpoint.pth",
    },
    {
        "name": "3. +MSLEP only",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "method1_linea_entropy/configs/ablation_mslep_only_musid.py",
        "WEIGHTS_PATH": "output/ablation_mslep_only_musid_e130/best_checkpoint.pth",
    },
    {
        "name": "4. +MSLEP + SAI",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "method1_linea_entropy/configs/ablation_mslep_sai_musid.py",
        "WEIGHTS_PATH": "output/ablation_mslep_sai_musid_e130/best_checkpoint.pth",
    },
    {
        "name": "5. +MSLEP+SAI+EGAB (v2 weights, HASH off at inference)",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "method1_linea_entropy/configs/ablation_mslep_sai_egab_musid.py",
        "WEIGHTS_PATH": "output/linea_entropy_b_enhanced_musid_v2_e130/best_checkpoint.pth",
    },
    {
        "name": "6. +MSLEP+SAI+EGAB+HASH (Enhanced v1)",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "method1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py",
        "WEIGHTS_PATH": "output/linea_entropy_b_enhanced_musid_v1_e150/best_checkpoint.pth",
    },
    {
        "name": "7. +MSLEP+SAI+EGAB+HASH+Ranking (Enhanced v2)",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "method1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py",
        "WEIGHTS_PATH": "output/linea_entropy_b_enhanced_musid_v2_e130/best_checkpoint.pth",
    },
]


def main():
    for i, cfg in enumerate(TEST_CONFIGS):
        print("\n" + "#" * 70)
        print(f"# [{i+1}/{len(TEST_CONFIGS)}] {cfg['name']}")
        print(f"#   CONFIG:  {cfg['CONFIG_FILE']}")
        print(f"#   WEIGHTS: {cfg['WEIGHTS_PATH']}")
        print("#" * 70)

        # 覆盖全局变量
        T.MODE = cfg["MODE"]
        T.CONFIG_FILE = cfg["CONFIG_FILE"]
        T.WEIGHTS_PATH = cfg["WEIGHTS_PATH"]

        T.main()

    print("\n" + "=" * 70)
    print("  全部测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
