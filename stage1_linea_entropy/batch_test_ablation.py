"""
batch_test_ablation.py — 批量测试消融实验（Enhanced v2 / MSLEP only / MSLEP+SAI）

用法：
  在 PyCharm 中直接运行本文件，会依次测试 3 个模型并将结果分别保存。

输出目录结构：
  stage1_linea_entropy/test_outputs_v3/
    entropy_b_enhanced__linea_entropy_b_enhanced_musid_v2_e130/
    entropy_b_enhanced__ablation_mslep_only_musid_e130/
    entropy_b_enhanced__ablation_mslep_sai_musid_e130/
"""

import stage1_linea_entropy.test_linea_stage1 as T

# ============================================================
# 定义 3 组测试配置
# ============================================================
TEST_CONFIGS = [
    {
        "name": "Enhanced v2 (MSLEP+SAI+EGAB+HASH+Ranking)",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "stage1_linea_entropy/configs/linea_entropy_b_enhanced_musid.py",
        "WEIGHTS_PATH": "output/linea_entropy_b_enhanced_musid_v2_e130/best_checkpoint.pth",
    },
    {
        "name": "Ablation: MSLEP only",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "stage1_linea_entropy/configs/ablation_mslep_only_musid.py",
        "WEIGHTS_PATH": "output/ablation_mslep_only_musid_e130/best_checkpoint.pth",
    },
    {
        "name": "Ablation: MSLEP + SAI",
        "MODE": "entropy_b_enhanced",
        "CONFIG_FILE": "stage1_linea_entropy/configs/ablation_mslep_sai_musid.py",
        "WEIGHTS_PATH": "output/ablation_mslep_sai_musid_e130/best_checkpoint.pth",
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
