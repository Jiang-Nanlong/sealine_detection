"""
criterion_ranking.py — 带 Pairwise Ranking Loss 的 LINEA Criterion

在 LINEACriterion 基础上增加 loss_ranking：
  对每张图的所有候选线按 EPE 排序, 采样 (好线, 差线) pair,
  强制 score(好线) > score(差线) + margin。

这是 主线 C 的核心：将 stage2 图内排序思想内化回 detector。

用法：
  在配置中设置：
    criterionname = 'LINEACRITERION_RANKING'
    losses = ['labels', 'lines', 'ranking']
    weight_dict = {'loss_logits': 4, 'loss_line': 5, 'loss_ranking': 2}
    ranking_margin = 0.3
    ranking_num_pairs = 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from LINEA.models.linea.criterion import LINEACriterion
from LINEA.models.linea.matcher import build_matcher


class LINEACriterionWithRanking(LINEACriterion):
    """
    扩展 LINEACriterion, 增加 loss_ranking。

    loss_ranking 的核心思想:
      - 对每张图, 计算所有候选线与 GT 的 endpoint error
      - EPE 低的线应获得更高的 detector score
      - 采样 (pos, neg) pair, 用 margin ranking loss 约束
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,
                 ranking_margin=0.3, ranking_num_pairs=50):
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, losses)
        self.ranking_margin = ranking_margin
        self.ranking_num_pairs = ranking_num_pairs

    def _compute_epe(self, pred_lines, gt_line):
        """
        计算每条候选线与 GT 的 endpoint error。

        Args:
            pred_lines: [N, 4] — 候选线 (归一化坐标)
            gt_line:    [1, 4] — GT 线

        Returns:
            [N] — 每条候选线的 EPE
        """
        # 端点距离的均值
        diff = pred_lines - gt_line  # [N, 4]
        ep1 = (diff[:, 0] ** 2 + diff[:, 1] ** 2).sqrt()
        ep2 = (diff[:, 2] ** 2 + diff[:, 3] ** 2).sqrt()
        return (ep1 + ep2) / 2.0

    def loss_ranking(self, outputs, targets, indices, num_boxes):
        """
        Pairwise ranking loss: 对每张图的候选线, 好线 score > 差线 score + margin。

        使用 pred_logits 的 class-0 (海天线类) 分数做排序约束。
        """
        pred_logits = outputs['pred_logits']  # [B, N, C]
        pred_lines = outputs['pred_lines']    # [B, N, 4]
        B, N, C = pred_logits.shape

        # 取 class-0 的 logit 作为 score（海天线类）
        scores = pred_logits[:, :, 0]  # [B, N]

        total_loss = torch.tensor(0.0, device=pred_logits.device)
        num_valid = 0

        for b in range(B):
            # GT 线 — MU-SID 每张图只有 1 条 GT
            gt_lines_b = targets[b]['lines']  # [num_gt, 4]
            if gt_lines_b.shape[0] == 0:
                continue

            gt_line = gt_lines_b[0:1]  # [1, 4]
            pred_b = pred_lines[b].detach()  # [N, 4] — detach 坐标，只约束分数
            scores_b = scores[b]  # [N]

            # 计算所有候选线的 EPE
            epe = self._compute_epe(pred_b, gt_line)  # [N]

            # 按 EPE 排序, 取 top-K 好线和 bottom-K 差线
            K = min(self.ranking_num_pairs, N // 4)
            if K < 2:
                continue

            _, sorted_idx = epe.sort()
            pos_idx = sorted_idx[:K]      # EPE 最小的 K 条 = 好线
            neg_idx = sorted_idx[-K:]     # EPE 最大的 K 条 = 差线

            pos_scores = scores_b[pos_idx]  # [K]
            neg_scores = scores_b[neg_idx]  # [K]

            # Margin ranking loss: pos_score > neg_score + margin
            # 每条好线 vs 每条差线 → K*K pairs 太多, 用 1:1 配对
            loss_b = F.margin_ranking_loss(
                pos_scores, neg_scores,
                target=torch.ones_like(pos_scores),
                margin=self.ranking_margin,
                reduction='mean',
            )
            total_loss = total_loss + loss_b
            num_valid += 1

        if num_valid > 0:
            total_loss = total_loss / num_valid

        return {'loss_ranking': total_loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'lines': self.loss_lines,
            'lmap': self.loss_lmap,
            'ranking': self.loss_ranking,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


def build_criterion_ranking(args):
    num_classes = args.num_classes
    matcher = build_matcher(args)

    criterion = LINEACriterionWithRanking(
        num_classes,
        matcher=matcher,
        weight_dict=args.weight_dict,
        focal_alpha=args.focal_alpha,
        losses=args.losses,
        ranking_margin=getattr(args, 'ranking_margin', 0.3),
        ranking_num_pairs=getattr(args, 'ranking_num_pairs', 50),
    )
    return criterion
