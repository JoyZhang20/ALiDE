import numpy as np


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)。
    box1 和 box2 是形式为 [x, y, w, h] 的列表。
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集的坐标
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # 计算交集面积
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 计算并集面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    return inter_area / union_area if union_area != 0 else 0


def evaluate_detection(result, gt, iou_threshold=0.5):
    """
    评估目标检测结果与真实标注的精度。
    result 是预测框列表，gt 是真实框列表，格式为 [x, y, w, h]。
    iou_threshold 是IoU阈值，通常设为0.5。
    """
    if not gt:
        return -1
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # 标记哪些gt已经被匹配
    matched_gt = [False] * len(gt)

    # 对每个预测框进行评估
    for pred_box in result:
        best_iou = 0
        best_match_index = -1

        # 找到与预测框重叠度最大的真实框
        for idx, true_box in enumerate(gt):
            if not matched_gt[idx]:  # 如果该真实框还没有被匹配
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_index = idx

        # 如果最好的IoU大于阈值，则为TP，且该真实框被匹配
        if best_iou >= iou_threshold and best_match_index != -1:
            tp += 1
            matched_gt[best_match_index] = True  # 标记为已匹配
        else:
            fp += 1  # 如果没有匹配，则为FP

    # 对于所有未匹配的真实框，它们是FN
    fn = matched_gt.count(False)

    # 计算精度和召回率
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    # 计算F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def compute_ap(detect_data, gt_data, iou_threshold=0.3):
    all_scores = []
    all_matches = []
    total_gt_count = 0

    for fid in sorted(detect_data.keys()):
        gt_boxes = gt_data.get(fid, [])
        pred_boxes = detect_data[fid]
        # gt_boxes = gt_data[fid]
        # pred_boxes = detect_data.get(fid, [])
        total_gt_count += len(gt_boxes)

        matched_gt = [False] * len(gt_boxes)
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        for pred_box in pred_boxes_sorted:
            box = pred_box[:4]
            score = pred_box[4]
            best_iou = 0
            best_gt_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if not matched_gt[idx]:
                    iou = calculate_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt[best_gt_idx] = True
                all_matches.append(1)  # TP
            else:
                all_matches.append(0)  # FP

            all_scores.append(score)

    if total_gt_count == 0:
        return 1.0 if len(all_matches) == 0 else 0.0

    # 排序
    sorted_indices = np.argsort(all_scores)[::-1]
    all_matches = np.array([all_matches[i] for i in sorted_indices])

    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(1 - all_matches)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recalls = tp_cumsum / total_gt_count

    # ⬇️ COCO风格：使用逐点PR曲线积分（更平滑、更高）
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # 保证 precision 单调递减（从后向前更新）
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 计算PR曲线下的面积
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return ap
