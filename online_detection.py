from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
from tool.JoyTool import *
from effdet_infer import *
from tool.toCOCO import *
from tool.cal_complexity import get_complexity


def load_efficientdet(model_path, model_name=MODEL_NAME):
    model = create_model(
        model_name,
        bench_task='predict',
        pretrained=False,
        pretrained_backbone=False
    )
    model.model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


s1_model = load_efficientdet('checkpoints/SuperNet_epoch_28_mAP_34_90.pth').to(device)
s2_model = load_efficientdet('checkpoints/SuperNet_epoch_28_mAP_34_90.pth').to(device)


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


def broaden_RoI(x, y, w, h, h_img, w_img, scale):  # 把RoI扩大scale倍的函数，边界太过于贴近目标检测效果差
    # 扩大宽度和高度
    new_w = w * scale
    new_h = h * scale

    # 调整左上角坐标以保持中心对齐
    new_x = x - (new_w - w) / 2
    new_y = y - (new_h - h) / 2

    # 边界检查和调整
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + new_w > w_img:
        new_w = w_img - new_x
    if new_y + new_h > h_img:
        new_h = h_img - new_y

    return int(new_x), int(new_y), int(new_w), int(new_h)


def getRoI(frame, back_sub):
    if frame is None:
        print("无法读取图像文件")
        return

    # 应用背景减除器
    fg_mask = back_sub.apply(frame)

    # 后处理 - 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    complexity_score = get_complexity(contours)
    RoI_xywh = []  # 存储RoI的坐标
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        # 忽略小的轮廓，这里可以根据实际情况调整最小面q
        if cv2.contourArea(cnt) < 500:  # 最小轮廓面积
            continue
        # 获取RoI对应的坐标
        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = broaden_RoI(x, y, w, h, h_img, w_img, 1.05)
        RoI_xywh.append(
            {"RoI": frame[y:y + h, x:x + w], "xywh": [x, y, w, h]})
    return RoI_xywh, complexity_score


def example_miner(large_model, small_model, frame_path, gt_path, infer_ratio=1.0, class_id=2):
    all_data = []
    gt_data = defaultdict(list)
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=200,  # 历史帧数
        varThreshold=16,  # 灵敏度阈值
        detectShadows=False  # 是否检测阴影
    )
    # 读取GT数据
    with open(gt_path, 'r') as f:
        for line in f:
            fid, _, x, y, w, h, *_ = map(float, line.strip().split(','))
            # print(fid)
            gt_data[int(fid)].append([x, y, w, h])
    gt_data = defaultdict(list, {k: v for k, v in gt_data.items() if 1 <= k <= int(infer_ratio * len(gt_data))})
    frame_id = 0
    frames_list = os.listdir(frame_path)  # 存储图片列表
    frames_list.sort()  # 排序是为了按照帧的顺序进行，在linux系统可能会乱序
    frames_list = frames_list[:int(infer_ratio * len(frames_list))]
    for filename in tqdm(frames_list[:int(len(frames_list) / 3)]):
        frame_id += 1
        large_xywh = []
        small_xywh = []
        temp_data = []
        image_file = os.path.join(frame_path, filename)
        frame = cv2.imread(image_file)
        RoI_xywh, complexity_score = getRoI(frame, back_sub)
        for RoIs in RoI_xywh:
            RoI_img = RoIs["RoI"]
            x, y, w, h = RoIs["xywh"]
            slot_xywh = []
            # 大模型
            results = large_model(RoI_img, imgsz=160, verbose=False)
            for result in results:
                boxes = result.boxes  # 获取跟踪结果中的框信息
                for box in boxes:
                    class_idx = int(box.cls.item())
                    if class_idx != class_id:  # 剔除非目标类型的结果
                        continue
                    x1, y1, w1, h1 = box.xywh[0]  # 获取框的 [x, y, w, h] 坐标
                    x1, y1, w1, h1 = int(x1 - w1 / 2), int(y1 - h1 / 2), int(w1), int(h1)  # 转化为左上角坐标
                    large_xywh.append([x + x1, y + y1, w1, h1])
                    slot_xywh.append({"bbox": [int(x1), int(y1), int(w1), int(h1)], "category_id": 3})
            if results:
                temp_data.append({"image": RoI_img, "annotations": slot_xywh})

            # 小模型
            results = eff_infer(RoI_img, small_model)
            for box in results:
                x1, y1, w1, h1 = box.astype(int)  # 获取框的 [x, y, w, h] 坐标
                w1, h1 = w1 - x1, h1 - y1
                small_xywh.append([x + x1, y + y1, w1, h1])
        large_score = evaluate_detection(large_xywh, gt_data[frame_id])  # 计算精度
        small_score = evaluate_detection(small_xywh, gt_data[frame_id])  # 计算精度
        if complexity_score > 0.17:
            all_data.extend(temp_data)
    return all_data


def online_infer(model, scale, frame_path, gt_path, infer_ratio=1.0, class_id=2):
    all_result = []
    gt_data = defaultdict(list)
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=200,  # 历史帧数
        varThreshold=16,  # 灵敏度阈值
        detectShadows=False  # 是否检测阴影
    )
    # 读取GT数据
    with open(gt_path, 'r') as f:
        for line in f:
            fid, _, x, y, w, h, *_ = map(float, line.strip().split(','))
            # print(fid)
            gt_data[int(fid)].append([x, y, w, h])
    gt_data = defaultdict(list, {k: v for k, v in gt_data.items() if 1 <= k <= int(infer_ratio * len(gt_data))})
    frame_id = 0
    frames_list = os.listdir(frame_path)  # 存储图片列表
    frames_list.sort()  # 排序是为了按照帧的顺序进行，在linux系统可能会乱序
    frames_list = frames_list[:int(infer_ratio * len(frames_list))]
    for filename in tqdm(frames_list):
        frame_id += 1
        xywh = []
        image_file = os.path.join(frame_path, filename)
        frame = cv2.imread(image_file)
        RoI_xywh, _ = getRoI(frame, back_sub)
        for RoIs in RoI_xywh:
            RoI_img = RoIs["RoI"]
            x, y, w, h = RoIs["xywh"]
            if scale == 'large':
                results = model(RoI_img, imgsz=160, verbose=False)
                for result in results:
                    boxes = result.boxes  # 获取跟踪结果中的框信息
                    for box in boxes:
                        class_idx = int(box.cls.item())
                        if class_idx != class_id:  # 剔除非目标类型的结果
                            continue
                        x1, y1, w1, h1 = box.xywh[0]  # 获取框的 [x, y, w, h] 坐标
                        x1, y1, w1, h1 = int(x1 - w1 / 2), int(y1 - h1 / 2), int(w1), int(h1)  # 转化为左上角坐标
                        xywh.append([x + x1, y + y1, w1, h1])
            else:
                if frame_id < 500:
                    smodel = model
                elif 500 <= frame_id < 1000:
                    smodel = s1_model
                else:
                    smodel = s2_model

                results = eff_infer(RoI_img, smodel)
                for box in results:
                    x1, y1, w1, h1 = box.astype(int)  # 获取框的 [x, y, w, h] 坐标
                    w1, h1 = w1 - x1, h1 - y1
                    xywh.append([x + x1, y + y1, w1, h1])
        f1_score = evaluate_detection(xywh, gt_data[frame_id])  # 计算精度
        all_result.append(f1_score)
    return all_result, list(gt_data.keys())


def plot_single_frame(model, model_name, image_index, class_id=2):
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=200,  # 历史帧数
        varThreshold=16,  # 灵敏度阈值
        detectShadows=False  # 是否检测阴影
    )
    frame = cv2.imread(frame_path + '/' + str(image_index).zfill(5) + '.jpg')
    xywh = []
    RoI_xywh = getRoI(frame, back_sub)
    for RoIs in RoI_xywh:
        results = model(RoIs['RoI'], imgsz=160, verbose=False)
        x, y, w, h = RoIs['xywh']
        for result in results:
            boxes = result.boxes  # 获取跟踪结果中的框信息
            for box in boxes:
                class_idx = int(box.cls.item())
                if class_idx != class_id:  # 剔除非目标类型的结果
                    continue
                x1, y1, w1, h1 = box.xywh[0]  # 获取框的 [x, y, w, h] 坐标
                x1, y1, w1, h1 = int(x1 - w1 / 2), int(y1 - h1 / 2), int(w1), int(h1)  # 转化为左上角坐标
                xywh.append([x + x1, y + y1, w1, h1])

                # 绘制检测框
                cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w1, y + y1 + h1), (0, 255, 0), 2)

                # 添加类别标签
                # label = f"Class {class_idx}"
                # cv2.putText(frame, label, (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

    # 显示结果
    cv2.imwrite('output/img/' + str(image_index).zfill(5) + '_' + model_name + '.jpg', frame)
    # cv2.imshow("Detection", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':

    '''
    0.替换成effdet中的resdet50模型
    1.找到那些二者差距大的样本，将大模型的输出作为标签重训练小的模型，最终的效果就是让小模型和大模型的精度一致。注意在构造重训练数据集的时候可能需要加入一些正确的样本。
    2.提出一种技术：在不知道二者真实表现的情况下识别出这些关键样本。之前的方案：特征距离。
    3.提出一些想法：如何为重训练任务确定需要训练的epoch（可选：以及如何将这些epoch分布在不同的时隙中），以使其不影响推理的前提下最大化精度
    4.使用动量知识蒸馏解决灾难性遗忘的问题（困难，等前面先完成了再说吧）
    '''
    video_index = 'MVI_4077x'  # 数据集不同部分，可选MVI_39031/MVI_39211MVI_40773
    frame_path = "dataset/" + video_index
    gt_path = "labels/" + video_index + ".txt"
    large_model = YOLO('model/yolov8l.pt')
    small_model = load_efficientdet('checkpoints/SuperNet_epoch_28_mAP_34_90.pth').to(device)
    infer_ratio = 1  # 在线推理帧的比例，用于测试，例如0.1就是只推理10%的数据集部分

    # is_run_online_infer = True
    is_hard_example_miner = True
    is_run_online_infer = False
    is_plot_frame_vs = False
    # is_plot_frame_vs = False
    if is_hard_example_miner:
        results = example_miner(large_model, small_model, frame_path, gt_path, infer_ratio)
        convert_to_coco(results, 'example_data', 0.8)
    if is_run_online_infer:
        large_result, img_id = online_infer(large_model, 'large', frame_path, gt_path, infer_ratio)
        small_result, _ = online_infer(small_model, 'small', frame_path, gt_path, infer_ratio)
        write_multi_list_to_txt(
            [img_id, large_result, small_result, [a - b for a, b in zip(large_result, small_result)]],
            ['id', 'large', 'small', 'large-small'])  # 将结果保存在本地
    if is_plot_frame_vs:
        plot_frame_list = [66, 417, 450]
        for i in range(len(plot_frame_list)):
            plot_single_frame(large_model, 'v8L', plot_frame_list[i])
            plot_single_frame(small_model, 'v8S', plot_frame_list[i])
