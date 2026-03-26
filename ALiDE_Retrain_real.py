import random
random.seed(42)
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from tool.JoyTool import *
from effdet_infer_NAS import *
from tool.toCOCO import *
from tool.toCOCO import convert_to_coco
from effdet_train_NAS import start_retrain
from tool.getRoI import getRoI
from tool.cal_accuracy import evaluate_detection, compute_ap
from tool.cal_complexity import retrain_regressor, Init_regressor
import shutil
import time
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
retrain_epoch = 10
get_large_model_result = False  # 设置为True会同时生成大模型的结果，方便对比
yolo_class_id = 2  # yolo中目标种类
coco_class_id = 3
complexity_threshold = 0.05  # 进行 hard_example 采样时图像复杂度的阈值
normal_example_possibility = 0.3
video_index = 'S0101'
frame_path = "dataset/" + video_index
gt_path = "labels/" + video_index + ".txt"
retrain_interval = 500  # 指的是重训练间隔的帧数
frames_list = os.listdir(frame_path)  # 存储图片列表
frames_list.sort()  # 排序是为了按照帧的顺序进行，在linux系统可能会乱序
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=200,  # 历史帧数
    varThreshold=16,  # 灵敏度阈值
    detectShadows=False  # 是否检测阴影
)
subnet_idx = 7

large_f1_score = []
large_all_xywh = defaultdict(list)
small_all_xywh = defaultdict(list)
hard_example = []
feature_num = 5

gt_data = defaultdict(list)
# with open(gt_path, 'r') as f:
#     for line in f:
#         fid, _, x, y, w, h, *_ = map(float, line.strip().split(','))
#         gt_data[int(fid)].append([x, y, w, h])

large_model = YOLO('model/yolov8l.pt')
model = load_efficientdet()


def Init():
    folder_path = 'example_data'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除文件夹及其所有内容
        except Exception as e:
            print(f"删除 {file_path} 失败：{e}")
    print("初始化清除数据集成功！")


if __name__ == '__main__':
    gpu_utilize = []
    small_f1_score = []
    all_time = []
    predict_X = []
    predict_Y = []
    Init()
    Init_regressor(feature_num)
    small_all_ap = []
    results_ap = []
    for frame_id in tqdm(range(1, len(frames_list) + 1)):

        # 执行重训练
        if frame_id % retrain_interval == 0:
            print(np.array(predict_X, dtype=object).shape)
            retrain_regressor(predict_X, predict_Y)
            predict_X.clear()
            predict_Y.clear()

            small_ap = compute_ap(small_all_xywh, gt_data)
            small_all_ap.append(small_ap)
            small_all_xywh.clear()
            convert_to_coco(hard_example, 'example_data', max_num=1000)
            print(f"add {len(hard_example)} hard example to dataset pool")
            hard_example.clear()
            if frame_id == retrain_interval:
                print("The first time retrain")
                start_retrain(subnet_idx, retrain_epoch, "checkpoints/SuperNet_epoch_28_mAP_34_90.pth")
            else:
                start_retrain(subnet_idx, retrain_epoch,
                              "checkpoints/SuperNet_epoch_" + str(retrain_epoch - 1) + ".pth")
            print(f"already retrain times:{frame_id / retrain_interval}")
            model = load_efficientdet("checkpoints/SuperNet_epoch_" + str(retrain_epoch - 1) + ".pth").to(device)
            print("model update successfully")
        image_file = os.path.join(frame_path, frames_list[frame_id - 1])
        frame = cv2.imread(image_file)
        RoI_xywh, complexity_score, features = getRoI(frame, back_sub, feature_num)
        small_slot_result = []
        large_slot_result = []
        slot_time = []
        hard_example_flag = complexity_score > complexity_threshold or random.random() < normal_example_possibility
        hard_example_xywh = []
        for RoIs in RoI_xywh:
            RoI_img = RoIs["RoI"]
            x, y, w, h = RoIs["xywh"]
            # 小模型
            start = time.time()
            results, score, _, _ = eff_infer(RoI_img, model, subnet_idx, coco_class_id)
            end = time.time()
            slot_time.append(end - start)
            for i in range(len(results)):
                x1, y1, w1, h1 = results[i].astype(int)  # 获取框的 [x, y, w, h] 坐标
                w1, h1 = w1 - x1, h1 - y1
                small_slot_result.append([x + x1, y + y1, w1, h1])
                small_all_xywh[frame_id].append([x + x1, y + y1, w1, h1, score[i]])

            # 进行采样，并且用大模型进行标签获取，加入正常样例随机采样

            slot_xywh = []
            results = large_model(RoI_img, imgsz=160, verbose=False)
            for result in results:
                boxes = result.boxes  # 获取跟踪结果中的框信息
                for box in boxes:
                    class_idx = int(box.cls.item())
                    if class_idx != yolo_class_id:  # 剔除非目标类型的结果
                        continue
                    x1, y1, w1, h1 = box.xywh[0]  # 获取框的 [x, y, w, h] 坐标
                    x1, y1, w1, h1 = int(x1 - w1 / 2), int(y1 - h1 / 2), int(w1), int(h1)  # 转化为左上角坐标
                    hard_example_xywh.append([x1 + x, y1 + y, w1, h1])
                    slot_xywh.append({"bbox": [x1, y1, w1, h1], "category_id": coco_class_id})
                    gt_data[frame_id].append([x1 + x, y1 + y, w1, h1])
            if hard_example_flag and slot_xywh != []:
                hard_example.append({"image": RoI_img, "annotations": slot_xywh})

            if get_large_model_result:
                results = large_model(RoI_img, imgsz=160, verbose=False)
                for result in results:
                    boxes = result.boxes  # 获取跟踪结果中的框信息
                    for box in boxes:
                        class_idx = int(box.cls.item())
                        if class_idx != yolo_class_id:  # 剔除非目标类型的结果
                            continue
                        x1, y1, w1, h1 = box.xywh[0]  # 获取框的 [x, y, w, h] 坐标
                        x1, y1, w1, h1 = int(x1 - w1 / 2), int(y1 - h1 / 2), int(w1), int(h1)  # 转化为左上角坐标
                        large_slot_result.append([x1 + x, y1 + y, w1, h1])
                        large_all_xywh[frame_id].append([x1 + x, y1 + y, w1, h1, float(box.conf)])
        f1_score = evaluate_detection(small_slot_result, gt_data[frame_id])
        all_time.append(sum(slot_time))
        small_f1_score.append(f1_score)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilize.append(util.gpu)
        # 收集预测器重训练数据
        if hard_example_flag:
            hard_example_f1 = evaluate_detection(hard_example_xywh, gt_data[frame_id])
            predict_X.append(features)
            predict_Y.append(hard_example_f1 - f1_score)
        if get_large_model_result:
            f1_score = evaluate_detection(large_slot_result, gt_data[frame_id])
            large_f1_score.append(f1_score)

    if get_large_model_result:
        large_ap = compute_ap(large_all_xywh, gt_data)
        print(f'large ap{large_ap}')
    small_ap = compute_ap(small_all_xywh, gt_data)
    small_all_ap.append(small_ap)
    print(f'small ap: {small_all_ap}')
    results_ap.append(small_all_ap)
    if get_large_model_result:
        write_multi_list_to_txt([list(gt_data.keys()), small_f1_score, large_f1_score], ["id", "small", "large"])
        # write_multi_list_to_txt([list(gt_data.keys()), small_f1_score, large_f1_score,[row[0] for row in predict_X],[row[1] for row in predict_X],[row[2] for row in predict_X],[row[3] for row in predict_X],[row[4] for row in predict_X]], ["id", "small", "large", "area", "circularity","convexity","clusters","max_area_ratio"])
    else:
        write_multi_list_to_txt([list(gt_data.keys()), small_f1_score, all_time, gpu_utilize],
                                ["id", "small", "time", "gpu"])
