import cv2
from tool.cal_complexity import get_complexity


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


def getRoI(frame, back_sub, f_num):
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
    complexity_score, features = get_complexity(contours, param_num=f_num)
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
    return RoI_xywh, complexity_score, features
