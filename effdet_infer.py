import torch
from effdet import create_model, get_efficientdet_config
import torchvision.transforms as T
import cv2

# ------------------- 配置参数 -------------------
MODEL_NAME = 'resdet50'  # 可选: tf_efficientdet_d0 ~ d7
SCORE_THRESHOLD = 0.1  # 置信度阈值
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------- 加载模型 -------------------
def load_efficientdet(cheackpoint_pth=None,model_name=MODEL_NAME):
    config = get_efficientdet_config(model_name)
    model = create_model(
        model_name,
        bench_task='predict',
        pretrained=True,
        pretrained_backbone=True
    )
    if cheackpoint_pth is not None:
        model.model.load_state_dict(torch.load(cheackpoint_pth))
    model.eval()
    return model


# model = load_efficientdet().to(device)


def preprocess_image(frame, img_size=640):
    """
    使用 OpenCV 实现图像预处理
    输入: frame (OpenCV 图像, BGR 格式), img_size (目标尺寸)
    输出: 预处理后的张量 [1, 3, H, W], 原始图像尺寸 (width, height)
    """
    # 1. 记录原始图像尺寸
    orig_size = (frame.shape[1], frame.shape[0])  # (width, height)

    # 2. 调整图像尺寸
    resized_frame = cv2.resize(frame, (img_size, img_size))  # 缩放为 img_size x img_size

    # 3. 转换通道顺序: BGR -> RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # 4. 转换为 PyTorch 张量并归一化
    transform = T.Compose([
        T.ToTensor(),  # 将 numpy 数组转换为张量，并归一化到 [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    tensor_frame = transform(rgb_frame)  # [3, H, W]

    # 5. 添加 batch 维度
    tensor_frame = tensor_frame.unsqueeze(0)  # [1, 3, H, W]

    return tensor_frame, orig_size


# ------------------- 后处理 -------------------
def postprocess_detections(detections_tensor, orig_size, class_id, img_size=640, threshold=SCORE_THRESHOLD):
    """
    输入张量形状: [batch_size, num_detections, 6]
    6个值含义: [x1, y1, x2, y2, score, class_label]
    """
    # 提取检测结果（假设 batch_size=1）
    detections = detections_tensor[0].cpu().numpy()  # [num_detections, 6]

    # 过滤低置信度检测,label==3表明目标是car
    mask = (detections[:, 4] > threshold) & (detections[:, 5] == class_id)
    detections = detections[mask]

    # 解析 boxes、scores、labels
    boxes = detections[:, :4]  # [x1, y1, x2, y2]
    scores = detections[:, 4]  # 置信度
    labels = detections[:, 5]  # 类别标签

    # 坐标缩放
    w_ratio = orig_size[0] / img_size
    h_ratio = orig_size[1] / img_size
    boxes[:, [0, 2]] *= w_ratio
    boxes[:, [1, 3]] *= h_ratio

    return boxes, scores, labels.astype(int)


# ------------------- 可视化结果 -------------------
def visualize_detections(image_path, boxes, scores, labels):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        # 绘制边界框
        cv2.rectangle(image_path, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 显示标签和置信度
        text = f'Class {label}: {score:.2f}'
        cv2.putText(image_path, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite('00119_res.jpg', image_path)


def eff_infer(frame, model,class_id, img_size=640):
    """
    输入: frame (OpenCV 图像, BGR 格式), img_size (目标尺寸)
    输出: boxes [x1, y1, x2, y2]
    """
    # 1. 预处理图像
    img_tensor, orig_size = preprocess_image(frame, img_size=img_size)
    img_tensor = img_tensor.to(device)

    # 2. 推理（输出为张量）
    with torch.no_grad():
        detections = model(img_tensor)  # 形状 [1, num_detections, 6]
    # 3. 后处理
    boxes, scores, labels = postprocess_detections(detections, orig_size,class_id, img_size=img_size)
    return boxes

# if __name__ == '__main__':
#     # 使用示例
#     image_path = '00119.jpg'  # 替换为你的图片路径
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     boxes, scores, labels = eff_infer(img, model)
#     visualize_detections(img, boxes, scores, labels)
