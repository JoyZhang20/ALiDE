import pickle
import matplotlib.patches as patches
import torch
from matplotlib import pyplot as plt
import argparse
import yaml
from effdet import create_model
import torchvision.transforms as T
import cv2

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='data/config/args_infer.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def preprocess_image(frame, img_size=224):
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
def postprocess_detections(detections_tensor, orig_size, class_id, img_size=640, threshold=0.1):
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
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_path)
    for box, score, cls in zip(boxes, scores, labels):
        if score > 0.15:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            text = f'Class {cls}: {score:.2f}'
            ax.text(x1, y1 - 5, text, fontsize=10,
                    color='lime', bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def load_efficientdet(cheackpoint_pth=None):
    args, args_text = _parse_args()
    model = create_model(
        args.model,
        bench_task='train',

    )
    if cheackpoint_pth is not None:
        model.model.load_state_dict(torch.load(cheackpoint_pth, weights_only=True))
    else:
        model.model.load_state_dict(torch.load('checkpoints/SuperNet_epoch_28_mAP_34_90.pth', weights_only=True))
    model.cuda()
    model.eval()
    return model


def eff_infer(frame, model, subnet_idx, class_id=3, img_size=640):
    args, args_text = _parse_args()
    # model = create_model(
    #     args.model,
    #     bench_task='train',
    #
    # )
    # model.model.load_state_dict(torch.load("checkpoint/SuperNet_epoch_28_mAP_34_90.pth", weights_only=True))
    # model.cuda()
    # model.eval()

    subnets = [  # 这里就是预定义的网络结构，0是正常的一层网络，1,99表示两层合并为一层，2,99,99表示三层合并为一层。这里从上到下网络层数越来越多，依次为6-16层
        [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],  # 三个[]好像是表示不同的结构，特征提取、特征融合之类的
        [[0, 2, 99, 99, 2, 99, 99], [1, 99, 0, 2, 99, 99], [2, 99, 99]],
        [[0, 2, 99, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [0, 1, 99]],
        [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 0, 0]],
        [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 0, 0, 1, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    ]
    subnet = subnets[subnet_idx]
    img_tensor, orig_size = preprocess_image(frame, img_size=img_size)
    img_tensor = img_tensor.to(device)
    with open('my_dict.pkl', 'rb') as f:
        img_target = pickle.load(f)
    with torch.no_grad():
        output = model(img_tensor, img_target, subnet, args)  # 形状 [1, num_detections, 6]
    detections = output['detections']
    boxes, scores, labels = postprocess_detections(detections, orig_size, class_id,img_size=img_size, )
    return boxes, scores, output['detections'], labels


# if __name__ == '__main__':
#     image_path = 'dataset/MVI_39031/00550.jpg'  # 替换为你的图片路径
#     img = cv2.imread(image_path)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     model = load_efficientdet("checkpoints/SuperNet_epoch_28_mAP_34_90.pth")
#     boxes, scores, labels = infer(img, model, 0)
#
#     visualize_detections(img, boxes, scores, labels)
