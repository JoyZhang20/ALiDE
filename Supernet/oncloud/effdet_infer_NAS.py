from tqdm import tqdm
import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict
from tool.JoyTool import suppress_stdout
import argparse
import time
import yaml
import torch
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='data/config/args_infer.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def create_datasets_and_loaders(
        args,
        model_config,
        transform_train_fn=None,
        transform_eval_fn=None,
        collate_fn=None,
        # sample_num=None
):
    input_config = resolve_input_config(args, model_config=model_config)
    dataset_train, dataset_eval = create_dataset(args.dataset, args.root)
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)
    # if args.val_skip > 1:
    #     dataset_eval = SkipSubset(dataset_eval, args.val_skip)
    print("Ratio of used dataset ={}".format(args.dataratio))
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
        data_ratio=args.dataratio
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, pred_yxyx=False)

    return loader_eval, evaluator


def visualize_predictions(
        ann_file,
        pred_file,
        img_dir,
        num_images=10,
        score_thresh=0.3,
        save_dir=None
):
    """
    可视化 predictions.json 中的检测结果

    参数：
    - ann_file: COCO 格式标注文件路径（如 instances_val2014.json）
    - pred_file: 模型输出的 predictions.json
    - img_dir: 图像文件所在目录
    - num_images: 可视化的图像数量
    - score_thresh: 显示预测框的置信度阈值
    - save_dir: 若指定目录，则保存图像；否则用 matplotlib 显示
    """

    with suppress_stdout():
        coco = COCO(ann_file)

    with open(pred_file, 'r') as f:
        preds = json.load(f)

    pred_by_image = defaultdict(list)
    for pred in preds:
        pred_by_image[pred['image_id']].append(pred)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    img_ids = list(pred_by_image.keys())
    random.shuffle(img_ids)
    img_ids = img_ids[:num_images]

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)

        for det in pred_by_image[img_id]:
            x, y, w, h = det['bbox']
            score = det['score']
            cat_id = det['category_id']
            name = coco.loadCats(cat_id)[0]['name']

            if score < score_thresh:
                continue

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 3, f'{name}: {score:.2f}', color='lime', fontsize=10, weight='bold')

        ax.axis('off')
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, file_name)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
def compute_cosine_distances(features, reference_feature):
    # 输入：features [N, D]，reference_feature [D]
    # 输出：余弦距离（1 - 余弦相似度）[N]
    features = F.normalize(features, dim=1)  # 归一化样本特征
    reference_feature = F.normalize(reference_feature.unsqueeze(0), dim=1)  # [1, D]
    cosine_sim = torch.mm(features, reference_feature.T).squeeze(1)  # [N]
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def validate(subnet_index, model, loader, args, subnet):
    model.eval()
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_sum = torch.zeros(600).to(device)
    sample_count=0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            output = model(input, target, subnet=subnet, args=args)
            detections = output['detections']

            # features = detections.view(detections.size(0), -1)
            # coco_mean_feature = torch.load("coco_mean_feature.pt").to(device)
            # distances = compute_cosine_distances(features, coco_mean_feature)  # [B]
            # topk = int(0.1 * len(distances))  # 前10%
            # topk_indices = torch.topk(distances, topk).indices
            # print()
            # 获取这些图像及标签（可视化、保存等）
            # top_images = images[topk_indices]
            # top_labels = [cifar_dataset.classes[labels[i]] for i in topk_indices]

            features = detections.view(detections.size(0), -1)  # [B, 2048]
            feature_sum += features.sum(dim=0)
            sample_count += features.size(0)

            for i in range(input.size(0)):  # 遍历 batch
                image_id = int(target['img_idx'][i].item()) + 1  # 获取x image_id
                for det in detections[i]:
                    x1, y1, x2, y2, score, cls = det.tolist()
                    if score < 0.001:
                        continue
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                    results.append({
                        'image_id': image_id,
                        'category_id': 2,  # COCO 格式中是 int
                        'bbox': [x, y, w, h],
                        'score': float(score)
                    })
    coco_mean_feature = feature_sum / sample_count
    coco_mean_feature = coco_mean_feature / coco_mean_feature.norm()  # L2归一化

    # 6. 保存
    torch.save(coco_mean_feature, "coco_mean_feature.pt")
    print("COCO平均特征向量保存完成！")
    with open('output/coco_res/predictions' + str(subnet_index) + '.json', 'w') as f:
        json.dump(results, f)


def evaluate_result():
    acc_list = []
    subnets = [  # 这里就是预定义的网络结构，0是正常的一层网络，1,99表示两层合并为一层，2,99,99表示三层合并为一层。这里从上到下网络层数越来越多，依次为6-16层
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],  # 第一个网络结构是重复的，预热一下
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],  # 三个[]好像是表示不同的结构，特征提取、特征融合之类的
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 2, 99, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [0, 1, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 0, 0]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        # [[0, 0, 0, 0, 0, 1, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    ]
    print("============Evaluate results...==============")
    for i in range(len(subnets)):
        with suppress_stdout():
            coco_gt = COCO('tool/output_coco/annotations/instances_val2014.json')
            coco_dt = coco_gt.loadRes('output/coco_res/predictions' + str(i) + '.json')
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        acc_list.append(coco_eval.stats[-1])  # 如果在线推理的不是所有数据，这里可以乘以对应的倍数
    print('acc')
    for i in range(len(acc_list)):
        print(round(acc_list[i], 5))


def online_infer():
    args, args_text = _parse_args()
    args.dataset = 'coco2017'
    args.root = '/home/junjie/1_dataset/COCO2017'
    # args.dataset = 'coco2014'
    # args.root = 'tool/output_coco'
    args.modelpath = 'checkpoint/SuperNet_epoch_28_mAP_34_90.pth'
    args.dataratio = 0.1  # 允许只评估部分样本，最后mAP的时候乘以对应的倍数就可以
    args.batch_size = 1
    model = create_model(
        args.model,
        bench_task='train',
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        pretrained_backbone=args.pretrained_backbone,
        redundant_bias=args.redundant_bias,
        label_smoothing=args.smoothing,
        legacy_focal=args.legacy_focal,
        jit_loss=args.jit_loss,
        soft_nms=args.soft_nms,
        bench_labeler=args.bench_labeler,
        checkpoint_path=args.initial_checkpoint,
    )
    model_config = model.config
    model.model.load_state_dict(torch.load(args.modelpath, weights_only=True))
    model.cuda()
    with suppress_stdout():  # 屏蔽装载日志
        loader_eval, evaluator = create_datasets_and_loaders(args, model_config)

    subnets = [  # 这里就是预定义的网络结构，0是正常的一层网络，1,99表示两层合并为一层，2,99,99表示三层合并为一层。这里从上到下网络层数越来越多，依次为6-16层
        [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],  # 第一个网络结构是重复的，预热一下
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],  # 三个[]好像是表示不同的结构，特征提取、特征融合之类的
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 2, 99, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [0, 1, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 0, 0]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        # [[0, 0, 0, 0, 0, 1, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        # [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    ]
    latency_list = []
    for i in range(len(subnets)):
        s_time = time.time()
        validate(i, model, loader_eval, args, subnets[i])
        latency_list.append(time.time() - s_time)
    print('acc')
    for i in range(len(latency_list)):
        print(round(latency_list[i], 5))


if __name__ == '__main__':
    online_infer()
    # evaluate_result()
    # visualize_predictions(
    #     ann_file='tool/output_coco/annotations/instances_val2014.json',
    #     pred_file='output/coco_res/predictions0.json',
    #     img_dir='tool/output_coco/val2014',
    #     num_images=5,
    #     score_thresh=0.2,
    #     save_dir=None  # 或 'vis_results' 保存结果
    # )
