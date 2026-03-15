import json
import os

import cv2
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import platform
from tool.JoyTool import suppress_stdout
import argparse
import time
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
# from oncloud.others import mytools
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.models.layers import set_layer_config
from timm.utils import *
# from timm.optim import create_optimizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='output/config/args.yaml', type=str, metavar='FILE',
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


def validate(subnet_index, model, loader, args, subnet):
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            output = model(input, target, subnet=subnet, args=args)
            detections = output['detections']
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
    with open('output/coco_res/predictions' + str(subnet_index) + '.json', 'w') as f:
        json.dump(results, f)


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    # args.dataset = 'coco2017'
    # args.root = '/home/junjie/1_dataset/COCO2017'
    args.dataset = 'coco2014'
    args.root = 'tool/output_coco'
    args.modelpath = 'checkpoint/SuperNet_epoch_28_mAP_34_90.pth'
    args.dataratio = 0.1
    args.batch_size = 1
    # args.pretrained=True
    # print(f"args.pretrained={}")
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

    subnets = [#这里就是预定义的网络结构了，
        [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],
        [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],
        [[0, 2, 99, 99, 2, 99, 99], [1, 99, 0, 2, 99, 99], [2, 99, 99]],
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
    is_evaluate = True
    latency_list = []
    acc_list = []
    if is_evaluate:
        for i in range(len(subnets)):
            s_time = time.time()
            with suppress_stdout():
                validate(i, model, loader_eval, args, subnets[i])
            latency_list.append(time.time() - s_time)
    print("============Evaluate results...==============")
    for i in range(len(subnets)):
        with suppress_stdout():
            coco_gt = COCO(args.root + '/annotations/instances_val2014.json')
            coco_dt = coco_gt.loadRes('output/coco_res/predictions' + str(i) + '.json')
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        acc_list.append(coco_eval.stats[-1])
    print('acc\tlat')
    for i in range(len(acc_list)):
        print(acc_list[i], '\t', latency_list[i])


if __name__ == '__main__':
    main()
