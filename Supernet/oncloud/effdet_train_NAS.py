import json
import os
import argparse
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tool.JoyTool import suppress_stdout
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from AdaptiveNet import oncloud
from effdet import create_model, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.utils import *

torch.backends.cudnn.benchmark = True

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='data/config/args_train.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_clip_parameters(model, exclude_head=False):
    if exclude_head:
        return [p for n, p in model.named_parameters() if 'predict' not in n]
    else:
        return model.parameters()


def create_datasets_and_loaders(
        args,
        model_config,
        transform_train_fn=None,
        transform_eval_fn=None,
        collate_fn=None,
):
    input_config = resolve_input_config(args, model_config=model_config)
    with suppress_stdout():
        dataset_train, dataset_eval = create_dataset(args.dataset, args.root)

    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)
    # import pdb;pdb.set_trace()
    if args.val_skip > 1:
        dataset_train = SkipSubset(dataset_train, args.val_skip)
    print("Ratio of used dataset = {} ".format(args.dataratio))
    loader_train = create_loader(
        dataset_train,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        interpolation=args.train_interpolation or input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_train_fn,
        collate_fn=collate_fn,
        data_ratio=args.dataratio
    )
    if args.val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, args.val_skip)
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
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
        data_ratio=args.dataratio
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, distributed=args.distributed, pred_yxyx=False)

    return loader_train, loader_eval, evaluator


def train_epoch(epoch, model, loader, optimizer, args):
    model.train()
    oncloud.freeze_detection_bn(model)
    pbar = tqdm(
        loader,
        desc=f"Train Epoch [{epoch}/{args.epochs}]",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}] {postfix}"
    )

    for batch_idx, (input, target) in enumerate(pbar):
        subnet = model.model.generate_random_subnet()
        while oncloud.no_new_detection_subnet(subnet):
            subnet = model.model.generate_random_subnet()
        output = model(input, target, subnet=subnet, args=args)
        loss = output['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])


def validate(model, loader, args):
    model.eval()
    results = []
    with torch.no_grad():
        pbar = tqdm(loader,desc=f"Evaluation")
        for batch_idx, (input, target) in enumerate(pbar):
            subnet = model.model.generate_random_subnet()
            output = model(input, target, subnet=subnet, args=args)
            detections = output['detections']
            for i in range(input.size(0)):
                image_id = int(target['img_idx'][i].item()) + 1
                for det in detections[i]:
                    x1, y1, x2, y2, score, cls = det.tolist()
                    # if score < 0.2:
                    # if int(cls) not in [3,7,8,6,14,10]:
                    #     print("not in")
                    if score < 0.01:
                        continue
                    x, y = x1, y1
                    w, h = x2 - x1, y2 - y1
                    results.append({
                        'image_id': image_id,
                        'category_id': 2,  # COCO 格式中是 int
                        'bbox': [x, y, w, h],
                        'score': float(score)
                    })
    with open('output/coco_res/train_validate.json', 'w') as f:
        json.dump(results, f)


def main():
    args, args_text = _parse_args()
    # args.dataset = 'coco2017'
    # args.root = '/home/junjie/1_dataset/COCO2017'
    args.dataset = 'coco2014'
    args.root = 'tool/output_coco'
    args.modelpath = 'checkpoint/SuperNet_epoch_28_mAP_34_90.pth'
    args.dataratio = 1  # 允许只训练部分样本，最后mAP的时候乘以对应的倍数就可以
    args.batch_size = 8
    random_seed(args.seed, args.rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
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
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)  # 假设训练30个epoch,joy

    loader_train, loader_eval, evaluator = create_datasets_and_loaders(args, model_config)

    model, _ = oncloud.load_to_detectionmodel(args.headpath, model, part="head", freeze_head=True)
    model.model.load_state_dict(torch.load(args.recoverypath, weights_only=True))
    print(args.recoverypath + " loaded.")

    for epoch in range(args.epochs):
        train_epoch(
            epoch, model, loader_train, optimizer, args)
        # scheduler.step()

        # with suppress_stdout():  # 关闭控制台一大堆输出
        validate(model, loader_eval, args)
        with suppress_stdout():
            coco_gt = COCO(args.root + '/annotations/instances_val2014.json')
            coco_dt = coco_gt.loadRes('output/coco_res/train_validate.json')
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        mAp = coco_eval.stats[-1] * (1 / args.dataratio)
        # mAp = coco_eval.stats[0]
        print("======mAp={}======".format(round(mAp, 5)))
        name = "checkpoint/SuperNet_epoch_" + str(epoch) + "_mAP_" + f"{mAp * 100:.2f}".replace(
            '.', '_') + ".pth"
        torch.save(model.model.state_dict(), name)
        # send_post("epoch_" + str(epoch) + "_mAP_" + f"{eval_metrics['map'] * 100:.2f}".replace(
        #     '.', '_') )


if __name__ == '__main__':
    main()
