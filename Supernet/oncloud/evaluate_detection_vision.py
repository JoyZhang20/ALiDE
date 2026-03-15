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
from oncloud.others import mytools
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.models.layers import set_layer_config
from timm.utils import *
from timm.optim import create_optimizer

torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

"""==========================核心参数=========================="""
# parser.add_argument('--root', metavar='DIR', default='/home/junjie/1_dataset/COCO2017',
#                     help='path to dataset')
# parser.add_argument('--dataset', default='coco2017', type=str, metavar='DATASET',
#                     help='Name of dataset to train (default: "coco"')
parser.add_argument('--root', metavar='DIR', default='tool/output_coco',
                    help='path to dataset')
parser.add_argument('--dataset', default='coco2014', type=str, metavar='DATASET',
                    help='Name of dataset to train (default: "coco"')
parser.add_argument('--dataratio', type=float, default=0.02, metavar='ratio of dataset used',
                    help='训练和评估的数据集比例')
parser.add_argument('--model', default='resdet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tf_efficientdet_d1"')
parser.add_argument('--modelpath', metavar='DIR', help='path to dataset',
                    default='checkpoint/SuperNet_epoch_28_mAP_34_90.pth')
parser.add_argument('-b', '--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 32)')
"""==========================核心参数=========================="""

add_bool_arg(parser, 'redundant-bias', default=None, help='override model config for redundant bias')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--val-skip', type=int, default=5, metavar='N',
                    help='Skip every N validation samples.')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--no-pretrained-backbone', action='store_true', default=False,
                    help='Do not start with pretrained backbone weights, fully random.')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Optimizer parameters
parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=5e-4, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=5, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.65, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# loss
parser.add_argument('--smoothing', type=float, default=None, help='override model config label smoothing')
add_bool_arg(parser, 'jit-loss', default=None, help='override model config for torchscript jit loss fn')
add_bool_arg(parser, 'legacy-focal', default=None, help='override model config to use legacy focal loss')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--sync-bn', action='store_true', default=True,
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=True,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
add_bool_arg(parser, 'bench-labeler', default=False,
             help='label targets in model bench, increases GPU load at expense of loader processes')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='map', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "map"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--stage', type=int, default=0)


def get_model_size(Detmodel, subnet):
    multimodel = Detmodel.model.backbone
    model = []
    model.append(multimodel.conv1)
    model.append(multimodel.bn1)
    model.append(multimodel.act1)
    model.append(multimodel.maxpool)
    for blockidx in range(len(subnet[0])):
        if subnet[0][blockidx] != 99:
            model.append(multimodel.layer2[blockidx][subnet[0][blockidx]])
    for blockidx in range(len(subnet[1])):
        if subnet[1][blockidx] != 99:
            model.append(multimodel.layer3[blockidx][subnet[1][blockidx]])
    for blockidx in range(len(subnet[2])):
        if subnet[2][blockidx] != 99:
            model.append(multimodel.layer4[blockidx][subnet[2][blockidx]])
    multimodel = Detmodel.model
    model.append(multimodel.fpn)
    model.append(multimodel.class_net)
    model.append(multimodel.box_net)
    temp = torch.nn.Sequential(*model)
    tmp_model_file_path = 'tmp.model'
    torch.save(temp, tmp_model_file_path)
    model_size = os.path.getsize(tmp_model_file_path)
    os.remove(tmp_model_file_path)
    model_size /= 1024 ** 2
    del temp
    return model_size


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


def get_clip_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params
        return [p for n, p in model.named_parameters() if 'predict' not in n]
    else:
        return model.parameters()


def create_datasets_and_loaders(
        args,
        model_config,
        transform_train_fn=None,
        transform_eval_fn=None,
        collate_fn=None,
        # sample_num=None
):
    """ Setup datasets, transforms, loaders, evaluator.

    Args:
        args: Command line args / config for training
        model_config: Model specific configuration dict / struct
        transform_train_fn: Override default image + annotation transforms (see note in loaders.py)
        transform_eval_fn: Override default image + annotation transforms (see note in loaders.py)
        collate_fn: Override default fast collate function

    Returns:
        Train loader, validation loader, evaluator
    """
    input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset(args.dataset, args.root)

    # if sample_num is not None:
    #     idxs = np.random.choice(5000, sample_num, replace=False).tolist()
    #     dataset_eval = Subset(dataset_eval, idxs)
    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)

    if args.val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, args.val_skip)
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
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
        data_ratio=args.dataratio
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, distributed=args.distributed, pred_yxyx=False)

    return loader_eval, evaluator


def validate(model, loader, args, subnet, evaluator=None, log_suffix='', lats=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    latency = lats[0] + lats[-1]
    for layeridx in range(len(subnet)):
        for blockidx in range(len(subnet[layeridx])):
            if subnet[layeridx][blockidx] != 99:
                latency += lats[layeridx + 1][blockidx][subnet[layeridx][blockidx]]
    cx = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(loader)):
            img_list=['000005','000010','000015','000020','000025']
            if cx < len(img_list):
                output = model(input, target, subnet=subnet, args=args)
                detections = output['detections']  # [B, 100, 6]
                # frame = cv2.imread('F:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/COCO2017/val2017/' + img_list[cx] + '.jpg')
                frame = cv2.imread('tool/output_coco/val2014/' + img_list[cx] + '.jpg')
                for i in range(input.size(0)):  # 遍历 batch 内每张图
                    for det in detections[i]:
                        x1, y1, x2, y2, score, cls = det.tolist()
                        if score >= 0.2:
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    save_path = os.path.join('output/img/', f"{img_list[cx]}_subnet000.jpg")
                    cv2.imwrite(save_path, frame)
                cx += 1

            output = model(input, target, subnet=subnet, args=args)
            # output['detections']:[batchsize, 100, 6] 6->[x_min, y_min, x_max, y_max, score, class]
            loss = output['loss']
            # import pdb;pdb.set_trace()
            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)
            # import pdb;pdb.set_trace()
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            # if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
            #     log_name = 'Test' + log_suffix
            #     logging.info(
            #         '{0}: [{1:>4d}/{2}]  '
            #         'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            #         'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
            #             log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))
    # import pdb;pdb.set_trace()
    metrics = OrderedDict([('loss', losses_m.avg)])
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()
    # metrics['map']=metrics['map']*(1/args.dataratio)
    # print(metrics['map'])
    return metrics['map'], latency


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    # if platform.node() in ['Joy-FZU']:
    #     args.root = 'F:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/COCO2017'
    args.pretrained_backbone = not args.no_pretrained_backbone
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Testing with a single process on 1 GPU.')

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                            "Install NVIDA apex or upgrade to PyTorch 1.6.")

    if args.native_amp:
        if has_native_amp:
            use_amp = 'native'
        else:
            logging.warning("Native AMP not available, using float32. Upgrade to PyTorch 1.6.")
    elif args.apex_amp:
        if has_apex:
            use_amp = 'apex'
        else:
            logging.warning("APEX AMP not available, using float32. Install NVIDA apex")

    random_seed(args.seed, args.rank)

    with set_layer_config(scriptable=args.torchscript):
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
    model_config = model.config  # grab before we obscure with DP/DDP wrappers
    model.model.load_state_dict(torch.load(args.modelpath, map_location=torch.device('cpu')), strict=True)
    # print(model)
    # lats = mytools.get_lats(model)
    # if args.local_rank == 0:
    #     logging.info('Model %s created, param count: %d' % (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.distributed and args.sync_bn:
        if has_apex and use_amp == 'apex':
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            logging.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model. Use `--dist-bn reduce` instead of `--sync-bn`'
        model = torch.jit.script(model)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        # if args.local_rank == 0:
        #     logging.info('Using native Torch AMP. Training in mixed precision.')
    elif use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            logging.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            logging.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            unwrap_bench(model), args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        if args.resume:
            load_checkpoint(unwrap_bench(model_ema), args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp == 'apex':
            if args.local_rank == 0:
                logging.info("Using apex DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.device])
        # NOTE: EMA model does not need to be wrapped by DDP...
        if model_ema is not None and not args.resume:
            # ...but it is a good idea to sync EMA copy of weights
            # NOTE: ModelEma init could be moved after DDP wrapper if using PyTorch DDP, not Apex.
            model_ema.set(model)
    with suppress_stdout():
        loader_eval, evaluator = create_datasets_and_loaders(args, model_config)
    # import pdb;pdb.set_trace()

    # if model_config.num_classes < loader_train.dataset.parser.max_label:
    #     logging.error(
    #         f'Model {model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
    #     exit(1)
    # if model_config.num_classes > loader_train.dataset.parser.max_label:
    #     logging.warning(
    #         f'Model {model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')
    latencys = [0.0013505160206496113, [[0.002749936749236752, 0.0027569592601121074, 0.0027531325215041035],
                                        [0.0024147708006579467, 0.0024153126610649955, 0.0020566660948474],
                                        [0.0024782840651695177, 0.0020377660038495304, 0.002047897589327109],
                                        [0.002727144896382033, 0.0027433910755196, 0.0027275567102913903],
                                        [0.0014948363255972814, 0.0014833994586058337, 0.0014968544545799795],
                                        [0.0014858390345717921, 0.0014849455669672803], [0.0014843988900232797]],
                [[0.0019092318987605549, 0.0019139496967045947, 0.0019048416253292198],
                 [0.001112234712851168, 0.001113807312165848, 0.0011128006559429746],
                 [0.00111214801518604, 0.0011126152192703401, 0.0011093207079954821],
                 [0.0011120492761785333, 0.0011246782360654888, 0.0011179784331658874],
                 [0.0011128126972853535, 0.0011120805836687185], [0.001109693989609227]],
                [[0.001725912094116211, 0.0017217361565792198, 0.0017308177370013614],
                 [0.001049742554173325, 0.0010378144004128196], [0.0010321405198838976]], 0.011298762427435981]
    eval_metric = args.eval_metric
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
    with open('output/config/args_infer.yaml', 'w') as f:
        f.write(args_text)
    subnets = mytools.get_val_subnets(model.model)
    # with open('visualization/detection/subnets.yaml', 'w') as f:
    #     yaml.dump(subnets, f)
    # lats, maps, sizes = [], [], []
    latency_list = []
    acc_list = []
    # print("len of subnets=",len(subnets))
    # subnets=subnets[:3]
    # subnets = mytools.get_val_subnets(model.model)
    subnets = [
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 2, 99, 99, 0], [2, 99, 99]],
        # [[0, 2, 99, 99, 2, 99, 99], [1, 99, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 2, 99, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [2, 99, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 2, 99, 99], [0, 1, 99]],
        # [[0, 0, 1, 99, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 1, 99]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 1, 99], [0, 0, 0]],
        # [[0, 0, 0, 0, 2, 99, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        # [[0, 0, 0, 0, 0, 1, 99], [0, 0, 0, 0, 0, 0], [0, 0, 0]],
        # [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    ]
    for subnet in subnets:
        # print("subnet=", subnet)
        s_time = time.time()
        with suppress_stdout():
            mAP, lat = validate(model, loader_eval, args, subnet, evaluator, lats=latencys)
        mAP = round(mAP, 5)
        time_used = round(time.time() - s_time, 3)
        latency_list.append(time_used)
        acc_list.append(mAP)
        print(f'===========mAP: {mAP}, latency: {time_used}, subnet_architecture: {subnet}============')
    print('acc\tlatency')
    for i in range(len(latency_list)):
        print(acc_list[i], '\t', latency_list[i])
        # with open('visualization/detection/maps.yaml', 'w') as f:
        #     yaml.dump(maps, f)
        # print("joy time used={}".format(round(time.time() - s_time, 5)))
    # print(sizes, ",", maps)
    # subnet = [[2, 99, 99, 2, 99, 99, 0], [2, 99, 99, 2, 99, 99], [2, 99, 99]]
    # eval_metrics, latency = validate(model, loader_eval, args, subnet, evaluator)
    # subnet = model.model.generate_main_subnet()
    # eval_metrics, latency = validate(model, loader_eval, args, subnet, evaluator)


if __name__ == '__main__':
    main()
