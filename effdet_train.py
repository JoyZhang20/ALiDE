import os
import argparse
import time
import yaml
import logging
from collections import OrderedDict
import torch
import torchvision.utils
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
from effdet import create_model, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models.layers import set_layer_config
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
torch.backends.cudnn.benchmark = True


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
"""=============主要关心这些参数就可以============="""
parser.add_argument('--root', metavar='DIR',default='C:/Users/Administrator/Desktop/4_CAVE_Retrain/example_data',
                    help='COCO数据集路径')
parser.add_argument('--dataset', default='coco2014', type=str, metavar='DATASET',
                    help='数据集类型')
parser.add_argument('--model', default='resdet50', type=str, metavar='MODEL',
                    help='可选的模型')
parser.add_argument('--load-checkpoint', action='store_true', default=True,
                    help='是否加载之前训练的模型，注意在代码中需要将-替换_，即args.load_checkpoint')
parser.add_argument('--checkpoint_path', default='checkpoints/resdet50_epoch_2_mAP_0_35506150697175537.pth',
                    metavar='DIR', help='之前模型的路径')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='并行线程的数量')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='是否加载effdet提供的预训练模型')
parser.add_argument('--no-pretrained-backbone', action='store_true', default=False,
                    help='是否不加载effdet提供的预训练主干')
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01),1.1e-5')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
"""=====================END====================="""


add_bool_arg(parser, 'redundant-bias', default=None, help='override model config for redundant bias')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--val-skip', type=int, default=0, metavar='N',
                    help='Skip every N validation samples.')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
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
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
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
parser.add_argument('--warmup-lr', type=float, default=1.3e-5, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=1, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.985, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
# parser.add_argument('--aa', type=str, default=None, metavar='NAME',
#                     help='Use AutoAugment policy. "v0" or "original". (default: None)'),
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
# parser.add_argument('--model-ema', action='store_true', default=False,
#                     help='Enable tracking moving average of model weights')
# parser.add_argument('--model-ema-decay', type=float, default=0.9998,
#                     help='decay factor for model weights moving average (default: 0.9998)')

# Misc
# parser.add_argument('--sync-bn', action='store_true',default=True,
#                     help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
# parser.add_argument('--dist-bn', type=str, default='',
#                     help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
# parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
#                     help='how many batches to wait before writing recovery checkpoint')

parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
# parser.add_argument('--amp', action='store_true', default=True,
#                     help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
# parser.add_argument('--apex-amp', action='store_true', default=False,
#                     help='Use NVIDIA Apex AMP mixed precision')
# parser.add_argument('--native-amp', action='store_true', default=False,
#                     help='Use Native Torch AMP mixed precision')
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
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--device_id', type=str, default='0')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    # if args_config.config:
    #     with open(args_config.config, 'r') as f:
    #         cfg = yaml.safe_load(f)
    #         parser.set_defaults(**cfg)

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

    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)
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
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_train_fn,
        collate_fn=collate_fn,
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
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, pred_yxyx=False)

    return loader_train, loader_eval, evaluator


def train_epoch(
        epoch, model, loader, optimizer, args,
        lr_scheduler=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    # import pdb;pdb.set_trace()
    clip_params = get_clip_parameters(model, exclude_head='agc' in args.clip_mode)
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        output = model(input, target)
        loss = output['loss']
        # import pdb;pdb.set_trace()
        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            dispatch_clip_grad(clip_params, value=args.clip_grad, mode=args.clip_mode)
        optimizer.step()

        torch.cuda.synchronize()
        # if model_ema is not None:
        #     model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            # if args.distributed:
            #     reduced_loss = reduce_tensor(loss.data, args.world_size)
            #     losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                logging.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                # if args.save_images and output_dir:
                #     torchvision.utils.save_image(
                #         input,
                #         os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                #         padding=0,
                #         normalize=True)

        # if saver is not None and args.recovery_interval and (
        #         last_batch or (batch_idx + 1) % args.recovery_interval == 0):
        #     saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, args, evaluator=None, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            # subnet = model.model.generate_random_subnet()
            # subnet=[[2, 99, 99, 1, 99, 1, 99], [2, 99, 99, 1, 99, 0], [1, 99, 0]]
            # print("validate subnet={}".format(subnet))
            output = model(input, target)
            loss = output['loss']

            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)
            # import pdb;pdb.set_trace()
            # if args.distributed:
            #     reduced_loss = reduce_tensor(loss.data, args.world_size)
            # else:
            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))
    # import pdb;pdb.set_trace()
    metrics = OrderedDict([('loss', losses_m.avg)])
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()

    return metrics


def start_retrain(epochs=None,model_path=None):
    setup_default_logging()
    args, args_text = _parse_args()

    args.pretrained_backbone = not args.no_pretrained_backbone
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if model_path is not None:
        args.checkpoint_path = model_path
    if epochs is not None:
        args.epochs = epochs
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0

    random_seed(42,0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
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
    model_config = model.config
    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' % (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)


    optimizer = create_optimizer(args, model)


    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0


    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))
    loader_train, loader_eval, evaluator = create_datasets_and_loaders(args, model_config)
    if model_config.num_classes < loader_train.dataset.parser.max_label:
        logging.error(
            f'Model {model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
        exit(1)
    if model_config.num_classes > loader_train.dataset.parser.max_label:
        logging.warning(
            f'Model {model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')

    eval_metric = args.eval_metric

    '''加载之前训练过的模型'''
    if args.load_checkpoint:
        model.model.load_state_dict(torch.load(args.checkpoint_path))

    for epoch in range(start_epoch, num_epochs):
        train_epoch(
            epoch, model, loader_train, optimizer, args,
            lr_scheduler=lr_scheduler)

        eval_metrics = validate(model, loader_eval, args, evaluator)
        name = "checkpoints/resdet50_epoch_" + str(epoch)+"_mAP_"+f"{eval_metrics['map']*100:.2f}".replace('.','_') + ".pth"  # + "loss" + str(int(train_metrics['loss'] * 1000)) + ".pth"
        torch.save(model.model.state_dict(), name)
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])



