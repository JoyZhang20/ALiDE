from .efficientdet import EfficientDet, HeadNet
from .bench import DetBenchTrain, DetBenchPredict
from .config import get_efficientdet_config
from .helpers import load_pretrained, load_checkpoint


def create_model(
        model_name, bench_task='', num_classes=None, pretrained=False,
        checkpoint_path='', checkpoint_ema=False, **kwargs):
    config = get_efficientdet_config(model_name)
    return create_model_from_config(
        config, bench_task=bench_task, num_classes=num_classes, pretrained=pretrained,
        checkpoint_path=checkpoint_path, checkpoint_ema=checkpoint_ema, **kwargs)


def create_model_from_config(
        config, bench_task='', num_classes=None, pretrained=False,
        checkpoint_path='', checkpoint_ema=False, **kwargs):

    # 获取是否使用预训练骨干网络的参数，默认为True
    pretrained_backbone = kwargs.pop('pretrained_backbone', True)
    # 如果指定了预训练权重或检查点路径，则不加载骨干网络权重
    if pretrained or checkpoint_path:
        pretrained_backbone = False  # 没有必要加载骨干网络权重

    # 配置覆盖，通过kwargs覆盖配置中的某些值
    overrides = (
        'redundant_bias', 'label_smoothing', 'legacy_focal', 'jit_loss', 'soft_nms', 'max_det_per_image', 'image_size')
    for ov in overrides:
        value = kwargs.pop(ov, None)
        # 如果在kwargs中传入了覆盖值，则更新配置项
        if value is not None:
            setattr(config, ov, value)

    # 获取是否使用benchmark标签器的参数，默认为False
    labeler = kwargs.pop('bench_labeler', False)
    # print(config)
    # 创建基础模型，传入配置和其他参数
    model = EfficientDet(config, pretrained_backbone=pretrained_backbone, **kwargs)

    # 如果指定了预训练权重，加载预训练权重
    if pretrained:
        load_pretrained(model, config.url)

    # 如果num_classes不为None并且与配置中的num_classes不匹配，则重置模型头部
    if num_classes is not None and num_classes != config.num_classes:
        model.reset_head(num_classes=num_classes)

    # 如果指定了训练检查点路径，加载指定的训练检查点
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=checkpoint_ema)

    # 如果bench_task为'train'，则将模型封装为训练模式，使用指定的标签器
    if bench_task == 'train':
        model = DetBenchTrain(model, create_labeler=labeler)
    # 如果bench_task为'predict'，则将模型封装为预测模式
    elif bench_task == 'predict':
        model = DetBenchPredict(model)

    # 返回最终的模型
    return model

