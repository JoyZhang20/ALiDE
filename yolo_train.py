from ultralytics import YOLO


def start_retrain(epochs=10, model_path=None):
    if model_path is None:
        model_path = 'checkpoints/yolov8s.pt'
    model = YOLO(model_path)
    model.train(
        data='coco.yaml',  # 改成你的数据集 yaml 文件路径
        epochs=epochs,  # 100轮左右一般能收敛（小数据集）
        batch=32,  # 4090可以轻松开到64甚至更大，建议64起步
        imgsz=640,  # 输入尺寸，默认640，速度快且兼容性好
        lr0=0.003,  # 初始学习率，适中（不太大防止爆）
        lrf=0.01,  # 最低学习率因子，收敛时能细调
        optimizer='AdamW',  # 推荐用 AdamW，收敛快且稳定性好（小数据集）
        weight_decay=0.001,  # 正则项，防止过拟合
        patience=20,  # 20个epoch val指标不提升就early stop
        device=0,  # 使用第一块GPU
        workers=8,  # 线程数，8或者更高，看CPU配置
        save=True,  # 保存模型
        save_period=10,  # 每10轮保存一次权重
        val=True,  # 每个epoch都验证
        pretrained=True,  # 用yolov8s.pt作为预训练初始化
        seed=42,  # 随机种子，保证可复现
    )


if __name__ == '__main__':
    start_retrain()