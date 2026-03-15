import torch
import torch.nn.functional as F

class AdaInf:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.coco_mean_feature = torch.load("tool/coco_mean_feature.pt").to(self.device)
        self.coco_mean_feature = F.normalize(self.coco_mean_feature.unsqueeze(0), dim=1)
        self.sample_list = []  # 用于存储所有样本特征

    def compute_cosine_distances(self, features):
        features = F.normalize(features, dim=1)  # 归一化样本特征
        cosine_sim = torch.mm(features, self.coco_mean_feature.T).squeeze(1)  # [N]
        cosine_dist = 1 - cosine_sim
        return cosine_dist

    def get_sample_len(self):
        return len(self.sample_list)

    def add_sample(self, detection):
        """
        detection: Tensor of shape [D] or [1, D]
        添加单个样本或多个样本（已是向量）
        """
        if detection.dim() == 1:
            detection = detection.unsqueeze(0)  # 变为 [1, D]
        self.sample_list.append(detection.detach().to(self.device))

    def select_sample(self, top_ratio=0.5):
        for i, t in enumerate(self.sample_list):
            print(f"Tensor {i}: shape {t.shape}")
        """
        将所有累积的样本合并，计算与 COCO 平均特征的余弦距离，选出 top x% 距离最大的样本
        返回：indices, top_features
        """
        if len(self.sample_list) == 0:
            raise ValueError("No samples available. Please add samples first.")

        detections = torch.cat(self.sample_list, dim=0)  # [N, D]
        features = detections.view(detections.size(0), -1)
        distances = self.compute_cosine_distances(features)  # [N]
        topk = max(1, int(top_ratio * len(distances)))  # 至少1个
        topk_indices = torch.topk(distances, topk).indices
        # top_features = features[topk_indices]
        self.sample_list = []  # 用于存储所有样本特征

        return topk_indices
