import json
import os
import cv2
import random
# 假设你已有的数据列表如下（这里只是形式示意）
# all_data = [
#     {
#         "image": img,  # cv2.imread读入的图像
#         "annotations": [
#             {"bbox": [x1, y1, w1, h1], "category_id": 0},
#             {"bbox": [x2, y2, w2, h2], "category_id": 1}
#         ]
#     },
#     ...
# ]

def load_data_from_custom_format(image_dir, label_file):
    all_data = []
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # 每行代表一个目标，每一帧图像可能有多个目标
    annotations_per_image = {}

    for line in lines:
        parts = line.strip().split(',')
        frame_id = int(parts[0])
        x, y, w, h = map(float, parts[2:6])
        category_id = 2  # 假设都是 car（你定义的类别 id 是 2）

        ann = {
            "bbox": [x, y, w, h],
            "category_id": category_id
        }

        if frame_id not in annotations_per_image:
            annotations_per_image[frame_id] = []
        annotations_per_image[frame_id].append(ann)

    for frame_id, ann_list in annotations_per_image.items():
        img_path = os.path.join(image_dir, f"{str(frame_id).zfill(5)}.jpg")
        if not os.path.exists(img_path):
            continue  # 防止标签中有图像不存在
        img = cv2.imread(img_path)
        all_data.append({
            "image": img,
            "annotations": ann_list
        })

    return all_data

def convert_to_coco(all_data, save_dir, train_ratio=0.8,
                    json_train_name="instances_train2014.json",
                    json_val_name="instances_val2014.json",
                    target_size=(640, 640)):

    os.makedirs(os.path.join(save_dir, "train2014"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "val2014"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)

    # 随机打乱并划分数据
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    def process_data(data, subset_name, start_img_id=1, start_ann_id=1):
        images = []
        annotations = []
        img_id = start_img_id
        ann_id = start_ann_id

        for item in data:
            img = item["image"]
            orig_h, orig_w = img.shape[:2]

            # 缩放图像到目标尺寸
            resized_img = cv2.resize(img, target_size)
            new_h, new_w = target_size
            file_name = f"{str(img_id).zfill(6)}.jpg"
            save_path = os.path.join(save_dir, subset_name + "2014", file_name)

            # 保存图像
            cv2.imwrite(save_path, resized_img)

            # 添加图像信息
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": new_w,
                "height": new_h
            })

            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            # 添加标注信息
            for ann in item["annotations"]:
                x, y, bw, bh = ann["bbox"]
                new_x = x * scale_x
                new_y = y * scale_y
                new_bw = bw * scale_x
                new_bh = bh * scale_y

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [new_x, new_y, new_bw, new_bh],
                    "area": new_bw * new_bh,
                    "iscrowd": 0
                })
                ann_id += 1

            img_id += 1

        return images, annotations

    # 类别信息（请根据需要修改）
    categories = [
        {"id": 0, "name": "class_0"},
        {"id": 1, "name": "class_1"},
        {"id": 2, "name": "car"}
    ]

    # 处理训练集
    train_images, train_annotations = process_data(train_data, "train")
    train_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    with open(os.path.join(save_dir, "annotations", json_train_name), 'w') as f:
        json.dump(train_json, f, indent=4)

    # 处理验证集
    val_images, val_annotations = process_data(val_data, "val")
    val_json = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }
    with open(os.path.join(save_dir, "annotations", json_val_name), 'w') as f:
        json.dump(val_json, f, indent=4)

    print(f"转换完成！训练集: {len(train_images)} 张图，验证集: {len(val_images)} 张图")

if __name__ == '__main__':
    image_dir = "../video_frame/MVI_39031"
    label_file = "../video_frame/labels/MVI_39031.txt"
    save_dir = "output_coco"  # 输出 COCO 格式的保存路径

    all_data = load_data_from_custom_format(image_dir, label_file)
    convert_to_coco(all_data, save_dir)