import json
import os
import cv2
import random
from glob import glob

# 类别信息
categories = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 13, "name": "stop sign"},
    {"id": 14, "name": "parking meter"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
    {"id": 17, "name": "cat"},
    {"id": 18, "name": "dog"},
    {"id": 19, "name": "horse"},
    {"id": 20, "name": "sheep"},
    {"id": 21, "name": "cow"},
    {"id": 22, "name": "elephant"},
    {"id": 23, "name": "bear"},
    {"id": 24, "name": "zebra"},
    {"id": 25, "name": "giraffe"},
    {"id": 27, "name": "backpack"},
    {"id": 28, "name": "umbrella"},
    {"id": 31, "name": "handbag"},
    {"id": 32, "name": "tie"},
    {"id": 33, "name": "suitcase"},
    {"id": 34, "name": "frisbee"},
    {"id": 35, "name": "skis"},
    {"id": 36, "name": "snowboard"},
    {"id": 37, "name": "sports ball"},
    {"id": 38, "name": "kite"},
    {"id": 39, "name": "baseball bat"},
    {"id": 40, "name": "baseball glove"},
    {"id": 41, "name": "skateboard"},
    {"id": 42, "name": "surfboard"},
    {"id": 43, "name": "tennis racket"},
    {"id": 44, "name": "bottle"},
    {"id": 46, "name": "wine glass"},
    {"id": 47, "name": "cup"},
    {"id": 48, "name": "fork"},
    {"id": 49, "name": "knife"},
    {"id": 50, "name": "spoon"},
    {"id": 51, "name": "bowl"},
    {"id": 52, "name": "banana"},
    {"id": 53, "name": "apple"},
    {"id": 54, "name": "sandwich"},
    {"id": 55, "name": "orange"},
    {"id": 56, "name": "broccoli"},
    {"id": 57, "name": "carrot"},
    {"id": 58, "name": "hot dog"},
    {"id": 59, "name": "pizza"},
    {"id": 60, "name": "donut"},
    {"id": 61, "name": "cake"},
    {"id": 62, "name": "chair"},
    {"id": 63, "name": "couch"},
    {"id": 64, "name": "potted plant"},
    {"id": 65, "name": "bed"},
    {"id": 67, "name": "dining table"},
    {"id": 70, "name": "toilet"},
    {"id": 72, "name": "tv"},
    {"id": 73, "name": "laptop"},
    {"id": 74, "name": "mouse"},
    {"id": 75, "name": "remote"},
    {"id": 76, "name": "keyboard"},
    {"id": 77, "name": "cell phone"},
    {"id": 78, "name": "microwave"},
    {"id": 79, "name": "oven"},
    {"id": 80, "name": "toaster"},
    {"id": 81, "name": "sink"},
    {"id": 82, "name": "refrigerator"},
    {"id": 84, "name": "book"},
    {"id": 85, "name": "clock"},
    {"id": 86, "name": "vase"},
    {"id": 87, "name": "scissors"},
    {"id": 88, "name": "teddy bear"},
    {"id": 89, "name": "hair drier"},
    {"id": 90, "name": "toothbrush"}
]

def convert_to_coco(all_data, save_dir, train_ratio=0.8,
                    json_train_name="instances_train2014.json",
                    json_val_name="instances_val2014.json",
                    target_size=(640, 640),
                    max_num=5000,
                    exchange_ratio=0.7):

    os.makedirs(os.path.join(save_dir, "images", "train2014"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images", "val2014"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)

    # 加载现有数据池
    pool_data = []
    existing_images = sorted(glob(os.path.join(save_dir, "images", "train2014", "*.jpg")))
    if existing_images:
        print(f"检测到已有数据池，图片数量: {len(existing_images)}，尝试进行替换...")

        with open(os.path.join(save_dir, "annotations", json_train_name), 'r') as f:
            existing_json = json.load(f)

        image_id_map = {img["id"]: img for img in existing_json["images"]}
        ann_map = {}
        for ann in existing_json["annotations"]:
            ann_map.setdefault(ann["image_id"], []).append(ann)

        for img in existing_json["images"]:
            file_path = os.path.join(save_dir, "images", "train2014", img["file_name"])
            image = cv2.imread(file_path)
            anns = ann_map.get(img["id"], [])
            annotations = [{"bbox": a["bbox"], "category_id": a["category_id"]} for a in anns]

            if image is not None:
                pool_data.append({
                    "image": image,
                    "annotations": annotations
                })

        replace_count = min(int(exchange_ratio * max_num), len(all_data))
        replace_indices = random.sample(range(len(pool_data)), min(replace_count, len(pool_data)))
        insert_data = random.sample(all_data, replace_count)

        for i, idx in enumerate(replace_indices):
            pool_data[idx] = insert_data[i]

        if len(pool_data) < max_num:
            remaining = max_num - len(pool_data)
            additional = random.sample(all_data, min(remaining, len(all_data)))
            pool_data.extend(additional)

        pool_data = pool_data[:max_num]

    else:
        print("未检测到数据池，创建新数据集...")
        pool_data = all_data[:max_num]
        if len(pool_data) < max_num:
            remaining = max_num - len(pool_data)
            extra_data = random.sample(all_data, min(remaining, len(all_data)))
            pool_data.extend(extra_data)

    random.shuffle(pool_data)
    split_idx = int(len(pool_data) * train_ratio)
    train_data = pool_data[:split_idx]
    val_data = pool_data[split_idx:]

    def process_data(data, subset_name, start_img_id=1, start_ann_id=1):
        images, annotations = [], []
        img_id, ann_id = start_img_id, start_ann_id

        labels_dir = os.path.join(save_dir, "labels", subset_name + "2014")
        os.makedirs(labels_dir, exist_ok=True)

        for item in data:
            img = item["image"]
            orig_h, orig_w = img.shape[:2]
            resized_img = cv2.resize(img, target_size)
            file_name = f"{str(img_id).zfill(6)}.jpg"
            save_path = os.path.join(save_dir, "images", subset_name + "2014", file_name)
            cv2.imwrite(save_path, resized_img)

            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": target_size[0],
                "height": target_size[1]
            })

            scale_x = target_size[0] / orig_w
            scale_y = target_size[1] / orig_h

            label_lines = []

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

                center_x = (new_x + new_bw / 2) / target_size[0]
                center_y = (new_y + new_bh / 2) / target_size[1]
                width = new_bw / target_size[0]
                height = new_bh / target_size[1]

                coco_id = ann["category_id"]
                class_id = next(i for i, cat in enumerate(categories) if cat["id"] == coco_id)

                label_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

                ann_id += 1

            label_file = os.path.join(labels_dir, f"{str(img_id).zfill(6)}.txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(label_lines))

            img_id += 1

        return images, annotations

    train_images, train_annotations = process_data(train_data, "train")
    val_images, val_annotations = process_data(val_data, "val")

    with open(os.path.join(save_dir, "annotations", json_train_name), 'w') as f:
        json.dump({
            "images": train_images,
            "annotations": train_annotations,
            "categories": categories
        }, f, indent=4)

    with open(os.path.join(save_dir, "annotations", json_val_name), 'w') as f:
        json.dump({
            "images": val_images,
            "annotations": val_annotations,
            "categories": categories
        }, f, indent=4)

    print(f"转换完成！数据池大小: {len(pool_data)}；训练集: {len(train_images)}；验证集: {len(val_images)}")

    train_txt_path = os.path.join(save_dir, "train2014.txt")
    val_txt_path = os.path.join(save_dir, "val2014.txt")

    with open(train_txt_path, 'w') as f:
        for img in train_images:
            f.write(f"./images/train2014/{img['file_name']}\n")

    with open(val_txt_path, 'w') as f:
        for img in val_images:
            f.write(f"./images/val2014/{img['file_name']}\n")

    print(f"图片路径txt保存完成！train: {train_txt_path}，val: {val_txt_path}")
