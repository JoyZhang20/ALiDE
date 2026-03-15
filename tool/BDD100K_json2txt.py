import json

# 输入文件路径
input_json = "/home/junjie/1_dataset/BDD100K/labels_coco/bdd100k_labels_images_det_coco_val.json"
output_txt = "bdd100k_det_labels.txt"

# 读取 COCO 格式 JSON
with open(input_json, "r") as f:
    data = json.load(f)

images = {img["id"]: img["file_name"] for img in data["images"]}
annotations = data["annotations"]

with open(output_txt, "w") as f:
    for ann in annotations:
        image_id = ann["image_id"]  # 图像 ID
        bbox = ann["bbox"]          # [x, y, w, h]

        # 转换成指定格式：image_id, -1, x, y, w, h, -1, -1, -1
        line = f"{image_id},-1,{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},-1,-1,-1\n"
        f.write(line)

print(f"转换完成，输出文件保存在 {output_txt}")
