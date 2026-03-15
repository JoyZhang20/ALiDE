import os

input_dir = "output/txt/sample_ratio"
output_file = "output/txt/sample_ratio_merged.txt"

# 获取并排序所有 txt 文件
txt_files = sorted([
    f for f in os.listdir(input_dir)
    if f.endswith(".txt")
])

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in txt_files:
        file_path = os.path.join(input_dir, fname)
        with open(file_path, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")  # 文件之间换行

print(f"✅ 已合并 {len(txt_files)} 个文件 -> {output_file}")
