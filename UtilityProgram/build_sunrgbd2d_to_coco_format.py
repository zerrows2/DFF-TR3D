import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

# ====== 类别映射（只保留以下10类）======
valid_classes = {
    'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4,
    'desk': 5, 'dresser': 6, 'night_stand': 7, 'bookshelf': 8, 'bathtub': 9
}

# valid_classes = {
#     'bed': 65,                # COCO id
#     'table': 67,              # COCO id
#     'sofa': 63,               # COCO id
#     'chair': 62,              # COCO id
#     'toilet': 70,             # COCO id
#     'desk': 100,              # 自定义，防止和 COCO 冲突
#     'dresser': 101,           # 自定义
#     'night_stand': 102,       # 自定义
#     'bookshelf': 103,         # 自定义
#     'bathtub': 104            # 自定义
# }



# ====== 输入路径 ======
image_dir = "/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_trainval/image"
label_dir = "/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_trainval/label"

# ====== 输出路径 ======
output_base = "/home/star/user/HEYU/DATA/sunrgbd2d/coco_sunrgbd2d"
train_out = os.path.join(output_base, "train")
valid_out = os.path.join(output_base, "valid")
os.makedirs(train_out, exist_ok=True)
os.makedirs(valid_out, exist_ok=True)

# ====== 解析单个txt文件 ======
def parse_txt_file(file_id):
    txt_path = os.path.join(label_dir, f"{file_id}.txt")
    image_path = os.path.join(image_dir, f"{file_id}.jpg")

    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except:
        return None

    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        if cls not in valid_classes:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            if w <= 0 or h <= 0:
                continue
        except:
            continue
        ann = {
            "bbox": [x, y, w, h],
            "category_id": valid_classes[cls],
            "area": w * h,
            "iscrowd": 0
        }
        annotations.append(ann)

    return {
        "file_name": f"{file_id}.jpg",
        "id": int(file_id),
        "annotations": annotations
    }

# ====== 构建COCO格式JSON ======
def build_coco_json(image_ids, split_dir):
    images = []
    annotations = []
    ann_id = 1

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_txt_file, image_ids), total=len(image_ids)))

    for result in results:
        if result is None or len(result['annotations']) == 0:
            continue
        images.append({
            "file_name": result["file_name"],
            "id": result["id"],
            "width": 640,    # 若你有实际尺寸可以替换这里
            "height": 480
        })
        for ann in result['annotations']:
            ann.update({
                "image_id": result["id"],
                "id": ann_id
            })
            annotations.append(ann)
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k, "supercategory": "object"} for k, v in valid_classes.items()]
    }

    with open(os.path.join(split_dir, "_annotations.coco.json"), 'w') as f:
        json.dump(coco_dict, f, indent=2)

    return len(images), len(annotations)


# ====== 复制jpg图片 ======
def copy_images(image_ids, dst_dir):
    for file_id in tqdm(image_ids, desc=f"Copying images to {dst_dir}"):
        src_img = os.path.join(image_dir, f"{file_id}.jpg")
        dst_img = os.path.join(dst_dir, f"{file_id}.jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)


# ====== 构建 ID 列表并执行转换 ======
val_ids = [f"{i:06d}" for i in range(1, 5051)]
train_ids = [f"{i:06d}" for i in range(5051, 10336)]

print("Processing validation set...")
val_stats = build_coco_json(val_ids, valid_out)
print(f"Validation set: {val_stats[0]} images, {val_stats[1]} annotations")

print("Processing training set...")
train_stats = build_coco_json(train_ids, train_out)
print(f"Training set: {train_stats[0]} images, {train_stats[1]} annotations")

# ====== 复制图片文件到对应目录 ======
copy_images(val_ids, valid_out)
copy_images(train_ids, train_out)
