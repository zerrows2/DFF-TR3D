# 寻找固定ROI区域

import os

def load_box(line):
    parts = line.strip().split()
    cls, conf = parts[0], float(parts[1])
    x1, y1, x2, y2 = map(float, parts[2:])
    return {'cls': cls, 'conf': conf, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

def box_part(b, region='right_upper'):
    # 将原始框分成四象限中的一个子区域
    x_mid = (b['x1'] + b['x2']) / 2
    y_mid = (b['y1'] + b['y2']) / 2

    if region == 'right_upper':
        return {'x1': x_mid, 'y1': b['y1'], 'x2': b['x2'], 'y2': y_mid}
    elif region == 'left_lower':
        return {'x1': b['x1'], 'y1': y_mid, 'x2': x_mid, 'y2': b['y2']}
    else:
        raise ValueError("Unknown region")

def boxes_overlap(a, b):
    # 判断两个框是否有重叠区域
    return not (a['x2'] <= b['x1'] or a['x1'] >= b['x2'] or
                a['y2'] <= b['y1'] or a['y1'] >= b['y2'])

def special_overlap(b1, b2):
    ru1 = box_part(b1, 'right_upper')
    ll2 = box_part(b2, 'left_lower')
    ru2 = box_part(b2, 'right_upper')
    ll1 = box_part(b1, 'left_lower')
    return boxes_overlap(ru1, ll2) or boxes_overlap(ru2, ll1)

# 数据路径
data_path = "/home/star/user/HEYU/DATA/sunrgbd2d/rf-detr/rf-detr_cof0.3_iou0.45"

# 遍历所有 txt 文件
for idx in range(1, 10336):
    filename = f"{idx:06d}.txt"
    filepath = os.path.join(data_path, filename)

    if not os.path.exists(filepath):
        continue

    with open(filepath, 'r') as f:
        lines = f.readlines()
        if len(lines) != 2:
            continue

        box1 = load_box(lines[0])
        box2 = load_box(lines[1])

        if special_overlap(box1, box2):
            print(f"{filename}")
