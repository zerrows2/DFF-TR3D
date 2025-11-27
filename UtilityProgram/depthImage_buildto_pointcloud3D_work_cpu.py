import numpy as np
import cv2
import scipy.io as sio
import os
from tqdm import tqdm
from typing import Tuple
from joblib import Parallel, delayed

# ================= 工具函数 =================

def parse_matrix_line(line: str) -> np.ndarray:
    line = line.strip().replace('[', '').replace(']', '')
    rows = line.split(';')
    matrix = np.array([[float(num) for num in row.split(',')] for row in rows], dtype=np.float32)
    return matrix

def decode_sunrgbd_depth(depth_vis: np.ndarray) -> np.ndarray:
    depth_inpaint = np.bitwise_or(np.right_shift(depth_vis, 3), np.left_shift(depth_vis, 13))
    depth_inpaint = depth_inpaint.astype(np.float32) / 1000.0
    depth_inpaint[depth_inpaint > 8] = 8.0
    return depth_inpaint

def read_3d_pts_general(depth_inpaint, K, depth_inpaint_size, image_name=None):
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    H, W = depth_inpaint_size
    invalid = (depth_inpaint == 0)

    if image_name and os.path.exists(image_name):
        im = cv2.imread(image_name, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rgb = im.astype(np.float32) / 255.0
    else:
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[:, :, 1] = 1.0

    rgb_flat = rgb.reshape(-1, 3)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x3 = (x - cx) * depth_inpaint / fx
    y3 = (y - cy) * depth_inpaint / fy
    z3 = depth_inpaint

    points3d_matrix = np.stack((x3, z3, -y3), axis=-1)
    points3d_matrix[invalid] = np.nan
    points3d = points3d_matrix.reshape(-1, 3)
    points3d[invalid.flatten()] = np.nan

    return rgb_flat, points3d

def read3dPoints(depthpath: str, K: np.ndarray, rgbpath: str, Rtilt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth_vis = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    if depth_vis is None:
        raise FileNotFoundError(f"无法读取深度图: {depthpath}")
    imsize = depth_vis.shape[:2]
    depth_inpaint = decode_sunrgbd_depth(depth_vis)
    rgb, points3d = read_3d_pts_general(depth_inpaint, K, imsize, image_name=rgbpath)
    valid_mask = ~np.isnan(points3d).any(axis=1)
    points3d[valid_mask] = (Rtilt @ points3d[valid_mask].T).T
    return rgb, points3d

# ================= 路径设置 =================

depth_dir = "/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_trainval/depth_image_2dbboxrs_YOLOv11x"
rgb_dir = "/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_trainval/image"
save_dir = "/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_trainval/depth_2dbboxes_YOLOv11x_10w"
os.makedirs(save_dir, exist_ok=True)

K_txt_path = "/home/star/user/HEYU/DATA/sunrgbd/KR/K.txt"
R_txt_path = "/home/star/user/HEYU/DATA/sunrgbd/KR/R.txt"

# ================= 加载 K 和 R =================

with open(K_txt_path, 'r') as f_k, open(R_txt_path, 'r') as f_r:
    K_lines = f_k.readlines()
    R_lines = f_r.readlines()

assert len(K_lines) == len(R_lines), "K.txt 和 R.txt 行数不一致！"

# ================= 并行处理函数 =================

def process_sample(i: int):
    id_str = f"{i:06d}"
    depth_path = os.path.join(depth_dir, f"{id_str}.png")
    rgb_path = os.path.join(rgb_dir, f"{id_str}.jpg")
    save_path = os.path.join(save_dir, f"{id_str}.mat")

    try:
        K = parse_matrix_line(K_lines[i - 1])
        Rtilt = parse_matrix_line(R_lines[i - 1])
        rgb, points3d = read3dPoints(depth_path, K, rgb_path, Rtilt)
        valid_mask = ~np.isnan(points3d).any(axis=1)
        points3d_rgb = np.hstack((points3d[valid_mask], rgb[valid_mask])).astype(np.float32)
        sio.savemat(save_path, {'instance': points3d_rgb})
    except Exception as e:
        return f"{id_str} 失败: {e}"
    return None

# ================= 并行运行 =================

print(f"🌟 开始并行处理 {len(K_lines)} 个样本...")

errors = Parallel(n_jobs=64)(
    delayed(process_sample)(i) for i in tqdm(range(1, len(K_lines) + 1))
)

# 打印错误信息（如果有）
errors = list(filter(None, errors))
if errors:
    print("\n❌ 以下样本处理失败：")
    for err in errors:
        print(err)
else:
    print("\n✅ 全部完成，无错误。")
