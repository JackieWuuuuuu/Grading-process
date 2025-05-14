import os
import csv
from ultralytics import YOLO
import torch
import math
from tqdm import tqdm

lesion_params = {
    'MA': {'R': 3.0, 'w': 0.1},
    'H': {'R': 2.5, 'w': 0.2},
    'EX': {'R': 2.0, 'w': 0.3},
    'CW': {'R': 1.5, 'w': 0.4},
    'NB': {'R': 1.0, 'w': 0.0}, 

PIXEL_TO_MM = 30 / 256
IMAGE_RADIUS_MM = 15.0 

def get_center_and_class(result):
    xywh = result.boxes.xywh.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    lesion_classes = ['MA', 'H', 'EX', 'CW', 'NB']
    results = []
    for (x, y, _, _), cls_id in zip(xywh, cls_ids):
        class_name = lesion_classes[cls_id] if cls_id < len(lesion_classes) else f"Class_{cls_id}"
        results.append(((x, y), class_name))
    return results

def get_single_center(result):
    xywh = result.boxes.xywh
    centers = xywh[:, :2].cpu().numpy()
    return centers[0] if len(centers) > 0 else None

def euclidean_distance_mm(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dist_px = math.sqrt(dx**2 + dy**2)
    return dist_px, dist_px * PIXEL_TO_MM

def calculate_E_spatial(lesion_data, macula_center):
    E_spatial = 0.0
    for (lesion_center, lesion_type) in lesion_data:
        if lesion_type not in lesion_params:
            continue
        _, d_mm = euclidean_distance_mm(lesion_center, macula_center)
        R_l = lesion_params[lesion_type]['R']
        w_l = lesion_params[lesion_type]['w']
        E = w_l * (d_mm / R_l) ** 2
        E_spatial += E
    return E_spatial

if __name__ == '__main__':
    image_folder = r'C:\UltraLight-VMUNet-Output\IDRiD_Output\4_pdr\Source_Photo'
    output_csv = r'C:\UltraLight-VMUNet-Output\IDRiD_Output\4_pdr\4-E3.csv'

    # 加载模型
    model_lesion = YOLO(r'C:\PythonCode\ultralytics-main\runs\detect\train12\weights\best.pt')
    model_macula = YOLO(r'C:\PythonCode\ultralytics-v11-main\runs\train\exp2\weights\best.pt')

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = []

    for img_file in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(image_folder, img_file)

        # 检测病灶
        results_lesion = model_lesion(img_path, verbose=False)
        lesion_data = get_center_and_class(results_lesion[0])

        # 检测黄斑
        results_macula = model_macula(img_path, verbose=False)
        macula_center = get_single_center(results_macula[0])

        if macula_center is None:
            print(f"图像 {img_file} 未检测到黄斑区域，得分设置为 1")
            results.append((img_file, 1.0))
            continue

        # 计算空间代价
        E_spatial = calculate_E_spatial(lesion_data, macula_center)
        E_spatial_normalized = E_spatial / (IMAGE_RADIUS_MM ** 2)

        results.append((img_file, round(E_spatial_normalized, 4)))

    # 写入 CSV 文件
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageName", "Score"])
        writer.writerows(results)

    print(f"\n所有图像处理完成，结果已保存至: {output_csv}")
