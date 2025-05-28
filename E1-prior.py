import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib
from PIL import Image

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 模型
MACULA_MODEL_PATH = r"xxx.pt"
macula_model = YOLO(MACULA_MODEL_PATH)

# 获取黄斑中心坐标
def get_macula_center(image_path):
    results = macula_model(image_path)
    xywh = results[0].boxes.xywh
    if len(xywh) == 0:
        print(f"图像 {image_path} 未检测到黄斑中心，使用图像中心代替")
        return None
    center = xywh[0][:2].cpu().numpy()
    return tuple(map(int, center))

# 计算 E1 分数
def calculate_E1(lesion_image_path, macula_center, visualize=False):
    color_def = {
        "EX": ([0, 100, 100], [10, 255, 255]),      
        "H":  ([100, 100, 100], [140, 255, 255]),   
        "MA": ([40, 100, 100], [80, 255, 255]),    
        "CW": ([150, 50, 50], [180, 255, 255])      
    }

    weights = {
        "MA": 0.1,
        "H": 0.2,
        "EX": 0.3,
        "CW": 0.4
    }

    lesion_img = Image.open(lesion_image_path).convert("RGB").resize((600, 600))
    lesion_array = np.array(lesion_img)
    hsv_lesion = cv2.cvtColor(lesion_array, cv2.COLOR_RGB2HSV)

    width, height = 600, 600
    if macula_center:
        center_x, center_y = macula_center
    else:
        center_x, center_y = width // 2, height // 2

    # 象限区域
    regions = [
        (0, 0, center_y, center_x),                   
        (0, center_x, center_y, width),               
        (center_y, 0, height, center_x),            
        (center_y, center_x, height, width)           
    ]

    results = []

    for i, (top, left, bottom, right) in enumerate(regions):
        region_hsv = hsv_lesion[top:bottom, left:right]
        total_pixels = region_hsv.shape[0] * region_hsv.shape[1]
        scaled_total = total_pixels // 100

        counts = {}
        for label, (lower, upper) in color_def.items():
            mask = cv2.inRange(region_hsv, np.array(lower), np.array(upper))
            counts[label] = np.count_nonzero(mask)

        percentages = {label: (count / scaled_total) for label, count in counts.items()}
        scores = {label: weights[label] * (1 + np.sqrt(percentages[label])) for label in color_def}

        max_label = max(scores, key=scores.get)
        max_score = scores[max_label]

        results.append({
            "region": i + 1,
            "percentages": percentages,
            "raw_counts": counts,
            "scores": scores,
            "max_score": max_score,
            "max_label": max_label
        })

    E1 = np.mean([r["max_score"] for r in results])
    return E1, results

# 主程序
def process_with_dual_folders(folder_A, folder_B, output_csv):
    filenames = os.listdir(folder_A)
    output_data = []

    for filename in filenames:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path_A = os.path.join(folder_A, filename)
        path_B = os.path.join(folder_B, filename)

        if not os.path.isfile(path_B):
            print(f"病灶图缺失: {filename}")
            continue

        image = Image.open(path_A).convert("RGB").resize((600, 600))
        temp_path = "temp_input.png"
        image.save(temp_path)

        macula_center = get_macula_center(temp_path)
        E1_score, details = calculate_E1(path_B, macula_center)

        print(f"{filename} -> E1: {E1_score:.4f}")
        output_data.append([filename, E1_score])

    # 写入 CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'Score'])
        writer.writerows(output_data)

    print(f"\n所有处理完成，结果保存至 {output_csv}")

# 调用
if __name__ == "__main__":
    folder_A = r"A"  
    folder_B = r"B"   
    output_csv = r"XXX.csv"
    process_with_dual_folders(folder_A, folder_B, output_csv)
