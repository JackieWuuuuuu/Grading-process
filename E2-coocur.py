import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import csv

# 定义病灶颜色
color_def = {
    "EX": ([0, 100, 100], [10, 255, 255]),      
    "H":  ([100, 100, 100], [140, 255, 255]),   
    "MA": ([40, 100, 100], [80, 255, 255]),   
    "CW": ([150, 50, 50], [180, 255, 255])     
}
labels = list(color_def.keys())
label_to_idx = {label: i for i, label in enumerate(labels)}

def normalize_matrix(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

def compute_global_co_matrix(folder_path):
    total_co_matrix = np.zeros((4, 4), dtype=int)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path).convert("RGB").resize((600, 600))
                hsv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
                height, width = hsv_img.shape[:2]
                half_h, half_w = height // 2, width // 2
                regions = [
                    (0, 0, half_h, half_w),
                    (0, half_w, half_h, width),
                    (half_h, 0, height, half_w),
                    (half_h, half_w, height, width)
                ]
                for top, left, bottom, right in regions:
                    region_hsv = hsv_img[top:bottom, left:right]
                    presence_set = set()
                    for label, (lower, upper) in color_def.items():
                        mask = cv2.inRange(region_hsv, np.array(lower), np.array(upper))
                        if np.count_nonzero(mask) > 0:
                            presence_set.add(label)
                    for a in presence_set:
                        for b in presence_set:
                            if a != b:
                                i, j = label_to_idx[a], label_to_idx[b]
                                total_co_matrix[i][j] += 1
            except Exception as e:
                print(f"跳过图像 {filename}, 出错: {e}")
    return total_co_matrix

def get_regions_from_center(height, width, center_x, center_y):
    return [
        (0, 0, center_y, center_x),
        (0, center_x, center_y, width),
        (center_y, 0, height, center_x),
        (center_y, center_x, height, width)
    ]

def visualize_results(image_path, co_matrix_global_norm, center_x, center_y, show_plot=False):
    try:
        img = Image.open(image_path).convert("RGB").resize((600, 600))
        img_array = np.array(img)
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        height, width = hsv_img.shape[:2]

        # 使用图像中心作为默认中心点（如未识别到黄斑）
        if center_x is None or center_y is None:
            center_x, center_y = width // 2, height // 2

        regions = get_regions_from_center(height, width, int(center_x), int(center_y))
        block_scores = []

        for idx, (top, left, bottom, right) in enumerate(regions):
            region_hsv = hsv_img[top:bottom, left:right]
            presence_set = set()
            for label, (lower, upper) in color_def.items():
                mask = cv2.inRange(region_hsv, np.array(lower), np.array(upper))
                if np.count_nonzero(mask) > 0:
                    presence_set.add(label)

            P = np.array([1 if label in presence_set else 0 for label in labels])
            S = co_matrix_global_norm @ P
            indices_present = [i for i, label in enumerate(labels) if label in presence_set]
            S_max = max(S[i] for i in indices_present) if indices_present else 0.0
            block_scores.append(S_max)

        E2 = np.mean(block_scores)

        if show_plot:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            for idx, (top, left, bottom, right) in enumerate(regions):
                region_img = img_array[top:bottom, left:right]
                axes[idx].imshow(region_img)
                axes[idx].set_title(f"Block {idx + 1}")
                axes[idx].axis('off')
            plt.tight_layout()
            plt.show()

        return E2
    except Exception as e:
        print(f"处理图像失败: {e}")
        return None

def get_yolo_center(model_path, detect_image_path):
    resized_image_path = "temp_yolo_input.png"
    image = Image.open(detect_image_path).convert("RGB").resize((600, 600))
    image.save(resized_image_path)
    model = YOLO(model_path)
    results = model(resized_image_path)
    for result in results:
        if result.boxes.xywh is not None and len(result.boxes.xywh) > 0:
            xywh = result.boxes.xywh[0].cpu().numpy()
            return xywh[0], xywh[1]
    return None, None

#主程序
if __name__ == "__main__":
    model_path = r"XXX.pt"
    yolo_input_img_folder = r"A"
    analysis_target_img_folder = r"B"
    output_csv_path = r"XXX.csv"

    # 计算全局共现矩阵
    C_total = compute_global_co_matrix(analysis_target_img_folder)
    C_norm = normalize_matrix(C_total)

    print("\n共现矩阵归一化结果：\n", C_norm)

    # 写入CSV文件
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageName", "Score"])

        for filename in os.listdir(yolo_input_img_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                yolo_img_path = os.path.join(yolo_input_img_folder, filename)
                seg_img_path = os.path.join(analysis_target_img_folder, filename)
                if not os.path.exists(seg_img_path):
                    print(f"跳过 {filename}，分割图像不存在。")
                    continue

                center_x, center_y = get_yolo_center(model_path, yolo_img_path)

                E2 = visualize_results(seg_img_path, C_norm, center_x, center_y, show_plot=False)
                if E2 is not None:
                    writer.writerow([filename, round(E2, 4)])
                    print(f"{filename} 的 E2 值: {E2:.4f}")
                else:
                    print(f"{filename} 处理失败")
