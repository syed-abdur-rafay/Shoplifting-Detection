import cv2
import os
import random

# Paths for new dataset (update video_root with your new download path)
video_root = r'C:\Users\Focus Computer\Downloads\shoplifting.v1i.yolov8'  # e.g., root with class folders like 'shoplifter' and 'non_shoplifter'
output_root = r'C:\Users\Focus Computer\Desktop\shop_model_new\data\images'  # Save extracted images here

# Create subfolders
os.makedirs(output_root, exist_ok=True)
for split in ['train', 'val']:
    for class_name in ['non_shoplifter', 'shoplifter']:  # Update class names based on your new dataset
        os.makedirs(os.path.join(output_root, split, class_name), exist_ok=True)

image_count = 0
target_images = 10000

for class_name in ['non_shoplifter', 'shoplifter']:  # Update to match new folder names
    class_video_dir = os.path.join(video_root, class_name)
    if not os.path.exists(class_video_dir):
        print(f"Folder not found: {class_video_dir} - check path.")
        continue
    
    for video_file in os.listdir(class_video_dir):
        if image_count >= target_images:
            break
        video_path = os.path.join(class_video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or image_count >= target_images:
                break
            if frame_num % 5 == 0:
                split = 'train' if random.random() < 0.8 else 'val'
                frame_path = os.path.join(output_root, split, class_name, f'{video_file}_{frame_num}.jpg')
                cv2.imwrite(frame_path, frame)
                image_count += 1
            frame_num += 1
        cap.release()

print(f'Extracted {image_count} images.')