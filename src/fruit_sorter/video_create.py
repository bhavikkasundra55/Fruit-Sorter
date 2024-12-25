import cv2
import numpy as np
import time
import os

IMG_SIZE = (224, 224)
CLASS_LABELS = ['Stage_0 (Unripe)', 'Stage_1 (Early Ripe)', 'Stage_2 (Partially Ripe)', 'Stage_3 (Ripe)']
IMAGE_FOLDER = 'dataset/Stage_0 (Unripe)/Test'  # Replace with the path to your image folder
VIDEO_OUTPUT = 'unripe.mp4'
FPS = 3  

def create_video_from_images(image_folder, video_path, fps):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  
    
    if not images:
        print("No images found in the folder.")
        return
    
    print(len(images))

    # Get the size of the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {video_path}")

    

create_video_from_images(IMAGE_FOLDER, VIDEO_OUTPUT, FPS)