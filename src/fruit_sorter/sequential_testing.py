import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = 'models/mango_ripeness_mobilenetv2.h5'
CLASS_LABELS = ['Stage_0 (Unripe)', 'Stage_1 (Early Ripe)', 'Stage_2 (Partially Ripe)', 'Stage_3 (Ripe)']
IMAGE_FOLDER = 'dataset/Stage_1(Early Ripe)/Training'  # Replace with the path to your image folder
VIDEO_OUTPUT = 'earlyripe.mp4'
FPS = 10  

# Load pre-trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Step 1: Create Video from Images
def create_video_from_images(image_folder, video_path, fps):
    # Fetch all image files with supported extensions
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Sort images alphabetically

    if not images:
        print("No valid images found in the folder.")
        return

    print(f"Found {len(images)} images.")

    # Read the first image to determine video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)

    if first_frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return

    height, width, _ = first_frame.shape
    print(f"Video resolution: {width}x{height}")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Process and write each image to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Skipping invalid image: {image_path}")
            continue

        if frame.shape[:2] != (height, width):
            print(f"Resizing image: {image_path}")
            frame = cv2.resize(frame, (width, height))

        video_writer.write(frame)
        print(f"Added frame: {image_path}")

    video_writer.release()
    print(f"Video saved successfully at {video_path}")

# Step 2: Process Video for Real-Time Prediction
def process_video_with_predictions(video_path, model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, IMG_SIZE)
        normalized_frame = resized_frame / 255.0
        input_array = np.expand_dims(normalized_frame, axis=0)

        # Prediction
        start_time = time.time()
        prediction = model.predict(input_array, verbose=0)
        end_time = time.time()
        predicted_class = CLASS_LABELS[np.argmax(prediction)]

        # Overlay prediction
        cv2.putText(
            frame,
            f"Class: {predicted_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        cv2.putText(
            frame,
            f"Time: {int((end_time - start_time) * 1000)} ms",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        # Display frame
        cv2.imshow("Video with Predictions", frame)
        if cv2.waitKey(10000 // FPS) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Execute steps
# create_video_from_images(IMAGE_FOLDER, VIDEO_OUTPUT, FPS)
process_video_with_predictions(VIDEO_OUTPUT, model)
