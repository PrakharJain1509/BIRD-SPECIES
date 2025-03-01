import os
import cv2
from ultralytics import YOLO

# Path to dataset and output folder for classification
data_folder = "dataset"
classification_data_folder = "classification_data"
os.makedirs(classification_data_folder, exist_ok=True)

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # Choose yolov8n, yolov8s, etc., based on your GPU capacity

# Map COCO class IDs to bird species (update with your specific species mapping)
class_mapping = {
    0: "blasti",    # Replace with actual species name
    1: "bonegl",    # Replace with actual species name
    2: "brhkyt",    # Replace with actual species name
    3: "cbrtsh",    # Replace with actual species name
    4: "cmnmyn",    # Replace with actual species name
    5: "gretit",    # Replace with actual species name
    6: "hilpig",    # Replace with actual species name
    7: "himbul",    # Replace with actual species name
    8: "himgri",    # Replace with actual species name
    9: "hsparo",    # Replace with actual species name
    10: "indvul",   # Replace with actual species name
    11: "kjpond",   # Replace with actual species name
    12: "lbccrw",   # Replace with actual species name
    13: "mgprob",   # Replace with actual species name
    14: "piedbu",   # Replace with actual species name
    15: "ribgul",   # Replace with actual species name
    16: "rufsun",   # Replace with actual species name
    17: "whthro",   # Replace with actual species name
    18: "wlbpro",   # Replace with actual species name
    19: "wtblbt",   # Replace with actual species name
}

# Function to process images and save cropped bird images
def process_and_crop(input_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)

                # Perform object detection
                results = model.predict(img, conf=0.25)

                # Save cropped bird images
                for result in results:
                    boxes = result.boxes.xyxy  # Get bounding box coordinates
                    class_ids = result.boxes.cls  # Get class IDs

                    for box, class_id in zip(boxes, class_ids):
                        class_id = int(class_id)
                        if class_id in class_mapping:  # Check if the class is a bird species
                            x1, y1, x2, y2 = map(int, box)
                            cropped_img = img[y1:y2, x1:x2]

                            # Save cropped image in the class folder
                            species_folder = os.path.join(classification_data_folder, class_mapping[class_id])
                            os.makedirs(species_folder, exist_ok=True)
                            cropped_img_path = os.path.join(species_folder, f"{os.path.basename(file_path)}")
                            cv2.imwrite(cropped_img_path, cropped_img)

# Directories for training and testing
train_data_path = os.path.join(data_folder, "train_data")
test_data_path = os.path.join(data_folder, "test_data")

process_and_crop(train_data_path)
process_and_crop(test_data_path)

print("Processing complete. Cropped bird images are saved for classification.")
