import os
import cv2
import random
from model import MLModel
import torch
import numpy as np
from PIL import Image

async def draw_rectangles_on_image(image, detection_results):
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for confidence, label, box in detection_results:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(image, start_point, end_point, color, 3)
        label_text = f"{label} - {confidence:.3f}"
        cv2.putText(image, label_text, (start_point[0], start_point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

async def process_image(input_dir, output_dir, file_name):
    image_path = os.path.join(input_dir, file_name)
    image = Image.open(image_path)
    results = await detect_objects(MLModel.cache["processor"], MLModel.cache["model"], image)

    marked_image = await draw_rectangles_on_image(image, results)
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, marked_image)

async def detect_objects(processor, model, image):
    inputs = processor(images=image, return_tensors="pt", size=(800, 800), padding="longest")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    label_names = [model.config.id2label[label.item()] for label in results["labels"]]

    return zip(results["scores"], label_names, results["boxes"])

def detect_objects_live(processor, model, image):
    inputs = processor(images=image, return_tensors="pt", size=(800, 800), padding="longest")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    label_names = [model.config.id2label[label.item()] for label in results["labels"]]

    return zip(results["scores"], label_names, results["boxes"])

def perform_live_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not available.")
        return

    # Get the screen width and height
    screen_width = cap.get(3)
    screen_height = cap.get(4)

    # Create a full-screen window
    cv2.namedWindow('Object Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = detect_objects_live(MLModel.cache["processor"], MLModel.cache["model"], image)
        
        for score, label, box in results:
            if score > 0.5:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"Label: {label}", (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()