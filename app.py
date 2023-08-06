import os
import logging
import shutil
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import cv2
import uuid
from PIL import Image
from helpers import draw_rectangles_on_image, detect_objects
from model import MLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fastapi_app = FastAPI(
    title='DETR (End-to-End Object Detection)',
    description='The DETR (Detection Transformer) API is a FastAPI-based application designed for end-to-end object detection using the DETR model. This API enables users to upload images for object detection, perform real-time analysis, and receive annotated images with detected objects highlighted.',
    version='1.0.0'
)
TEMP_DATA_DIR = "./temp_data"

@fastapi_app.on_event("startup")
async def startup_event():
    await MLModel.loadModel()

@fastapi_app.post(tags=["Detect"], path="/annotate/")
async def annotate_image(file: UploadFile = File(...)):

    """
    Annotate uploaded image with detected objects.

    Parameters:
    - file: UploadFile
        The image file to be annotated.

    Returns:
    - FileResponse: Annotated image file as response if successful.
    - JSONResponse: Error response if file format is not allowed.
    """
    
    start_time = time.time()

    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    filename = str(uuid.uuid4())
    input_image_path = os.path.join(TEMP_DATA_DIR, f"{filename}.jpg")
    annotated_image_path = os.path.join(TEMP_DATA_DIR, f"{filename}_annotated.jpg")

    with open(input_image_path, "wb") as f_dest, file.file as f_src:
        shutil.copyfileobj(f_src, f_dest)
    
    if any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        image = Image.open(input_image_path)
        results = await detect_objects(MLModel.cache["processor"], MLModel.cache["model"], image)
        annotated_image = await draw_rectangles_on_image(image, results)
        cv2.imwrite(annotated_image_path, annotated_image)
        end_time = time.time()

        processing_time = end_time - start_time
        logger.info(f"Request processing time: {processing_time:.4f} seconds")

        os.remove(input_image_path)
        return FileResponse(annotated_image_path)
    else:
        logger.warning("File format not allowed: %s", file.filename)
        return JSONResponse(
            status_code=400,
            content={"error": "File format not allowed. Allowed extensions: .jpg, .jpeg, .png"}
        )
    

@fastapi_app.post(tags=["Detect"], path="/annotate_video/")
async def annotate_video(file: UploadFile = File(...)):

    """
    Annotate uploaded video with detected objects.

    Parameters:
    - file: UploadFile
        The video file to be annotated.

    Returns:
    - FileResponse: Annotated video file as response if successful.
    - JSONResponse: Error response if file format is not allowed.
    """
    
    start_time = time.time()

    allowed_extensions = {'.mp4', '.avi', '.mov'}
    filename = str(uuid.uuid4())
    input_video_path = os.path.join(TEMP_DATA_DIR, f"{filename}.mp4")
    annotated_video_path = os.path.join(TEMP_DATA_DIR, f"{filename}_annotated.mp4")

    with open(input_video_path, "wb") as f_dest, file.file as f_src:
        shutil.copyfileobj(f_src, f_dest)
    
    if any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        cap = cv2.VideoCapture(input_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = await detect_objects(MLModel.cache["processor"], MLModel.cache["model"], image)
            annotated_frame = await draw_rectangles_on_image(frame, results)
            
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Request processing time: {processing_time:.4f} seconds")

        os.remove(input_video_path)
        return FileResponse(annotated_video_path)
    else:
        logger.warning("File format not allowed: %s", file.filename)
        return JSONResponse(
            status_code=400,
            content={"error": "File format not allowed. Allowed extensions: .mp4, .avi, .mov"}
        )
    
    
@fastapi_app.get(tags=["Cleanup"], path="/delete_files/")
async def delete_temp_files():
    """
    Delete all files from the TEMP_DATA_DIR.

    Returns:
    - dict: A response indicating success or failure.
    """
    try:
        file_list = os.listdir(TEMP_DATA_DIR)
        for file_name in file_list:
            file_path = os.path.join(TEMP_DATA_DIR, file_name)
            os.remove(file_path)
        
        return {"message": "All files deleted successfully"}
    except Exception as e:
        return {"error": str(e)}