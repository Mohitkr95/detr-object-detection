import os
import asyncio
from model import MLModel
from helpers import process_image, perform_live_detection

async def main():
    allowed_extensions = ['.jpg', '.jpeg', '.png']

    input_dir = "./input"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for file_name in os.listdir(input_dir):
        if any(file_name.lower().endswith(ext) for ext in allowed_extensions):
            tasks.append(process_image(input_dir, output_dir, file_name))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(MLModel.loadModel()) 
    asyncio.run(main())
    perform_live_detection()