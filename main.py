from fastapi import FastAPI, HTTPException
from nudenet import NudeDetector
import uvicorn
from typing import List, Dict
import os
import concurrent.futures
import asyncio
from pydantic import BaseModel, HttpUrl
import aiohttp
import uuid
import ssl

class ImageUrlRequest(BaseModel):
    url: HttpUrl

class DetectionResponse(BaseModel):
    detections: List[Dict]

app = FastAPI(title="Nude Detection Service")

process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)

TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def init_detector():
    global detector
    detector = NudeDetector(model_path="./640m.onnx", inference_resolution=640)

def process_image(image_path: str) -> List[Dict]:
    try:
        return detector.detect(image_path)
    finally:
        # Удаляем временный файл
        try:
            os.remove(image_path)
        except:
            pass

async def download_image(url: str) -> str:
    """
    Скачивает изображение по URL и сохраняет во временный файл
    """
    conn = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=conn) as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Не удалось скачать изображение: {response.status}")
            
            # Генерируем уникальное имя файла
            file_name = f"{uuid.uuid4()}.jpg"
            temp_path = os.path.join(TEMP_DIR, file_name)
            
            # Сохраняем изображение
            content = await response.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            return temp_path

@app.post("/detect/url", response_model=DetectionResponse)
async def detect_image_url(request: ImageUrlRequest):
    try:
        temp_path = await download_image(str(request.url))
        
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(process_pool, process_image, temp_path)
        
        return DetectionResponse(detections=detections)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    tasks = []
    for _ in range(process_pool._max_workers):
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(process_pool, init_detector)
        tasks.append(task)
    
    await asyncio.gather(*tasks)

@app.on_event("shutdown")
async def shutdown_event():
    process_pool.shutdown(wait=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

all_labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]