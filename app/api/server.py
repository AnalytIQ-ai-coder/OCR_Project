from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import pika

from app.core.engine import analyze_image, load_models
app = FastAPI()

@app.on_event("startup")
def startup():
    load_models()

@app.post("/analyze")
async def analyze(file: UploadFile):
    img = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return analyze_image(img)

@app.post("/enqueue")
async def enqueue(file: UploadFile):
    body = await file.read()

    conn = pika.BlockingConnection(
        pika.ConnectionParameters("rabbitmq")
    )
    ch = conn.channel()
    ch.queue_declare(queue="plates", durable=True)

    ch.basic_publish(
        exchange="",
        routing_key="plates",
        body=body
    )

    conn.close()
    return {"status": "queued"}

@app.get("/results")
def results():
    import sqlite3
    conn = sqlite3.connect("/app/data/results.db")
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM results ORDER BY created_at DESC").fetchall()
    return rows
