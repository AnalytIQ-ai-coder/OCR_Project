import time
import pika
import cv2
import numpy as np

from app.core.engine import analyze_image, load_models
from app.db.repository import save_result
from app.db.repository import init_db

RABBITMQ_HOST = "rabbitmq"
QUEUE_NAME = "plates"

def connect_rabbitmq(retries=30, delay=2):

    for i in range(retries):
        try:
            return pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )
            )
        except pika.exceptions.AMQPConnectionError:
            print(f"[worker] RabbitMQ not ready ({i+1}/{retries}), retrying...")
            time.sleep(delay)

    raise RuntimeError("Could not connect to RabbitMQ")

def callback(ch, method, properties, body):
    try:
        img = np.frombuffer(body, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data")

        result = analyze_image(img)
        save_result(result)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("[worker] Error while processing message:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def main():
    print("[worker] Booting...")

    print("[worker] Loading models...")
    load_models()
    print("[worker] Models loaded")

    print("[worker] Initializing database...")
    init_db()

    conn = connect_rabbitmq()
    ch = conn.channel()

    ch.queue_declare(queue=QUEUE_NAME, durable=True)

    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)

    print("[worker] Ready, waiting for messages")
    ch.start_consuming()

if __name__ == "__main__":
    main()
