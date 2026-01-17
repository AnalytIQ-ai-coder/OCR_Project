import pika

RABBITMQ_HOST = "rabbitmq"
QUEUE = "plates"


def enqueue_image(image_bytes: bytes):
    conn = pika.BlockingConnection(
        pika.ConnectionParameters(RABBITMQ_HOST)
    )
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE, durable=True)

    ch.basic_publish(
        exchange="",
        routing_key=QUEUE,
        body=image_bytes,
    )

    conn.close()
