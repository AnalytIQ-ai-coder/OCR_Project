# OCR Project

Asynchronous license plate OCR service built with **FastAPI**, **RabbitMQ**, **SQLite**, and worker processes. The API accepts images, enqueues them for background processing, and stores OCR results in a local database.

## Architecture

- **API**: FastAPI app that exposes HTTP endpoints for synchronous OCR and queueing jobs.
- **Worker**: RabbitMQ consumer that runs OCR and writes results to SQLite.
- **Database**: SQLite database mounted into the containers at `/app/data/results.db`.

## Prerequisites

- Docker + Docker Compose
- (Optional) GPU with CUDA if you want EasyOCR to use GPU

## Quick start (Docker)

```bash
docker compose up --build
```

The API will be available at: `http://localhost:8000`

## API usage

### 1) Synchronous OCR

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@/path/to/plate.jpg"
```

### 2) Enqueue a job for background processing

```bash
curl -X POST "http://localhost:8000/enqueue" \
  -F "file=@/path/to/plate.jpg"
```

### 3) Read saved results

```bash
curl "http://localhost:8000/results"
```

## Project layout

```
app/
  api/        # FastAPI server
  core/       # OCR + detection logic
  db/         # SQLite repository helpers
  queue/      # RabbitMQ publisher helpers
  workers/    # Background consumers
```

## Notes

- The YOLO weights are expected at `runs/plate_yolo/weights/best.pt`.
- The worker and API containers share the same database volume (`/app/data`).

