# Real-Time Credit Risk Scoring System

This project is a Real-Time Credit Risk Scoring system for UPI-based lending.

## Project Structure

- `backend/`: FastAPI application handling API requests, business logic, and model inference.
- `ml/`: Python scripts and Jupyter notebooks for data processing, model training, and exporting to ONNX.
- `frontend/`: React dashboard (Vite) for lenders to view risk scores and borrower data.
- `docker/`: Dockerfiles and `docker-compose.yml` for containerizing the backend, Redis, and MongoDB.
- `data/`: Raw, processed, and synthetic datasets.

## Requirements

- Python 3.11+
- Node 20+
- Docker and Docker Compose

## Quick Start

1. **Environment Variables**: Copy `.env.example` to `.env` and fill in the required values.
2. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
3. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
4. **Docker Services**:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```
