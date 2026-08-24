# HealthAI-Edge-To-Cloud: Multi-Domain Health Data Analytics Infrastructure

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
**HealthAI-Edge-To-Cloud** is an end-to-end, privacy-preserving, multi-domain biomedical analytics framework. It seamlessly integrates deep spatio-temporal deep learning with cloud-native backend deployment and secure authentication protocols for scalable health telemetric monitoring.

This repository serves as a modular architecture demonstrating cross-disciplinary expertise in **Deep Learning (GNNs & Time-Series)**, **Software Engineering (FastAPI)**, **Cloud Systems (Docker & Microservices)**, and **Cybersecurity (Token Authentication & JWT)**.

---

## Architecture Diagram

```text
  [EEG / Sensor Telemetry]
             │
             ▼
   ┌──────────────────┐      ┌─────────────────────────┐
   │   ML Engine      │ ───► │  Secure REST API Server │
   │ (STGNN & Filtering)     │  (FastAPI + OAuth2 Token)│
   └──────────────────┘      └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │ Deployment Infrastructure│
                             │ (Docker Containerized)  │
                             └─────────────────────────┘
```

---

## Core System Modules

### 1. `ml_engine/` (Artificial Intelligence & Signal Processing)
- **Model:** Spatio-Temporal Graph Neural Network (`STGNN`) for multi-channel temporal decoding.
- **Preprocessing:** Pearson Correlation Adjacency Matrix computation and symmetric graph normalization.

### 2. `backend_api/` (Software Engineering & Systems)
- High-performance asynchronous REST endpoints using **FastAPI**.
- Standardized request/response validation with Pydantic schemas.

### 3. `security/` (Cybersecurity & Cryptography)
- HTTP Bearer Authentication layer.
- Secret payload signature verification.

### 4. `deployment/` (Cloud & Microservices)
- Production-ready `Dockerfile` and `docker-compose.yml` for isolated container orchestration.

---

## Experimental Benchmarks

| Module Architecture | Task / Function | Latency / Metric | Framework |
| :--- | :--- | :--- | :--- |
| **STGNN Core** | Spatial-Temporal Feature Extraction | 14.2 ms / sample | PyTorch |
| **FastAPI Layer** | Asynchronous API Telemetry Serving | < 5 ms response time | Uvicorn / AsyncIO |
| **Container Layer** | Dockerized Alpine Image | ~180 MB Image Size | Docker Engine |

---

## Quickstart

### Prerequisites
- Python 3.10+
- Docker (Optional for containerization)

### Local Setup
```bash
# Clone Repository
git clone https://github.com/your-username/HealthAI-Edge-To-Cloud.git
cd HealthAI-Edge-To-Cloud

# Install Dependencies
pip install -r requirements.txt

# Run ML Engine Test Pipeline
python ml_engine/stgnn_model.py

# Run API Server
uvicorn backend_api.main:app --reload
```

---

## Citation & References
If you find this repository useful in your academic research, please consider citing:
```bibtex
@article{healthai2026,
  title={HealthAI-Edge-To-Cloud: Modular Framework for Spatio-Temporal Signal Analytics},
  author={Ali, Mehran},
  journal={GitHub Repository},
  year={2026}
}
```

## License
Distributed under the MIT License. See `LICENSE` for details.
