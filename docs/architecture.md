# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js 14)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Dashboard │  │ 3D Twin  │  │Simulator │  │Repair AI │    │
│  │(Health)  │  │(R3F+Drei)│  │(What-if) │  │(NLP TF-IDF)│  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       └──────────────┴─────────────┴─────────────┘          │
│                          ▼ HTTP/REST                        │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    FastAPI Backend (:8000)                    │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │              Fusion Orchestration Layer                │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Cross-Domain Attribution (ECS ↔ Engine)        │  │  │
│  │  │  Global Health Scorer (weighted average)        │  │  │
│  │  │  AOG Risk Calculator (P×$150K/hr×hours)         │  │  │
│  │  │  ACARS 220-char Compiler                        │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────┬───────┬───────┬───────┬───────┬───────────┘  │
│              │       │       │       │       │               │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐         │
│  │Engine │ │Hydr.  │ │Land.  │ │ APU   │ │ ECS   │         │
│  │BiLSTM │ │AutoEnc│ │XGBoost│ │  RF   │ │Brayton│         │
│  │+Attn  │ │1D Conv│ │Class. │ │Health │ │Thermo │         │
│  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘         │
│      └─────────┴─────────┴─────────┴─────────┘              │
│                   ONNX Runtime (Phase 2)                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Fusion Architecture (not Ensemble)
We use a **fusion orchestration** pattern, not model ensembling. Each subsystem has an independent model with its own training data, feature space, and validation metrics. The fusion layer aggregates and cross-references outputs — it does not blend raw predictions.

### 2. Cross-Domain Coupling
The ECS↔Engine coupling is the system's core differentiator:
- ECS fouling → increased bleed air demand → elevated engine compressor readings
- An isolated engine model would flag this as engine degradation (false positive)
- The fusion layer checks ECS state before classifying engine anomalies

### 3. Honesty-First API Design
Every prediction response carries `is_synthetic_data: bool`. This is enforced at the Pydantic schema level — it cannot be accidentally omitted.

### 4. Pluggable Inference
Each service class has a `MODEL_LOADED` flag. When `False`, physics-informed simulation is used. When `True`, ONNX Runtime inference is used. The swap requires no API changes.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check with subsystem status |
| `/predict/engine` | POST | Engine RUL prediction |
| `/predict/engine/quick` | GET | Quick engine prediction |
| `/predict/hydraulics` | POST | Hydraulics anomaly detection |
| `/predict/hydraulics/quick` | GET | Quick hydraulics check |
| `/predict/landing-gear` | GET | Landing gear brake wear |
| `/predict/apu` | GET | APU health scoring |
| `/predict/ecs` | GET | ECS health + coupling analysis |
| `/fusion/health` | GET | Full aircraft health assessment |
| `/fusion/acars` | GET | ACARS 220-char compressed alert |
| `/simulate/what-if` | POST | Fault injection simulator |
| `/nlp/recommend` | GET/POST | NLP repair recommendation |

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend | Next.js 14, React Three Fiber, Zustand, Recharts | App Router for routing, R3F for 3D, Zustand for lightweight state |
| Backend | FastAPI, Pydantic v2 | Async Python, auto-generated OpenAPI docs, schema validation |
| ML (Phase 2) | PyTorch → ONNX, XGBoost, scikit-learn | ONNX for portable inference, PyTorch for deep learning |
| NLP (Phase 1) | scikit-learn TF-IDF | Lightweight retrieval. Phase 2: Fine-tuned LLM |
| Data Storage | PostgreSQL, MongoDB Atlas, Elasticsearch, AWS S3 | Polyglot persistence: Postgres (metadata), Mongo (logs), Elastic (telemetry), S3 (models) |
| Testing | pytest + httpx | Async test client for FastAPI |
