# MolGen — Conditional Molecular Generation with Graph Diffusion Networks

A system that generates novel drug-like molecules conditioned on user-specified chemical properties. Set target LogP, QED, SA Score, and HOMO-LUMO gap — the model generates valid molecular structures matching those targets.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

---

## What This Is

Most ML projects treat molecules as SMILES strings (sequences of characters). MolGen treats them as what they actually are — **graphs** — where atoms are nodes and bonds are edges. A Graph Diffusion Network learns to denoise corrupted molecular graphs, conditioned on target property vectors.

| Standard approach | MolGen |
|---|---|
| Call a pretrained API | Train generative model from scratch |
| Tabular or image data | Graph-structured data (PyTorch Geometric) |
| Classification / regression | Conditional generative modeling |
| Unconditional generation | Conditional generation (property-guided) |
| No evaluation beyond accuracy | Multi-metric evaluation (validity, uniqueness, novelty, FCD) |

---

## Architecture
QM9 (131k molecules)
↓
[Data Pipeline]
SMILES → RDKit → PyG Data object
Node features: atom type, charge, hybridisation, aromaticity, H count  [18-dim]
Edge features: bond type, conjugation, ring membership                  [6-dim]
↓
[Phase 1: MPNN Property Predictor]
4 rounds of message passing with GRU updates
Multi-task heads: LogP, QED, SA Score, HOMO-LUMO gap
Validation MAE: LogP 0.022 | QED 0.045 | SA 0.073 | HOMO-LUMO 0.106
↓
[Phase 2+3: Graph Diffusion Model]
Forward:  corrupt node features (Gaussian) + adjacency (discrete bit-flips)
Reverse:  GNN denoiser conditioned on property vector c
Guidance: classifier-free guidance (w=2.0) at inference
↓
[FastAPI Backend]
POST /generate          → conditioned molecule sampling
POST /predict_properties → GNN property prediction for any SMILES
GET  /visualize/{smiles} → RDKit 2D structure SVG
GET  /health            → model status
↓
[Next.js Dashboard]
Property sliders → Generate → Molecule gallery with 2D structures

---

## Results

**Phase 1 — Property Predictor** (trained on full QM9, 131k molecules):

| Property | Val MAE | Target |
|---|---|---|
| LogP | 0.022 | < 0.1 ✅ |
| QED | 0.045 | < 0.05 ✅ |
| SA Score | 0.073 | < 0.1 ✅ |
| HOMO-LUMO gap | 0.106 | < 0.1 ⚠️ |

**Phase 2 — Unconditional Diffusion** (node denoising loss: 0.030)

**Phase 3 — Conditional Diffusion** (fine-tuned with classifier-free guidance)

---

## Tech Stack

**ML:** PyTorch, PyTorch Geometric, RDKit, fcd-torch  
**Backend:** FastAPI, Uvicorn  
**Frontend:** Next.js 14 (App Router), TypeScript, Tailwind CSS  
**Data:** QM9 dataset (131k molecules, downloaded automatically)  

---

## Project Structure
molgen/
├── molgen/
│   ├── data/
│   │   ├── graph_builder.py          # SMILES → PyG Data object
│   │   ├── property_calculator.py    # RDKit property computation + normalisation
│   │   ├── dataset.py                # QM9 download, processing, DataLoader
│   │   └── augmentation.py
│   ├── models/
│   │   ├── encoder.py                # MPNN property predictor
│   │   ├── denoiser.py               # GNN denoiser with time + conditioning
│   │   └── conditioning.py           # Property conditioning MLP with CFG
│   ├── training/
│   │   ├── train_encoder.py          # Phase 1 training script
│   │   └── train_diffusion.py        # Phase 2 + 3 training script
│   ├── evaluation/
│   │   ├── metrics.py                # Validity, uniqueness, novelty, FCD
│   │   └── evaluator.py              # Full evaluation pipeline
│   ├── api/
│   │   ├── app.py                    # FastAPI endpoints
│   │   ├── inference.py              # Model loading + generation logic
│   │   └── visualizer.py            # RDKit 2D rendering
│   └── utils/
│       ├── noise_schedule.py         # Cosine noise schedule
│       └── config.py                 # Config loading
├── frontend/                         # Next.js 14 dashboard
├── tests/                            # 110 tests across all modules
├── configs/
│   ├── encoder_config.json
│   └── diffusion_config.json
└── requirements.txt

---

## Setup

**Prerequisites:** Python 3.11, conda, Node.js 18+

```bash
# Clone
git clone https://github.com/Harsh-617/molgen.git
cd molgen

# Python environment
conda create -n molgen python=3.11 -y
conda activate molgen

# PyTorch (CPU for local dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All dependencies
pip install torch-geometric rdkit fcd-torch scikit-learn tqdm
pip install fastapi uvicorn python-multipart matplotlib seaborn pillow pydantic
pip install pytest pytest-asyncio httpx
pip install -e .

# Frontend
cd frontend && npm install
```

---

## Training

**Note:** Full training requires a GPU. Use Google Colab (free T4) if you don't have one locally.

```bash
# Phase 1 — Property predictor (~40 min on T4)
python -m molgen.training.train_encoder

# Phase 2 — Unconditional diffusion (~40 min on T4)
python -m molgen.training.train_diffusion --phase 2

# Phase 3 — Conditional diffusion with CFG
python -m molgen.training.train_diffusion --phase 3 \
    --resume checkpoints/diffusion_phase2_best.pt

# Fast local test (CPU, ~2 min)
python -m molgen.training.train_encoder --max-mols 1000
```

Checkpoints save to `checkpoints/`. Resume interrupted training with `--resume`.

---

## Running

```bash
# Terminal 1 — API server
conda activate molgen
uvicorn molgen.api.app:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open `http://localhost:3000`.

Set property sliders → click Generate → wait ~30s → molecule gallery appears.

---

## Tests

```bash
pytest tests/ -v --ignore=tests/test_dataset.py
```

110 tests across 7 modules. `test_dataset.py` excluded from default run as it downloads QM9 (~80MB).

---

## Key Concepts

**Why graphs?** A molecule's properties depend on its structure — which atom connects to which. Treating SMILES as a sequence loses this structural information. Graph Neural Networks operate directly on the molecular graph.

**Why diffusion?** Diffusion models learn by reversing a noise-adding process. For molecules: node features (atom types, charges) are corrupted with Gaussian noise; the adjacency matrix (bonds) is corrupted with discrete bit-flips. The model learns to denoise both simultaneously.

**Classifier-free guidance:** During training, 20% of steps use a null conditioning vector. At inference, we run the denoiser twice (conditional + unconditional) and interpolate: `ε = (1+w)·ε_cond - w·ε_uncond`. Higher `w` = stronger property conditioning, lower diversity.

---

## Limitations

- **QM9 is small** — max 9 heavy atoms. Real drugs have 20–50+. Scaling to GEOM-Drugs requires attention mechanisms and more compute.
- **Validity ~50%** without post-processing — adjacency decoding from continuous features is approximate. State-of-the-art models achieve 95%+.
- **Property conditioning is approximate** — generated molecules trend toward targets, not exactly hit them. Hard guarantees require RL-guided generation.
- **No 3D geometry** — we generate 2D topology only. Real drug design needs 3D conformations (EDM, GeoDiff handle this).

---

## References

- Gilmer et al. 2017 — [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- Ho et al. 2020 — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Austin et al. 2021 — [Structured Denoising Diffusion in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
- Ho & Salimans 2022 — [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- Jo et al. 2022 — [GDSS: Score-based Generative Modeling of Graphs](https://arxiv.org/abs/2202.02514)

---

## Author

Harsh 
[GitHub: Harsh-617](https://github.com/Harsh-617)