This repository implements **Effort**, the SVD-based residual modeling detector for generalizable AIGI/deepfake detection, built on DeepfakeBench and glued to a FastAPI backend plus a Next.js frontend.

**Quick Purpose**: Help AI coding agents stay productive by surfacing the key components (DeepfakeBench training/inference, backend API, frontend UI), the required conventions (`uv run`, `python3.13`), and the everyday commands and scripts people actually use.

**Big Picture**
- **DeepfakeBench/training**: canonical training/eval/inference pipeline; detectors live under `training/detectors/`, configs under `training/config/detector/` (see `effort.yaml` / `effort_finetune.yaml`), and the residual `Effort` model is detailed in `training/detectors/effort_detector.py`.
- **Backend**: `backend/server.py` instantiates the Effort detector, backs it with Grad-CAM reasoning, and exposes `/predict`, `/health` via FastAPI/uvicorn while reusing DeepfakeBench inference utilities (device manager, face alignment).
- **Frontend**: `frontend/` is a Next.js 16 + React 19 SPA that calls the backend `/predict` endpoint; the UI and assets live entirely inside this folder and use `npm`/Next scripts.

**Where to look first**
- `README.md` for the quick start, uv-based commands, and the API overview.
- `DeepfakeBench/training/train.py`, `finetune.py`, `evaluate_finetune.py`, `inference.py`, and `test.py` for the core ML training/eval workflows.
- `DeepfakeBench/training/config/detector/effort_finetune.yaml` for training/evaluation hyperparameters and `effort.yaml` for inference defaults.
- `backend/server.py` to understand model loading, Grad-CAM wiring, and how predictions / heatmaps reach the frontend.
- `frontend/` for the Next.js UI, its `package.json` scripts, and the hooks that fetch `/predict`.
- `finetune.sh`, `eval.sh`, `infer.sh` in the project root for the curated `uv run` commands people copy; each one hardcodes dataset/weight paths that need replacement in a local setup.

**Developer workflows / concrete commands**
- **Environment setup**
  ```bash
  python3.13 -m venv .venv
  source .venv/bin/activate
  python3.13 -m pip install -U pip
  uv sync
  ```
  `uv sync` installs everything listed in `pyproject.toml`, honoring the versions locked in `uv.lock`. The repo tracks `torch` 2.9.1+, `clip`, `dlib`, `insightface`, etc.
- **Finetuning (example)**
  ```bash
  uv run DeepfakeBench/training/finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --train_dataset /path/to/extracted/faces/train/… \
    --test_dataset /path/to/extracted/faces/val/… \
    --pretrained_weights /path/to/effort_clip_L14_trainOn_FaceForensic.pth
  ```
  Replace the dataset/weight paths with your local downloads; `finetune.sh` contains a concrete example that points at `/Volumes/Crucial/Large_Downloads/…`.
- **Evaluation**
  ```bash
  uv run DeepfakeBench/training/evaluate_finetune.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --weights /path/to/logs/<run>/test/avg/ckpt_best.pth \
    --test_dataset /path/to/dataset1 /path/to/dataset2 … \
    --output_dir evaluation_results
  ```
  Look inside `DeepfakeBench/training/logs/<run>/test/avg/ckpt_best.pth` for the latest checkpoint if you don’t have a standalone `.pth`.
- **Inference**
  ```bash
  uv run DeepfakeBench/training/inference.py \
    --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
    --landmark_model DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat \
    --weights /path/to/ckpt_best.pth \
    --image /path/to/images
  ```
- **Distributed training**
  ```bash
  python3.13 -m torch.distributed.launch --nproc_per_node=4 \
    DeepfakeBench/training/train.py \
    --detector_path DeepfakeBench/training/config/detector/effort.yaml \
    --train_dataset … --test_dataset … --ddp
  ```
  `uv run` is not required for `torch.distributed.launch`, but the script itself should still be run inside the Python 3.13 env.
- **Backend dev server**
  ```bash
  uv run backend/server.py
  ```
  It auto-loads the best checkpoint from `DeepfakeBench/training/logs` (falls back to the `WEIGHTS_PATH` constant or even random weights with a warning) and serves Grad-CAM overlays.
- **Frontend dev server**
  ```bash
  cd frontend
  npm install
  npm run dev
  ```
  Or `uv run npm run dev` if you prefer to stick to the `uv` wrapper. The Next app talks to `http://localhost:8000/predict`.
- **Tests**
  ```bash
  uv run DeepfakeBench/training/test.py \
    --detector_config DeepfakeBench/training/config/detector/effort.yaml \
    --test_dataset /path/to/dataset \
    --weights /path/to/ckpt_best.pth
  ```
  Look in `DeepfakeBench/training/test.py` for additional flags (batch size override, GPU/CPU selection, etc.).
- **Curated shell scripts**: `finetune.sh`, `eval.sh`, and `infer.sh` wrap the above commands with the actual datasets the maintainer uses. They are authoritative examples of argument order, but edit the dataset/weights paths before running on a new machine.

**Automation policy**
- Always run repo scripts via `uv run` (or `uv add` for installing new packages). The maintainer uses `python3.13` + `uv` as the official execution environment; automations that invoke `python` directly violate policy. If the agent notices `uv` is missing, it must raise a blocker rather than falling back to raw `python`.
- `uv.lock` frames the dependency tree. If you add/upgrade packages, run `uv add <pkg>` and commit the updated lockfile.

**Project-specific conventions & patterns**
- Configs live under `DeepfakeBench/training/config/detector/`: copy the frozen structure (optimize, data augmentation, SVD residual toggles) when adding a new detector.
- Detectors are registered in `DeepfakeBench/training/detectors/`. Follow the existing constructors (see `effort_detector.py`) to keep initialization consistent.
- Dataset manifests live under `DeepfakeBench/preprocessing/dataset_json/`, and preprocessing helpers share common artifacts under `DeepfakeBench/preprocessing/`.
- Logs and checkpoints live under `DeepfakeBench/training/logs/<run>/test/avg/ckpt_best.pth` (pruned weights are not checked in). Keep an eye on these paths when loading weights in automation scripts or debugging `backend/server.py`.
- The backend reuses `DeepfakeBench/training/inference.DeviceManager`, `FaceAlignment`, and the Grad-CAM helpers under `backend/gradcam_utils.py`. This keeps the API aligned with the training code.

**Integration points & notable dependencies**
- CLIP-backed Vision Transformer (the `Effort` detector wraps the CLIP encoder; see `DeepfakeBench/training/detectors/effort_detector.py`).
- Grad-CAM (`pytorch_grad_cam`) to produce explanations shown in the frontend.
- Torch distributed (`torch.distributed.launch`, `--ddp`) for scaling training.
- FastAPI/uvicorn to serve predictions (`backend/server.py` and `/health`, `/predict` endpoints).
- Next.js 16 + React 19 for the frontend UI; `frontend/package.json` lists the scripts.
- Dlib + InsightFace + the 81-point shape predictor (`DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat`) for face alignment and landmarks.
- External checkpoints and extracted faces are not part of the repo; expect to download them separately.

**Gotchas & notes discovered in repo**
- **Python 3.13 only** (see `pyproject.toml`). Always activate a `python3.13` venv before running scripts or `uv`.
- **`uv run` required**: automation + CI wrap every command with `uv run` (or `uv add`). Scripts must not call `python` directly.
- **Checkpoints reside under finetuned_weights** (`DeepfakeBench/training/weights/finetuned`). `backend/server.py` searches for `ckpt_best.pth`, so adding new training runs should produce that file in the logs tree.
- **Datasets & weights are large**: they are not committed. Mirror your own `DeepfakeBench/facedata/…` structure or point the scripts at wherever you store the extracted faces.
- **Landmark model**: ingestion scripts and the backend expect `DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat`. Download it from the Dlib repo and place it there, or pass `--landmark_model` every time.
- **finetune/eval/infer scripts reference `/Volumes/Crucial/Large_Downloads…`**; update those absolute paths before running on a new machine.

**How to propose code changes / where to run tests**
- For ML behavior, rerun `uv run DeepfakeBench/training/evaluate_finetune.py` to validate inference/backbone changes before pushing.
- Backend regressions: run `uv run backend/server.py` locally and call `/predict` with a known image (the frontend UI, cURL, or `inference.py` can help).
- Frontend/UI changes: run `cd frontend && npm run lint` / `npm run dev` (or `uv run npm run dev`) to verify TypeScript/Next behavior.

**Troubleshooting (quick fixes)**
- **Python / env**: Ensure `python3.13` is installed.
  ```bash
  python3.13 --version
  python3.13 -m venv .venv
  source .venv/bin/activate
  python3.13 -m pip install -U pip
  uv sync
  ```
- **`uv` wrapper requirement**: If `uv` is missing, install it (`pip install uv`) and then run scripts via `uv run ...`. Do not run `python ...` by hand.
- **InsightFace / RetinaFace / dlib**:
  ```bash
  python3.13 -c "import insightface; print(insightface.__version__)"
  python3.13 -c "import dlib; print(dlib.__version__)"
  ```
  Force CPU by passing `--device cpu` to the dataset filtering or inference scripts if GPU initialization fails.
- **Missing checkpoints / weights**: After training or evaluation, look under `DeepfakeBench/training/logs/<run>/test/avg/ckpt_best.pth`. Backend server will automatically search for the most recent `ckpt_best.pth` in the logs directory.
- **Landmarks missing**: Download `shape_predictor_81_face_landmarks.dat` and point scripts/servers at it (`--landmark_model`).
- **Distributed training issues**: Set `MASTER_ADDR` and `MASTER_PORT` if you hit networking errors, and use `python3.13 -m torch.distributed.launch --nproc_per_node=N ...`.
- **General debugging**: Reproduce failures with small subsets / `--device cpu` before scaling; use `training/logs` artifacts for metrics, and inspect `evaluation_results/` or `inference_results/` output JSON for regression signals.
