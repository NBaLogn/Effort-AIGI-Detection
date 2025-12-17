This repository implements "Effort" — an SVD-based residual modeling approach for generalizable
AIGI / deepfake detection built on top of DeepfakeBench and an alternative benchmark (UniversalFakeDetect).

**Quick Purpose**: Help AI coding agents be productive by summarizing architecture, workflows,
conventions, and concrete commands used by contributors.

**Big Picture**
- **Architecture**: Two main stacks coexist:
  - DeepfakeBench-based training/evaluation: code under `DeepfakeBench/` (preprocessing, training, detectors).
  - Alternative reproduction code: `UniversalFakeDetect_Benchmark/` (separate training/eval pathway).
- **Core idea**: the `Effort` detector plugs into ViT/CLIP-style models by decomposing weights via SVD
  and freezing a low-rank main component while training residual components. See:
  - [DeepfakeBench/training/detectors/effort_detector.py](DeepfakeBench/training/detectors/effort_detector.py) (implementation)
  - The high-level method snippet in `README.md`.

**Where to look first**
- Project overview: [README.md](README.md)
- Training & demo: [DeepfakeBench/training/demo.py](DeepfakeBench/training/demo.py)
- Training entrypoint/config: [DeepfakeBench/training/train.py](DeepfakeBench/training/train.py) and detector config files under
  `DeepfakeBench/training/config/detector/` (e.g. `effort.yaml`).
- Preprocessing & dataset manifests: `DeepfakeBench/preprocessing/` and `DeepfakeBench/preprocessing/dataset_json/`.
- Face extraction & utilities: [face_detection_filter_retinaface.py](face_detection_filter_retinaface.py), [face_detection_filter.py](face_detection_filter.py), and [validate_retinaface.py](validate_retinaface.py).

**Developer workflows / concrete commands**
- Install dependencies: run the repository `install.sh` from project root: `sh install.sh`.
- Demo (inference) using a released checkpoint:
  ```bash
  cd DeepfakeBench/
  python3 training/demo.py --detector_config training/config/detector/effort.yaml \
    --weights ./training/weights/{CKPT}.pth --image {IMAGE_PATH_OR_FOLDER}
  ```
  If processing faces add the landmark model: `--landmark_model ./preprocessing/shape_predictor_81_face_landmarks.dat`.
- Training (single GPU):
  ```bash
  python3 training/train.py --detector_path training/config/detector/effort.yaml \
    --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2
  ```
- Training (multi-GPU / distributed):
  ```bash
  python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py \
    --detector_path training/config/detector/effort.yaml --train_dataset FaceForensics++ \
    --test_dataset Celeb-DF-v2 --ddp
  ```
- Testing/evaluation example:
  ```bash
  python3 training/test.py --detector_path training/config/detector/effort.yaml \
    --test_dataset simswap_ff blendface_ff uniface_ff fomm_ff deepfacelab \
    --weights_path ./training/weights/{CKPT}.pth
  ```

- Automation policy: when an agent or automation runs repository scripts, **always** use `uv add` or `uv run` (see `effort.sh`). Do **not** invoke `python` directly. If `uv` is missing, the agent must raise an explicit error or warning and halt — do not implement a fallback to `python`.

    - Common helper script: `effort.sh` contains curated `uv run` commands used day-to-day (finetune, evaluate, perf runs).
      - The project maintainer uses `python3.13` and `uv run` as a lightweight wrapper for running scripts; replicate that pattern locally.
      - Example (from `effort.sh`, evaluation command):
        ```bash
        uv run DeepfakeBench/training/evaluate_finetune.py \
          --detector_config DeepfakeBench/training/config/detector/effort_finetune.yaml \
          --weights ./DeepfakeBench/training/logs/.../ckpt_best.pth \
          --test_dataset /path/to/val --output_dir evaluation_results
        ```

**Project-specific conventions & patterns**
- DeepfakeBench is used as the canonical dataset/bench pipeline — prefer its dataset JSON manifests and preprocessing code
  rather than ad-hoc scripts.
- Detector plumbing: detectors are registered/loaded under `DeepfakeBench/training/detectors/` — new detectors should follow the
  same constructor/signature pattern used by existing detectors (see `effort_detector.py`).
- Weight/ckpt layout: trained checkpoints are expected in `DeepfakeBench/training/weights/` and referenced by CLI `--weights`/`--weights_path`.
- Dataset manifests: dataset lists are JSON files under `DeepfakeBench/preprocessing/dataset_json/` and the project expects those structures.

**Integration points & notable dependencies**
- CLIP integration: uses CLIP models (`clip` dependency) as a backbone for some detectors.
- InsightFace RetinaFace: local utilities use `insightface` for face detection (`face_detection_filter_retinaface.py`) and support `mps` auto-detection on macOS.
- Dlib landmarks: some face-processing code expects a `shape_predictor_81_face_landmarks.dat` file for face cropping.
- PyTorch DDP: training supports distributed launch via `torch.distributed.launch` with a `--ddp` flag.

**Gotchas & notes discovered in repo**
- The project targets **Python 3.13 only** (see `pyproject.toml`). Use `python3.13` as the interpreter for runs, virtualenvs, and CI jobs.
- The codebase relies heavily on external checkpoints and preprocessed datasets. Without those artifacts, training/evaluation won't reproduce reported results.

**How to propose code changes / where to run tests**
- Local quick checks: run example scripts in `DeepfakeBench/` for small inference jobs.
- For larger changes, run training with a small dataset subset or disabled heavy augmentations to validate end-to-end behavior.

If any of the above commands or file locations look out-of-date, tell me which part to update and I will refine this instruction file.

**Troubleshooting (quick fixes)**
- **Python / env**: This project requires `python3.13`. Verify and create an environment:
  ```bash
  python3.13 --version
  python3.13 -m venv .venv
  source .venv/bin/activate
  python3.13 -m pip install -U pip
  sh install.sh
  ```

- **`uv` wrapper requirement**: The maintainer uses `uv run`/`uv add` from `effort.sh` as the standard execution wrapper. Agents must use `uv` and must not fall back to `python` invocations. If `uv` is not installed or not available in the environment, the agent should emit a clear error/warning and stop rather than attempting to run `python` directly.

- **InsightFace / RetinaFace issues (common)**
  - Check MPS availability on macOS:
    ```bash
    python3.13 -c "import torch; print(torch.backends.mps.is_available())"
    ```
  - If RetinaFace model fails to load or prepare, force CPU by passing `--device cpu` to `face_detection_filter_retinaface.py` or use the script's device mapping.
  - Verify `insightface` is installed and compatible (pyproject lists `insightface>=0.7.3`):
    ```bash
    python3.13 -c "import insightface; print(insightface.__version__)"
    ```
  - If a model name is not found, list available models:
    ```bash
    python3.13 -c "import insightface; print(insightface.model_zoo.list_models())"
    ```

- **Dlib landmarks missing**: If face cropping/inference complains about missing landmarks, place or download `shape_predictor_81_face_landmarks.dat` and pass `--landmark_model PATH` to demo/evaluation scripts. Check `DeepfakeBench/preprocessing/` for expected locations.

- **Missing checkpoints / weights**: Ensure weights are available under `DeepfakeBench/training/weights/` or pass full path to `--weights`/`--weights_path` when running `demo.py`, `test.py`, or `evaluate_finetune.py`.

- **Distributed training (DDP) failures**:
  - Use `python3.13 -m torch.distributed.launch --nproc_per_node=N ...` and ensure CUDA/MPS availability for the target devices.
  - When debugging networking issues set `MASTER_ADDR` and `MASTER_PORT` env vars explicitly.

- **General debugging tips**:
  - Reproduce locally with a very small dataset subset before scaling to full experiments.
  - When in doubt, run the same script with `--device cpu` to separate device-specific errors from code errors.
