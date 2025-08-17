import os
import math
import json
import traceback
import multiprocessing
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import numpy as np

# Optional deps (loaded lazily where used)
# onnxruntime is required only if you pass model paths
try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # We'll handle at runtime

# --- Your project imports ---
from recognizeapp.utils import (
    Dataset,
    dedupe_images,
    generate_report,
    get_chunks,
)

NO_FACE_DETECTED = "NO FACE DETECTED"

MODEL_BACKEND_COMPATIBILITY = {
    "VGG-Face": ["opencv", "mtcnn", "retinaface", "ssd"],
    "Facenet": ["mtcnn", "retinaface", "mediapipe"],
    "Facenet512": ["mtcnn", "retinaface", "mediapipe"],
    "OpenFace": ["opencv", "mtcnn"],
    "DeepFace": ["mtcnn", "ssd", "dlib"],
    "DeepID": ["mtcnn", "opencv"],
    "Dlib": ["dlib", "mediapipe"],
    "ArcFace": ["mtcnn", "retinaface"],
    "SFace": ["mtcnn", "retinaface", "mediapipe"],
    "GhostFaceNet": ["mtcnn", "retinaface", "centerface", "yolov8"],
}

DEFAULT_BACKENDS = {
    "VGG-Face": "retinaface",
    "Facenet": "mtcnn",
    "Facenet512": "mtcnn",
    "OpenFace": "mtcnn",
    "DeepFace": "retinaface",
    "DeepID": "mtcnn",
    "Dlib": "dlib",
    "ArcFace": "retinaface",
    "SFace": "retinaface",
    "GhostFaceNet": "mtcnn",
}

# ===============================
# Anti-spoof & Deepfake detectors
# ===============================

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

@dataclass
class OnnxBinaryClassifier:
    """
    Generic ONNX binary classifier wrapper.
    Assumes output logits for 2 classes. You can adapt preprocess dims & normalization.
    """
    model_path: Optional[str]
    input_size: Tuple[int, int] = (128, 128)
    input_name: Optional[str] = None
    output_name: Optional[str] = None
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float]  = (0.5, 0.5, 0.5)
    nchw: bool = True  # True if model expects NCHW; False for NHWC
    rgb: bool = True   # Convert BGR->RGB if True

    def __post_init__(self):
        self.enabled = False
        self.session = None
        if self.model_path and os.path.exists(self.model_path):
            if ort is None:
                print(f"[WARN] onnxruntime is not installed; detector disabled: {self.model_path}")
                return
            try:
                self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
                if self.input_name is None:
                    self.input_name = self.session.get_inputs()[0].name
                if self.output_name is None:
                    self.output_name = self.session.get_outputs()[0].name
                self.enabled = True
                print(f"[OK] Loaded ONNX model: {self.model_path}")
            except Exception as e:
                print(f"[WARN] Could not load ONNX model: {self.model_path} | {e}")
        else:
            if self.model_path:
                print(f"[WARN] ONNX model not found: {self.model_path}. Detector disabled.")

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image in preprocess")
        # detect face crop outside, pass face chip here ideally; fallback is full frame
        h, w = self.input_size
        x = img_bgr.copy()
        if self.rgb:
            x = x[:, :, ::-1]  # BGR->RGB
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
        x = x.astype("float32") / 255.0
        x = (x - self.mean) / self.std
        if self.nchw:
            x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.expand_dims(x, axis=0)
        return x

    def predict_proba(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        try:
            x = self.preprocess(img_bgr)
            out = self.session.run([self.output_name], {self.input_name: x})[0]
            # out shape: (1,2) logits
            probs = _softmax(out)[0]
            return probs  # [p_class0, p_class1]
        except Exception as e:
            print(f"[WARN] ONNX inference failed: {e}")
            return None


# Anti-spoof: class index convention -> [spoof, live]  (so prob_live = probs[1])
class AntiSpoofONNX(OnnxBinaryClassifier):
    pass

# Deepfake: class index convention -> [fake, real]  (so prob_fake = probs[0])
class DeepfakeONNX(OnnxBinaryClassifier):
    pass


# ===============================
# Quality gates (face blur & size)
# ===============================
import cv2  # after typing to avoid circular static checkers

def variance_of_laplacian(gray: np.ndarray) -> float:
    # Higher -> sharper
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

@dataclass
class QualityConfig:
    min_face_size: int = 100  # minimum of width/height for face crop
    min_blur: float = 50.0    # Laplacian variance threshold

def face_quality_ok(face_chip_bgr: np.ndarray, qc: QualityConfig) -> Tuple[bool, Dict[str, Any]]:
    if face_chip_bgr is None or face_chip_bgr.size == 0:
        return False, {"reason": "empty_face_chip"}
    h, w = face_chip_bgr.shape[:2]
    if min(h, w) < qc.min_face_size:
        return False, {"reason": "small_face", "h": h, "w": w}
    gray = cv2.cvtColor(face_chip_bgr, cv2.COLOR_BGR2GRAY)
    blur = variance_of_laplacian(gray)
    if blur < qc.min_blur:
        return False, {"reason": "blurry", "lap_var": blur}
    return True, {"lap_var": blur, "h": h, "w": w}


# ===============================
# Face extraction helper
# ===============================
def extract_primary_face_bgr(image_path: str, detector_backend: str) -> Optional[np.ndarray]:
    """
    Use DeepFace.extract_faces to get the largest/aligned face as BGR for detectors/quality.
    Falls back to full image if detection fails.
    """
    try:
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )
        if not faces:
            return None
        # faces elements: {'face': np.ndarray in RGB, 'facial_area': {...}, 'confidence': float, ...}
        # pick largest by area
        best = max(faces, key=lambda f: f.get("facial_area", {}).get("w", 0) * f.get("facial_area", {}).get("h", 0))
        rgb = (best.get("face") * 255.0).astype("uint8")  # extract_faces returns float [0,1]
        bgr = rgb[:, :, ::-1]
        return bgr
    except Exception:
        # Fall back: load full image BGR
        try:
            bgr = cv2.imread(image_path)
            return bgr
        except Exception:
            return None


# ===============================
# Unified checks
# ===============================
@dataclass
class DetectorConfig:
    antispoof_model: Optional[str] = None
    deepfake_model: Optional[str] = None
    liveness_threshold: float = 0.80  # prob_live >= this -> pass
    deepfake_threshold: float = 0.70  # prob_fake < this -> pass (lower is better)
    quality: QualityConfig = QualityConfig()

def initialize_detectors(cfg: DetectorConfig) -> Tuple[AntiSpoofONNX, DeepfakeONNX]:
    anti = AntiSpoofONNX(
        model_path=cfg.antispoof_model,
        input_size=(128, 128),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        nchw=True,
        rgb=True,
    )
    df = DeepfakeONNX(
        model_path=cfg.deepfake_model,
        input_size=(224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        nchw=True,
        rgb=True,
    )
    return anti, df

def check_liveness_and_deepfake(
    image_path: str,
    detector_backend: str,
    anti: AntiSpoofONNX,
    df: DeepfakeONNX,
    cfg: DetectorConfig,
) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        "pass": bool,
        "live_pass": bool,
        "deepfake_pass": bool,
        "prob_live": float or None,
        "prob_fake": float or None,
        "reason": str or None,
        "quality": {...}
      }
    Decision: pass only if quality ok AND live_pass AND deepfake_pass
    """
    result = {
        "pass": False, "live_pass": True, "deepfake_pass": True,
        "prob_live": None, "prob_fake": None, "reason": None, "quality": {}
    }

    face_chip = extract_primary_face_bgr(image_path, detector_backend)
    if face_chip is None:
        result.update({"pass": False, "live_pass": False, "deepfake_pass": False, "reason": NO_FACE_DETECTED})
        return result

    # Quality
    ok, qinfo = face_quality_ok(face_chip, cfg.quality)
    result["quality"] = qinfo
    if not ok:
        result.update({"pass": False, "live_pass": False, "deepfake_pass": False, "reason": f"quality:{qinfo.get('reason')}"})
        return result

    # Anti-spoof (liveness)
    if anti.enabled:
        probs = anti.predict_proba(face_chip)  # [p_spoof, p_live]
        if probs is None:
            result.update({"live_pass": True, "prob_live": None})
        else:
            p_live = float(probs[1])
            result["prob_live"] = p_live
            if p_live < cfg.liveness_threshold:
                result.update({"live_pass": False, "reason": f"liveness_low:{p_live:.3f}"})
    else:
        # Detector disabled -> treat as pass but mark reason
        result["reason"] = (result["reason"] or "") + "|liveness_disabled"

    # Deepfake
    if df.enabled:
        probs = df.predict_proba(face_chip)  # [p_fake, p_real]
        if probs is None:
            result.update({"deepfake_pass": True, "prob_fake": None})
        else:
            p_fake = float(probs[0])
            result["prob_fake"] = p_fake
            if p_fake >= cfg.deepfake_threshold:
                # fail if fake probability too high
                prev = result.get("reason") or ""
                result.update({"deepfake_pass": False, "reason": (prev + f"|deepfake_high:{p_fake:.3f}").strip("|")})
    else:
        result["reason"] = (result["reason"] or "") + "|deepfake_disabled"

    result["pass"] = result["live_pass"] and result["deepfake_pass"]
    return result


# ===============================
# Original pipeline (with upgrades)
# ===============================
def validate_model_backend(model: str, backend: str) -> bool:
    compatible_backends = MODEL_BACKEND_COMPATIBILITY.get(model, [])
    return backend in compatible_backends

def suggest_backend(model: str) -> str:
    return DEFAULT_BACKENDS.get(model, "mtcnn")

def validate_and_adjust_options(options: Dict[str, Any]) -> Dict[str, Any]:
    model = options["model_name"]
    backend = options["detector_backend"]
    if not validate_model_backend(model, backend):
        click.echo(
            f"Error: Incompatible model-backend pair: Model '{model}' does not support Backend '{backend}'.",
            err=True,
        )
        raise click.Abort()
    return options


# --- UPDATED encode_faces: quality + liveness + deepfake gating ---
def encode_faces(files_chunk, options, pre_encodings, detectors_cfg: DetectorConfig):
    from deepface import DeepFace  # Import here for multiprocessing compatibility

    anti, df = initialize_detectors(detectors_cfg)
    encodings = {}
    added, existing = 0, 0
    skip_stats = {"noface": 0, "quality": 0, "liveness": 0, "deepfake": 0, "errors": 0}

    for img_path in files_chunk:
        try:
            # Quick skip if already encoded
            if img_path in pre_encodings:
                encodings[img_path] = pre_encodings[img_path]
                existing += 1
                continue

            # --- Liveness + Deepfake + Quality ---
            check = check_liveness_and_deepfake(
                img_path,
                options.get("detector_backend", "retinaface"),
                anti, df, detectors_cfg
            )

            if check["reason"] == NO_FACE_DETECTED:
                print(f"[SKIP] {img_path} -> {NO_FACE_DETECTED}")
                skip_stats["noface"] += 1
                continue

            if "quality:" in (check.get("reason") or ""):
                print(f"[SKIP] {img_path} -> poor quality ({check['quality']})")
                skip_stats["quality"] += 1
                continue

            if not check["live_pass"]:
                print(f"[SKIP] {img_path} -> liveness fail (p_live={check.get('prob_live')})")
                skip_stats["liveness"] += 1
                continue

            if not check["deepfake_pass"]:
                print(f"[SKIP] {img_path} -> deepfake fail (p_fake={check.get('prob_fake')})")
                skip_stats["deepfake"] += 1
                continue

            # --- Embedding (DeepFace) ---
            emb = DeepFace.represent(
                img_path,
                model_name=options.get("model_name", "ArcFace"),
                detector_backend=options.get("detector_backend", "retinaface"),
                enforce_detection=False,  # we've already checked face
            )[0]["embedding"]

            encodings[img_path] = emb
            added += 1
            print(f"[ENCODED] {img_path}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            skip_stats["errors"] += 1
            print(f"[FAILED] {img_path} | {e}")
            traceback.print_exc()

    # For visibility in metrics (printed later)
    encodings["_skip_stats"] = skip_stats  # stored transiently; removed before saving
    return encodings, added, existing


def process_files(config, num_processes, pre_findings):
    ds = Dataset(config)
    start_time = datetime.now()
    tracemalloc.start()

    total_files = ds.get_files()
    chunks = get_chunks(total_files, num_processes)

    # Build detector config from options
    opts = ds.get_encoding_config() or {}
    det_cfg = DetectorConfig(
        antispoof_model=opts.get("antispoof_model"),
        deepfake_model=opts.get("deepfake_model"),
        liveness_threshold=float(opts.get("liveness_threshold", 0.80)),
        deepfake_threshold=float(opts.get("deepfake_threshold", 0.70)),
        quality=QualityConfig(
            min_face_size=int(opts.get("min_face_size", 100)),
            min_blur=float(opts.get("min_blur", 50.0)),
        ),
    )

    # Encode faces in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_enc = pool.map(
                partial(
                    encode_faces,
                    options=ds.get_encoding_config(),
                    pre_encodings=ds.get_encoding(),
                    detectors_cfg=det_cfg,
                ),
                chunks,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            raise

    encodings = {}
    added = existing = 0
    skip_aggr = {"noface": 0, "quality": 0, "liveness": 0, "deepfake": 0, "errors": 0}
    for d, a, e in partial_enc:
        # Pull and drop skip stats
        sk = d.pop("_skip_stats", None)
        if sk:
            for k in skip_aggr:
                skip_aggr[k] += sk.get(k, 0)
        added += a
        existing += e
        encodings.update(d)
    encoding_time = datetime.now()

    # Dedup in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_find = pool.map(
                partial(
                    dedupe_images,
                    options=ds.get_dedupe_config(),
                    encodings=encodings,
                    pre_findings=pre_findings,
                ),
                chunks,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            raise

    findings = {}
    for d in partial_find:
        findings.update(d)
    end_time = datetime.now()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")
    ram_mb = 0.0
    if top_stats:
        ram_mb = top_stats[0].size / 1024 / 1024

    metrics = {
        "Encoding Time": str(encoding_time - start_time),
        "Deduplication Time": str(end_time - encoding_time),
        "Total Time": str(end_time - start_time),
        "RAM Mb": f"{ram_mb:.2f}",
        "Processes": num_processes,
        "------": "--------",
        "Total Files": len(total_files),
        "New Images": added,
        "Existing Encodings": existing,
        "Database": len(encodings),
        "Findings": len(findings),
        "Skip (No Face)": skip_aggr["noface"],
        "Skip (Quality)": skip_aggr["quality"],
        "Skip (Liveness)": skip_aggr["liveness"],
        "Skip (Deepfake)": skip_aggr["deepfake"],
        "Skip (Errors)": skip_aggr["errors"],
        "==": "==",
        **config["options"],
    }
    return encodings, findings, metrics


# ===============================
# CLI
# ===============================
@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--reset", is_flag=True, help="Reset the dataset (clear encodings and findings).")
@click.option("--queue", is_flag=True, help="Queue the task for background processing.")
@click.option("--report", is_flag=True, help="Generate a report after processing.")
@click.option(
    "-p",
    "--processes",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of processes to use for parallel execution.",
)
@click.option(
    "--model-name",
    type=click.Choice(
        [
            "VGG-Face",
            "Facenet",
            "Facenet512",
            "OpenFace",
            "DeepFace",
            "DeepID",
            "Dlib",
            "ArcFace",
            "SFace",
            "GhostFaceNet",
        ]
    ),
    default="ArcFace",
    help="Name of the face recognition model to use.",
)
@click.option(
    "--detector-backend",
    type=click.Choice(
        [
            "opencv",
            "retinaface",
            "mtcnn",
            "ssd",
            "dlib",
            "mediapipe",
            "centerface",
            "skip",
        ]
    ),
    default="retinaface",
    help="Face detection backend to use.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output for debugging.")
# Anti-spoof / deepfake specific
@click.option("--antispoof-model", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to anti-spoof (liveness) ONNX model (binary live/spoof).")
@click.option("--deepfake-model", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to deepfake ONNX model (binary fake/real).")
@click.option("--liveness-threshold", type=float, default=0.80,
              help="Threshold for prob_live to pass liveness (0..1).")
@click.option("--deepfake-threshold", type=float, default=0.70,
              help="Fail if prob_fake >= threshold (0..1).")
@click.option("--min-face-size", type=int, default=100, help="Minimum face chip min(width,height) in pixels.")
@click.option("--min-blur", type=float, default=50.0, help="Minimum Laplacian variance to consider image sharp.")
def cli(path, processes, reset, queue, report, verbose, **depface_options):
    """
    Process a folder of images to detect & deduplicate faces with liveness + deepfake checks.
    """
    depface_options = validate_and_adjust_options(depface_options)

    # Merge detector params into encoding options so workers can see them
    depface_options.update({
        "antispoof_model": depface_options.get("antispoof_model"),
        "deepfake_model": depface_options.get("deepfake_model"),
        "liveness_threshold": depface_options.get("liveness_threshold", 0.80),
        "deepfake_threshold": depface_options.get("deepfake_threshold", 0.70),
        "min_face_size": depface_options.get("min_face_size", 100),
        "min_blur": depface_options.get("min_blur", 50.0),
    })

    patterns = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [str(f.absolute()) for f in Path(path).iterdir()
             if f.is_file() and f.suffix.lower() in patterns]

    if not files:
        click.echo("No image files found in the provided directory. Exiting.", err=True)
        return

    processes = min(len(files), processes)

    if verbose:
        click.echo(json.dumps({
            "Model": depface_options["model_name"],
            "Backend": depface_options["detector_backend"],
            "Processes": processes,
            "LivenessThreshold": depface_options["liveness_threshold"],
            "DeepfakeThreshold": depface_options["deepfake_threshold"],
            "AntiSpoofModel": depface_options.get("antispoof_model"),
            "DeepfakeModel": depface_options.get("deepfake_model"),
            "MinFaceSize": depface_options.get("min_face_size"),
            "MinBlur": depface_options.get("min_blur"),
        }, indent=2))

    ds = Dataset({"path": path, "options": depface_options})
    report_file = Path(path) / f"_report_{depface_options['model_name']}_{depface_options['detector_backend']}.html"

    click.echo(f"Processing {len(files)} files in {path}")

    if reset:
        ds.reset()
    else:
        # safe unlink for findings db
        try:
            ds.storage(ds.findings_db_name).unlink(missing_ok=True)  # Python 3.8+: use try/except if older
        except TypeError:
            # For older Python: emulate missing_ok
            p = ds.storage(ds.findings_db_name)
            if p.exists():
                p.unlink()

    config = {"options": {**depface_options}, "path": path}

    if queue:
        from recognizeapp.tasks import process_dataset
        process_dataset.delay(config)
    else:
        click.echo(f"Spawn {processes} processes")
        pre_encodings = ds.get_encoding()
        click.echo(f"Found {len(pre_encodings)} existing encodings")
        pre_findings = ds.get_findings()

        encodings, findings, metrics = process_files(config, processes, pre_findings)

        # Remove transient skip stats if present
        if "_skip_stats" in encodings:
            encodings.pop("_skip_stats", None)

        for k, v in metrics.items():
            click.echo(f"{k:<25}: {v}")

        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)

        if report:
            try:
                generate_report(ds.path, ds.get_findings(), ds.get_perf(), report_file)
            except Exception as e:
                click.echo(f"[WARN] Failed to generate report: {e}", err=True)


if __name__ == "__main__":
    cli()
