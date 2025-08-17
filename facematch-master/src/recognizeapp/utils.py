import datetime
import json
import multiprocessing
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from deepface import DeepFace
from jinja2 import Template

NO_FACE_DETECTED = "NO_FACE_DETECTED"


# -----------------------------
# Dataset Class
# -----------------------------
class Dataset:
    """
    A class to manage dataset-related operations, such as encoding and deduplication.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = {**config}
        self.path: Path = Path(self.config.pop("path")).absolute()
        self.options: Dict[str, Any] = config["options"]
        self.storage: Callable[[Path], Path] = Path  # Default to Path

    def __str__(self):
        return f"<Dataset: {self.path.name}>"

    def _get_filename(self, prefix: str, suffix: str = ".json") -> Path:
        """Generate a file name based on the prefix and dataset options."""
        parts = [self.options["model_name"], self.options["detector_backend"]]
        extra = "_".join(parts)
        return self.storage(self.path) / f"_{prefix}_{extra}{suffix}"

    @cached_property
    def encoding_db_name(self) -> Path:
        return self._get_filename("encoding")

    @cached_property
    def findings_db_name(self) -> Path:
        return self._get_filename("findings")

    @cached_property
    def silenced_db_name(self) -> Path:
        return self._get_filename("silenced")

    @cached_property
    def runinfo_db_name(self) -> Path:
        return self._get_filename("perf")

    def reset(self) -> None:
        self.storage(self.encoding_db_name).unlink(missing_ok=True)
        self.storage(self.findings_db_name).unlink(missing_ok=True)
        self.storage(self.silenced_db_name).unlink(missing_ok=True)

    def get_encoding(self) -> Dict[Path, Union[str, List[float]]]:
        if self.storage(self.encoding_db_name).exists():
            return json.loads(self.storage(self.encoding_db_name).read_text())
        return {}

    def get_findings(self) -> Dict[Path, Any]:
        if self.storage(self.findings_db_name).exists():
            return json.loads(self.storage(self.findings_db_name).read_text())
        return {}

    def get_perf(self) -> Dict[Path, Any]:
        if self.storage(self.runinfo_db_name).exists():
            return json.loads(self.storage(self.runinfo_db_name).read_text())
        return {}

    def get_silenced(self) -> Dict[Path, Any]:
        if self.storage(self.silenced_db_name).exists():
            return json.loads(self.storage(self.silenced_db_name).read_text())
        return {}

    def get_files(self) -> List[str]:
        patterns = ("*.png", "*.jpg", "*.jpeg")
        return [str(f.absolute()) for f in self.storage(self.path).iterdir() if any(f.match(p) for p in patterns)]

    def update_findings(self, findings: Dict[str, Any]) -> Path:
        self.storage(self.findings_db_name).write_text(json.dumps(findings))
        return self.findings_db_name

    def update_encodings(self, encodings: Dict[str, Any]) -> Path:
        self.storage(self.encoding_db_name).write_text(json.dumps(encodings))
        return self.encoding_db_name

    def save_run_info(self, info: Dict[str, Any]) -> None:
        self.storage(self.runinfo_db_name).write_text(json.dumps(info))

    def get_encoding_config(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "model_name": self.options["model_name"],
            "detector_backend": self.options["detector_backend"],
        }

    def get_dedupe_config(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "model_name": self.options["model_name"],
            "detector_backend": self.options["detector_backend"],
        }


# -----------------------------
# Helper functions
# -----------------------------
def chop_microseconds(delta: datetime.timedelta) -> datetime.timedelta:
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def encode_faces(
    files: list[str], options=None, pre_encodings=None, progress=None
) -> tuple[dict[str, Union[str, list[float]]], int, int]:
    """Generate embeddings for images using DeepFace."""
    if not callable(progress):
        progress = lambda *a: True

    results = {}
    if pre_encodings:
        results.update(pre_encodings)
    added = existing = 0
    for n, file in enumerate(files):
        progress(n, file)
        if file in results:
            existing += 1
            continue
        try:
            result = DeepFace.represent(file, **(options or {}))
            if len(result) > 1:
                raise ValueError("More than one face detected")
            results[file] = result[0]["embedding"]
            added += 1
        except (TypeError, ValueError):
            results[file] = NO_FACE_DETECTED
    return results, added, existing


def get_chunks(elements: list[Any], max_len=multiprocessing.cpu_count()) -> list[list[Any]]:
    processes = min(len(elements), max_len)
    chunk_size = max(1, len(elements) // processes)
    return [elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)]


def dedupe_images(
    files: List[str],
    encodings: Dict[str, Union[str, List[float]]],
    options: Optional[Dict[str, Any]] = None,
    pre_findings: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, List[Union[str, float]]]:
    """Compare embeddings and find duplicates using DeepFace.verify."""
    if not callable(progress):
        progress = lambda *a: None
    findings: defaultdict = defaultdict(list)
    if pre_findings:
        findings.update(pre_findings)

    for n, file1 in enumerate(files):
        progress(n, file1)
        enc1 = encodings[file1]
        if enc1 == NO_FACE_DETECTED:
            findings[file1].append([NO_FACE_DETECTED, 99])
            continue
        for file2, enc2 in encodings.items():
            if file1 == file2 or enc2 == NO_FACE_DETECTED:
                continue
            if file2 in [x[0] for x in findings.get(file1, [])]:
                continue
            res = DeepFace.verify(enc1, enc2, **(options or {}))
            findings[file1].append([file2, res["distance"]])
    return findings


def generate_report(
    working_dir: Path,
    findings: Dict[str, List[Union[str, float]]],
    metrics: Dict[str, Any],
    report_file: Path,
    save_to_file: bool = True,
) -> None:
    """Generate HTML report of duplicate findings."""
    def _resolve(p: Union[Path, str]) -> Union[Path, str]:
        if p == NO_FACE_DETECTED:
            return NO_FACE_DETECTED
        return Path(p).absolute().relative_to(working_dir)

    template_path = Path(__file__).parent / "report.html"
    template = Template(template_path.read_text())

    results = []
    for img, duplicates in findings.items():
        for dup in duplicates:
            results.append([_resolve(img), _resolve(dup[0]), dup[1]])

    results = sorted(results, key=lambda x: x[2])
    rendered_content = template.render(metrics=metrics, findings=results)
    if save_to_file:
        report_file.write_text(rendered_content, encoding="utf-8")
        print(f"Report successfully saved to {report_file}")


def distance_to_similarity(distance: float) -> float:
    return 1 - distance


# -----------------------------
# Anti-spoofing (Liveness) Check
# -----------------------------
def check_liveness(image_path: str, model_path: str = "resources/anti_spoof_models") -> bool:
    """
    Perform liveness check using Silent-Face Anti-Spoofing.
    Returns True if the face is real/live, False if spoofed.
    """
    try:
        from silent_face import AntiSpoofPredict
        model = AntiSpoofPredict(model_path)
        image = cv2.imread(image_path)
        if image is None:
            return False
        prediction = model.predict(image)
        label = int(np.argmax(prediction))
        return label == 1  # 1 = live, 0 = spoof
    except Exception as e:
        print(f"[WARN] Liveness check failed: {e}")
        return False


# -----------------------------
# Deepfake Detector
# -----------------------------
class DeepFakeDetector(torch.nn.Module):
    """
    Pretrained CNN model for deepfake detection.
    Replace with a stronger model if needed.
    """

    def __init__(self, model_path="resources/deepfake_detector.pth"):
        super().__init__()
        try:
            self.model = torch.load(model_path, map_location="cpu")
            self.model.eval()
        except Exception as e:
            print(f"[WARN] Could not load deepfake model: {e}")
            self.model = None

    def predict(self, image_path: str) -> bool:
        """
        Predict if the image is deepfake.
        Returns True if fake, False if real.
        """
        if self.model is None:
            return True  # Fail-safe: treat as fake

        image = cv2.imread(image_path)
        if image is None:
            return True

        image = cv2.resize(image, (224, 224))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.model(image)
            prob = torch.softmax(output, dim=1)
            label = torch.argmax(prob).item()
        return label == 1  # assume 1 = deepfake, 0 = real
