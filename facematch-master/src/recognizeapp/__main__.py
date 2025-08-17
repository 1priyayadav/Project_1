import multiprocessing
import tracemalloc
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict
import click
import cv2  # <-- New for liveness detection

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

def check_liveness(image_path):
    """Stub liveness function: allow all images.
    Replace this logic with your real liveness model for production.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    # Insert your real liveness model logic here.
    return True

# --- Updated encode_faces with liveness detection integrated ---
def encode_faces(files_chunk, options, pre_encodings):
    from deepface import DeepFace  # Import here for multiprocessing compatibility
    encodings = {}
    added, existing = 0, 0
    for img_path in files_chunk:
        # ------ LIVENESS DETECTION ADDED ------
        if not check_liveness(img_path):
            print(f"[LIVENESS FAIL] {img_path}")
            continue  # Skip non-live/fake images
        try:
            # Sample: check if already encoded
            if img_path in pre_encodings:
                encodings[img_path] = pre_encodings[img_path]
                existing += 1
                continue
            # Replace with your face encoding logic as needed:
            emb = DeepFace.represent(
                img_path,
                model_name=options.get("model_name", "ArcFace"),
                detector_backend=options.get("detector_backend", "retinaface"),
            )[0]['embedding']
            encodings[img_path] = emb
            added += 1
            print(f"ENCODED: {img_path}")
        except Exception as e:
            print(f"FAILED TO ENCODE: {img_path} | Error: {e}")
    return encodings, added, existing

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

def process_files(config, num_processes, pre_findings):
    ds = Dataset(config)
    start_time = datetime.now()
    tracemalloc.start()
    total_files = ds.get_files()
    chunks = get_chunks(total_files, num_processes)

    # Encode faces in parallel (uses encode_faces WITH liveness detection)
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            partial_enc = pool.map(
                partial(
                    encode_faces,
                    options=ds.get_encoding_config(),
                    pre_encodings=ds.get_encoding(),
                ),
                chunks,
            )
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
    encodings = {}
    added = exiting = 0
    for d, a, e in partial_enc:
        added += a
        exiting += e
        encodings.update(d)
    encoding_time = datetime.now()

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
    findings = {}
    for d in partial_find:
        findings.update(d)
    end_time = datetime.now()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")
    metrics = {
        "Encoding Time": str(encoding_time - start_time),
        "Deduplication Time": str(end_time - encoding_time),
        "Total Time": str(end_time - start_time),
        "RAM Mb": str(top_stats[0].size / 1024 / 1024),
        "Processes": num_processes,
        "------": "--------",
        "Total Files": len(total_files),
        "New Images": added,
        "Database": len(encodings),
        "Findings": len(findings),
        "==": "==",
        **config["options"],
    }
    return encodings, findings, metrics

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
    default="VGG-Face",
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
def cli(path, processes, reset, queue, report, verbose, **depface_options):
    """
    CLI to process a folder of images to detect and deduplicate faces.
    :param path: Path to the folder containing images.
    :param processes: Number of processes to use.
    :param reset: Reset the dataset (clear encodings and findings).
    :param queue: Queue the task for background processing.
    :param report: Generate a report after processing.
    """
    depface_options = validate_and_adjust_options(depface_options)
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files = [str(f.absolute()) for f in Path(path).iterdir() if any(f.match(p) for p in patterns)]
    if not files:
        click.echo("No image files found in the provided directory. Exiting.", err=True)
        return
    processes = min(len(files), processes)
    if verbose:
        click.echo(f"Model: {depface_options['model_name']}")
        click.echo(f"Backend: {depface_options['detector_backend']}")
        click.echo(f"Process: {processes}")
    ds = Dataset({"path": path, "options": depface_options})
    report_file = Path(path) / f"_report_{depface_options['model_name']}_{depface_options['detector_backend']}.html"
    click.echo(f"Processing {len(files)} files in {path}")
    if reset:
        ds.reset()
    else:
        ds.storage(ds.findings_db_name).unlink(True)
    config = {"options": {**depface_options}, "path": path}
    if queue:
        from recognizeapp.tasks import process_dataset
        config = {"options": {**depface_options}, "path": path}
        process_dataset.delay(config)
    else:
        click.echo(f"Spawn {processes} processes")
        pre_encodings = ds.get_encoding()
        click.echo(f"Found {len(pre_encodings)} existing encodings")
        pre_findings = ds.get_findings()
        encodings, findings, metrics = process_files(config, processes, pre_findings)
        for k, v in metrics.items():
            click.echo(f"{k:<25}: {v}")
        ds.update_findings(findings)
        ds.update_encodings(encodings)
        ds.save_run_info(metrics)
        if report:
            generate_report(ds.path, ds.get_findings(), ds.get_perf(), report_file)

if __name__ == "__main__":
    cli()

