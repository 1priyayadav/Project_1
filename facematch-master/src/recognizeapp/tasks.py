from recognizeapp.c import app
from recognizeapp.utils import process_image, calculate_distance, check_liveness, check_deepfake
import os
import uuid

@app.task(bind=True)
def face_deduplication_task(self, image_paths, threshold=0.5, output_dir="results"):
    """
    Celery task to perform face deduplication with liveness + deepfake detection.
    
    Args:
        image_paths (list): List of file paths to input images.
        threshold (float): Distance threshold for considering two faces as duplicates.
        output_dir (str): Directory where results will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    findings = []
    metrics = {
        "total_images": len(image_paths),
        "processed_images": 0,
        "duplicates_found": 0,
        "spoofs_detected": 0,
        "deepfakes_detected": 0,
    }

    encodings = {}
    liveness_results = {}
    deepfake_results = {}

    # Step 1: Preprocess images
    for img_path in image_paths:
        encoding, face_img = process_image(img_path)

        if encoding is None:
            encodings[img_path] = None
            liveness_results[img_path] = "NO FACE DETECTED"
            deepfake_results[img_path] = "NO FACE DETECTED"
        else:
            encodings[img_path] = encoding
            # Run liveness + deepfake check
            liveness_results[img_path] = check_liveness(face_img)
            deepfake_results[img_path] = check_deepfake(face_img)

            if liveness_results[img_path] == "spoof":
                metrics["spoofs_detected"] += 1
            if deepfake_results[img_path] == "fake":
                metrics["deepfakes_detected"] += 1

        metrics["processed_images"] += 1

    # Step 2: Compare pairs
    for i, img1 in enumerate(image_paths):
        for j, img2 in enumerate(image_paths):
            if j <= i:
                continue  # skip duplicate pairs

            enc1, enc2 = encodings.get(img1), encodings.get(img2)

            if enc1 is None or enc2 is None:
                distance = None
            else:
                distance = calculate_distance(enc1, enc2)

            entry = {
                "first": img1 if enc1 is not None else "NO ENCODING",
                "second": img2 if enc2 is not None else "NO ENCODING",
                "distance": distance if distance is not None else "N/A",
                "liveness": liveness_results.get(img1, "N/A"),
                "deepfake": deepfake_results.get(img1, "N/A"),
                "pair_key": str(uuid.uuid4())  # unique pair identifier
            }

            if distance is not None and distance <= threshold:
                metrics["duplicates_found"] += 1

            findings.append(entry)

    # Step 3: Save results as dict (Celery backend or JSON output)
    results = {
        "metrics": metrics,
        "findings": findings
    }

    return results
