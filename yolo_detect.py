from pathlib import Path
import json

from ultralytics import YOLO

MODEL = "yolo11s.pt"
OUTPUT_EXTENSION = ".yolo.json"

def write_file(filename, content, overwrite=False):

    json_path = Path(filename).with_suffix(OUTPUT_EXTENSION)

    if (json_path.exists() and not overwrite):
        print(f"Skipping existing file: {filename}.")
    else:
        json_path.write_text(
            json.dumps({
                "image": filename,
                "model": MODEL,
                "detections": content
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Photo {filename} processed and the results are stored.")

def remove_nonexistant_images(files):
    return [
        path for path in files
        if Path(path).is_file()
    ]

def scan_for_images(folder):
    folder_path = Path(folder)

    return [
        str(p)
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".png"]
    ]

images = scan_for_images("./")

model = YOLO(MODEL)

filtered_images = remove_nonexistant_images(images)

results = model.predict(
    source=filtered_images,
    conf=0.25,
    save=False,
    verbose=False
)

for image_path, result in zip(filtered_images, results):
    if len(filtered_images) != len(results):
        raise RuntimeError("YOLO did not successfully process all images.")

    detections = []

    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bbox_xyxy": [
                round(x1, 1),
                round(y1, 1),
                round(x2, 1),
                round(y2, 1)
            ]
        })

    # YOLO processing results per image - for debug only
    # print(json.dumps({
    #     "image": str(result.path),
    #     "model": MODEL,
    #     "detections": detections
    # }, indent=2))

    write_file(image_path, detections)


    