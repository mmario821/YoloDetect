from pathlib import Path
import argparse
import json
import sys

from ultralytics import YOLO

MODEL = "yolo11s.pt"
OUTPUT_EXTENSION = ".yolo.json"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on image files and write .yolo.json sidecar files.")
    
    parser.add_argument(
        "--folder",
        default="./",
        help="Folder containing images to process, default is local directory."
    )

    parser.add_argument(
        "--file",
        help="Single image file to process, when set, --folder will be ignored."
    )

    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="YOLO model to use. Default: yolo11s.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25"
    )

    parser.add_argument(
        "--save-detection",
        action="store_true",
        help="Save processed image with detection zones embedded."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .yolo.json files."
    )

    return parser.parse_args()    

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

def remove_images_with_nonexistent_outputs(files):
    return [
        path for path in files
        if not Path(path).with_suffix(OUTPUT_EXTENSION).is_file()
    ]

def filter_to_existent_images_only(files):
    return [
        path for path in files
        if Path(path).is_file()
    ]


def scan_for_images(folder):
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Folder {folder_path} does not exists.")
        sys.exit(1)

    return [
        str(p)
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".png"]
    ]

def main():
    args = parse_args()

    if args.file == None:
        images = scan_for_images(str(args.folder))
    else:
        images = [ args.file ]

    model = YOLO(args.model)

    filtered_images = filter_to_existent_images_only(images)

    if args.overwrite != None and args.overwrite == False:
        filtered_images = remove_images_with_nonexistent_outputs(images)

    if not filtered_images:
        print("No images found to process.")
        sys.exit(0)

    results = model.predict(
        source=filtered_images,
        conf=args.conf,
        save=args.save_detection,
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

        write_file(image_path, detections, args.overwrite)

if __name__ == "__main__":
    main()


    