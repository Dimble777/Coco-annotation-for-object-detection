import cv2
import json
import os
from PIL import Image

# ---------- CONFIG ----------
image_folder = "/Users/anish/Desktop/not wearinr cap/Data"  # folder with your images
output_json = "instances_train.json"
classes = ["person", "dog", "car", "staff not wearing cap"]  # edit this list for your dataset
# ----------------------------

annotations = []
images = []
categories = []
bbox_id = 1
img_id = 1

for i, cls_name in enumerate(classes):
    categories.append({"id": i + 1, "name": cls_name, "supercategory": "none"})

# Mouse callback vars
drawing = False
ix, iy = -1, -1
current_boxes = []


def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, current_boxes, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = img_copy.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotator", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, y_min = min(ix, x), min(iy, y)
        w, h = abs(ix - x), abs(iy - y)
        current_boxes.append([x_min, y_min, w, h])
        cv2.rectangle(img_copy, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        cv2.imshow("Annotator", img_copy)


# Process each image
for filename in sorted(os.listdir(image_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)
    if img is None:
        continue

    height, width = img.shape[:2]
    images.append({
        "id": img_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    current_boxes = []
    img_copy = img.copy()
    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", draw_box)

    print(f"\n🖼️ Annotating: {filename}")
    print("Draw boxes with LEFT mouse button. Press ENTER when done, or 's' to skip this image.")

    while True:
        cv2.imshow("Annotator", img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER -> finish
            break
        elif key == ord("s"):  # skip image
            current_boxes = []
            break
        elif key == 27:  # ESC -> exit program
            cv2.destroyAllWindows()
            exit()

    # Label each bounding box
    for box in current_boxes:
        print(f"Box: {box}")
        for idx, name in enumerate(classes):
            print(f"{idx+1}: {name}")
        cls_id = int(input("Enter class number: ")) - 1

        annotations.append({
            "id": bbox_id,
            "image_id": img_id,
            "category_id": cls_id + 1,
            "bbox": box,
            "area": box[2] * box[3],
            "iscrowd": 0
        })
        bbox_id += 1

    img_id += 1

cv2.destroyAllWindows()

# Create COCO JSON structure
coco_data = {
    "info": {
        "description": "Manual COCO Annotation Dataset",
        "version": "1.0",
        "year": 2025
    },
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save JSON
with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"\n✅ Saved annotations in COCO format: {output_json}")
