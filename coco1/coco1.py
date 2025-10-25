import cv2
import json
import os
from PIL import Image

# ---------------- CONFIG ----------------
image_folder = "/Users/anish/Desktop/not wearing cap/Data"         # Folder containing input images
output_json = "instances_train.json"
classes = ["person", "dog", "car"]   # Define your class labels here
# ----------------------------------------

# Globals
boxes = []
current_image = None
selected_box = None
start_point = None
dragging = False

images = []
annotations = []
categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(classes)]
img_id = 1
ann_id = 1


def inside_box(x, y, box):
    bx, by, bw, bh = box["bbox"]
    return bx <= x <= bx + bw and by <= y <= by + bh


def draw_all_boxes(img):
    temp = img.copy()
    for box in boxes:
        x, y, w, h = map(int, box["bbox"])
        cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(temp, box["label"], (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return temp


def mouse_event(event, x, y, flags, param):
    global boxes, selected_box, start_point, dragging, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicked inside an existing box
        for box in boxes:
            if inside_box(x, y, box):
                selected_box = box
                start_point = (x, y)
                dragging = True
                return
        # Start new box
        start_point = (x, y)
        selected_box = None

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_box:
        # Move existing box
        dx = x - start_point[0]
        dy = y - start_point[1]
        selected_box["bbox"][0] += dx
        selected_box["bbox"][1] += dy
        start_point = (x, y)
        img_display = draw_all_boxes(current_image)
        cv2.imshow("Annotator", img_display)

    elif event == cv2.EVENT_LBUTTONUP:
        if selected_box is None and start_point is not None:
            # Finish new box
            x1, y1 = start_point
            x2, y2 = x, y
            x_min, y_min = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 10 and h > 10:
                print("\nAvailable classes:")
                for i, name in enumerate(classes):
                    print(f"{i + 1}: {name}")
                cls_idx = int(input("Enter class number: ")) - 1
                boxes.append({
                    "bbox": [x_min, y_min, w, h],
                    "label": classes[cls_idx],
                    "category_id": cls_idx + 1
                })
        dragging = False
        selected_box = None
        img_display = draw_all_boxes(current_image)
        cv2.imshow("Annotator", img_display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete box if clicked inside
        for box in boxes:
            if inside_box(x, y, box):
                boxes.remove(box)
                print(f"Deleted box: {box['label']}")
                img_display = draw_all_boxes(current_image)
                cv2.imshow("Annotator", img_display)
                return


def save_annotations():
    global ann_id
    if not boxes:
        return
    for box in boxes:
        x, y, w, h = box["bbox"]
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": box["category_id"],
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0
        })
        ann_id += 1


# ---------------- MAIN LOOP ----------------
for filename in sorted(os.listdir(image_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(image_folder, filename)
    current_image = cv2.imread(path)
    if current_image is None:
        continue

    height, width = current_image.shape[:2]
    images.append({
        "id": img_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    boxes = []
    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_event)
    img_display = current_image.copy()
    print(f"\n🖼️ Annotating: {filename}")
    print("Left-click + drag = draw box | Right-click = delete box | Drag inside box = move | ENTER = next image | ESC = exit")

    while True:
        cv2.imshow("Annotator", draw_all_boxes(img_display))
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            save_annotations()
            break
        elif key == 27:  # ESC → exit safely
            save_annotations()
            cv2.destroyAllWindows()
            # Write final COCO JSON before exit
            data = {
                "info": {"description": "Manual COCO Dataset", "version": "1.0"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": categories
            }
            with open(output_json, "w") as f:
                json.dump(data, f, indent=4)
            print(f"\n💾 Saved COCO annotations to {output_json}")
            exit()

    img_id += 1

# ---------------- SAVE FINAL JSON ----------------
cv2.destroyAllWindows()
data = {
    "info": {"description": "Manual COCO Dataset", "version": "1.0"},
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories
}
with open(output_json, "w") as f:
    json.dump(data, f, indent=4)

print(f"\n✅ All done! COCO annotations saved to: {output_json}")
