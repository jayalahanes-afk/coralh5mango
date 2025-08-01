import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r'C:\Users\Admin\Desktop\Results\MangoYOLO\yolov5\runs\train\mango_yolov5s16\weights\best.pt')

def count_mangoes(image_path):
    results = model(image_path)
    # results.xyxy[0] contains (xmin, ymin, xmax, ymax, confidence, class)
    detections = results.xyxy[0]
    # Filter by mango class (0) and confidence threshold
    mango_dets = [d for d in detections if int(d[5]) == 0 and d[4] > 0.3]
    return len(mango_dets)

# Example
img = r"C:\Users\Admin\Desktop\Raw-mangos-on-tree.jpg"
print(f"Mango count: {count_mangoes(img)}")
