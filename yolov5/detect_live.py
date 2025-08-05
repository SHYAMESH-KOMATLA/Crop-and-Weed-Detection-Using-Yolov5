import torch
import torch.serialization
from models.yolo import DetectionModel
import cv2
from pathlib import Path
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 🛠️ Custom scale_coords
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

# ✅ Allow YOLOv5 custom model to be loaded
torch.serialization.add_safe_globals({'models.yolo.DetectionModel': DetectionModel})

# ✅ Load model
weights_path = 'best_windows.pt'
device = select_device('')
ckpt = torch.load(weights_path, map_location=device, weights_only=False)
model = ckpt.to(device).float()
model.eval()

# 🏷️ Class names (You can modify these if different)
names = {0: 'crop', 1: 'weed'}

# 📷 Connect to camera
cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work

last_pred = None  # 🧠 Keep last valid detections to reduce flicker

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot grab frame from camera")
        break

    # 🧠 Resize to 416x416 for faster processing
    img = cv2.resize(frame, (416, 416))
    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.10, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            last_pred = pred.clone()  # Save last successful prediction
        elif last_pred is not None:
            pred = last_pred

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                label = f'{names.get(int(cls), f"class_{int(cls)}")} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('📷 Live Detection - DroidCam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
