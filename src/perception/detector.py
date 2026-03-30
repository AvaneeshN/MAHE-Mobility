from ultralytics import YOLO
import cv2


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, image):
        """
        Returns list of detections:
        [x1, y1, x2, y2, label, confidence]
        """

        results = self.model(image)[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = self.model.names[cls_id]

            detections.append([x1, y1, x2, y2, label, conf])

        return detections