from ultralytics import YOLO
import cv2
import numpy as np

class AutoLabeler:
    def __init__(self):
        print("Loading Auto-Labeler (YOLOv8)...")
        # Load a pre-trained model (it will download automatically)
        self.model = YOLO("yolov8n.pt")  # 'n' is nano (fastest)

    def label_image(self, image_pil):
        """
        Runs detection on the generated image and returns YOLO-formatted labels.
        """
        # Convert PIL to OpenCv format
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model(img_cv, verbose=False)[0]
        
        labels = []
        # Extract bounding boxes
        for box in results.boxes:
            # YOLO format: class_id x_center y_center width height (normalized 0-1)
            cls = int(box.cls[0])
            x, y, w, h = box.xywhn[0].tolist() # Normalized coordinates
            conf = float(box.conf[0])
            
            # Filter low confidence
            if conf > 0.4:
                labels.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                
        return "\n".join(labels)