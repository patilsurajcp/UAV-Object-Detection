# detect.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awning-tricycle',
    'bus', 'motor'
]

CLASS_COLORS = {
    'pedestrian'     : '#FF6B6B',
    'people'         : '#FF8E53',
    'bicycle'        : '#FFC300',
    'car'            : '#2ECC71',
    'van'            : '#1ABC9C',
    'truck'          : '#3498DB',
    'tricycle'       : '#9B59B6',
    'awning-tricycle': '#E91E63',
    'bus'            : '#FF5722',
    'motor'          : '#607D8B',
}

def _patch_torch_load():
    """Patch torch.load to always use weights_only=False"""
    original = torch.load
    def patched(*args, **kwargs):
        kwargs['weights_only'] = False
        return original(*args, **kwargs)
    torch.load = patched

def _add_safe_globals():
    """Add ultralytics classes to torch safe globals"""
    try:
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.conv  import Conv, Concat
        from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck, DFL
        from ultralytics.nn.modules.head  import Detect
        import torch.nn as nn
        from collections import OrderedDict

        torch.serialization.add_safe_globals([
            DetectionModel, Conv, Concat,
            C2f, SPPF, Bottleneck, DFL, Detect,
            nn.ModuleList, nn.Sequential,
            nn.Conv2d, nn.BatchNorm2d,
            nn.SiLU, nn.Upsample,
            nn.MaxPool2d, nn.Identity,
            OrderedDict, set,
        ])
        print("✅ Safe globals added!")
    except Exception as e:
        print(f"⚠️ Safe globals warning: {e}")


class UAVDetector:
    def __init__(self, model_path):
        print(f"⏳ Loading model: {model_path}")
        print(f"🔥 PyTorch version: {torch.__version__}")

        # ✅ Apply fixes before loading
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
        _add_safe_globals()
        _patch_torch_load()

        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Load error: {e}")
            raise e

        self.class_names = CLASS_NAMES
        self.colors      = CLASS_COLORS

    def detect_image(self, image, conf=0.35, iou=0.45):
        """Run detection on image array"""
        try:
            results = self.model(
                image,
                conf    = conf,
                iou     = iou,
                imgsz   = 640,
                verbose = False
            )[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls)
                if cls_id >= len(self.class_names):
                    continue

                cls_name     = self.class_names[cls_id]
                conf_val     = float(box.conf)
                x1,y1,x2,y2 = box.xyxy[0].tolist()

                detections.append({
                    'class'     : cls_name,
                    'confidence': round(conf_val, 3),
                    'bbox'      : [x1, y1, x2, y2],
                    'center_x'  : round((x1 + x2) / 2, 2),
                    'center_y'  : round((y1 + y2) / 2, 2),
                    'width'     : round(x2 - x1, 2),
                    'height'    : round(y2 - y1, 2),
                    'color'     : self.colors.get(cls_name, '#FFFFFF')
                })

            # ✅ Small text + thin boxes
            annotated = results.plot(
                font_size  = 8,
                line_width = 1,
                labels     = True,
                conf       = True,
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return annotated, detections

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return image, []

    def detect_video_frame(self, frame, conf=0.35, iou=0.45):
        """Run detection on single video frame"""
        try:
            results = self.model(
                frame,
                conf    = conf,
                iou     = iou,
                imgsz   = 640,
                verbose = False
            )[0]

            annotated = results.plot(
                font_size  = 8,
                line_width = 1,
                labels     = True,
                conf       = True,
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            return annotated, len(results.boxes)

        except Exception as e:
            print(f"❌ Frame error: {e}")
            return frame, 0