from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from collections import Counter
import time
from datetime import datetime

app = Flask(__name__)

model = YOLO("./epoch140.pt")

CLASS_NAMES = {0: "Drain Hole", 1: "Pothole", 2: "Sewer Cover"}

# Define severity levels based on area and confidence
def calculate_severity(class_name, area, confidence):
    # Customize these thresholds based on your requirements
    if class_name == "Pothole":
        if area > 10000 and confidence > 0.8:
            return "Critical"
        elif area > 5000 and confidence > 0.6:
            return "High"
        elif area > 2000 and confidence > 0.4:
            return "Medium"
        else:
            return "Low"
    else:  # For other classes
        if area > 8000 and confidence > 0.8:
            return "High"
        elif area > 4000 and confidence > 0.6:
            return "Medium"
        else:
            return "Low"

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image uploaded'}), 400
    
    img_data = request.json['image']
    
    try:
        img_bytes = base64.b64decode(img_data)
        image_np = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Store original image dimensions
        original_height, original_width = image.shape[:2]
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
    
    # Model prediction
    results = model.predict(image)
    r = results[0]
    
    detections = []
    class_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = r.boxes.conf.cpu().numpy().tolist()
    boxes = r.boxes.xyxy.cpu().numpy().tolist()
    
    # Get image with detection boxes
    annotated_img = r.plot()
    
    # Additional statistics
    total_area = original_width * original_height
    covered_area = 0
    severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    
    for i in range(len(class_ids)):
        x1, y1, x2, y2 = boxes[i]
        area = (x2 - x1) * (y2 - y1)
        covered_area += area
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        # Calculate relative position
        rel_x = center[0] / original_width
        rel_y = center[1] / original_height
        position_desc = ""
        
        # Determine position in image (9-grid)
        if rel_x < 0.33:
            if rel_y < 0.33:
                position_desc = "top-left"
            elif rel_y < 0.66:
                position_desc = "middle-left"
            else:
                position_desc = "bottom-left"
        elif rel_x < 0.66:
            if rel_y < 0.33:
                position_desc = "top-center"
            elif rel_y < 0.66:
                position_desc = "center"
            else:
                position_desc = "bottom-center"
        else:
            if rel_y < 0.33:
                position_desc = "top-right"
            elif rel_y < 0.66:
                position_desc = "middle-right"
            else:
                position_desc = "bottom-right"
        
        class_name = CLASS_NAMES.get(class_ids[i], "Unknown")
        severity = calculate_severity(class_name, area, confidences[i])
        severity_counts[severity] += 1
        
        # Calculate aspect ratio
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        detections.append({
            "class_id": class_ids[i],
            "class_name": class_name,
            "confidence": round(confidences[i], 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "center": [round(center[0], 2), round(center[1], 2)],
            "relative_position": [round(rel_x, 2), round(rel_y, 2)],
            "position_description": position_desc,
            "area": round(area, 2),
            "width": round(width, 2),
            "height": round(height, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "severity": severity,
            "percentage_of_image": round((area / total_area) * 100, 2)
        })
    
    # Count each class
    class_counts = dict(Counter(CLASS_NAMES.get(cid, "Unknown") for cid in class_ids))
    
    # Process annotated image
    _, buffer = cv2.imencode('.jpg', annotated_img)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate total processing time
    processing_time = round(time.time() - start_time, 3)
    
    # Sort detections by area (largest first)
    detections.sort(key=lambda x: x["area"], reverse=True)
    
    # Generate summary
    summary = {
        "total_detections": len(detections),
        "class_distribution": class_counts,
        "largest_detection": detections[0] if detections else None,
        "average_confidence": round(np.mean(confidences), 4) if confidences else 0,
        "total_covered_area": round(covered_area, 2),
        "percentage_covered": round((covered_area / total_area) * 100, 2),
        "severity_distribution": severity_counts,
        "processing_time_seconds": processing_time,
        "image_dimensions": {
            "width": original_width,
            "height": original_height,
            "aspect_ratio": round(original_width / original_height, 2),
            "resolution": original_width * original_height
        },
        "timestamp": datetime.now().isoformat()
    }

    print(summary)
    
    return jsonify({
        "detections": detections,
        "counts": class_counts,
        "summary": summary,
        "annotated_image": annotated_b64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)

