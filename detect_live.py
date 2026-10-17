"""
License Plate Detection - Video/Camera Feed
Detects license plates and reads text using YOLO + EasyOCR
"""

import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import torch
import sys
import re
from collections import Counter

# Initialize EasyOCR
print("Initializing EasyOCR...")
print(f"CUDA available: {torch.cuda.is_available()}")
ocr_reader = easyocr.Reader(['en'], gpu=True)

# Valid Indian state/UT codes
VALID_INDIAN_STATE_CODES = (
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", 
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", 
    "RJ", "SK", "TN", "TS", "TG", "TR", "UP", "UK", "WB", 
    "AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY", "BH"
)

# Character correction mappings (OCR confusion pairs)
NUM_TO_LETTER_TRANS = str.maketrans('0123456789', 'OIZEASGTBG')
LETTER_TO_NUM_TRANS = str.maketrans('OQDIZLJEAHSGCTYBPR', '000121134456677899')


def is_valid_plate(text):
    """Check if plate follows valid Indian format (10 chars, valid state code)"""
    return len(text) == 10 and text[:2] in VALID_INDIAN_STATE_CODES


def correct_plate_text(text):
    """
        Apply strict Indian plate format corrections.
        Format: SS DD XX NNNN (10 characters)
        - Positions 0-1: State code (letters)
        - Position 2-3: District code (numbers) - position 3 is almost always a number
        - Positions 4-5: Series code (letters) 
        - Positions 6-9: Registration number (numbers)
    """
    if len(text) != 10: return text
    state = text[:2].translate(NUM_TO_LETTER_TRANS)
    district = text[2:4].translate(LETTER_TO_NUM_TRANS)
    series = text[4:6].translate(NUM_TO_LETTER_TRANS)
    number = text[6:].translate(LETTER_TO_NUM_TRANS)
    return state + district + series + number

def preprocess_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None
    
    
    h, w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Angle correction 
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    if lines is not None and len(lines) > 0:
        angles = [np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) for l in lines]
        median_angle = np.median(angles) * 180 / np.pi
        if 2 < abs(median_angle) < 45:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
    
    # 2. Motion blur reduction
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2)
        gray = cv2.addWeighted(gray, 1.3, gaussian, -0.3, 0)
    
    # 3. Upscale small plates
    min_height = 80
    if h < min_height:
        scale = min(min_height / h, 4.0)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        h, w = gray.shape[:2]
    
    # 4. CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # 5. Light sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # 6. Adaptive denoising
    sigma = np.std(cv2.Laplacian(gray, cv2.CV_64F))
    if sigma > 80:
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
    elif sigma > 40:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
    # 7. Convert to BGR and pad
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    padded = cv2.copyMakeBorder(processed, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded, w / h

def read_plate_ocr(processed_img, aspect_ratio):
    """
    Run EasyOCR on preprocessed plate image.
    Handles single-line and two-line plates.
    """
    ocr_params = {
        'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'mag_ratio': 2,
        'contrast_ths': 0.1,
        'adjust_contrast': 0.7,
        'text_threshold': 0.3,
        'low_text': 0.3,
        'paragraph': False,
        'width_ths': 0.7,
    }
    
    try:
        # Single line OCR
        results = ocr_reader.readtext(processed_img, **ocr_params)
        results_single = ""
        if results:
            sorted_results = sorted(results, key=lambda r: r[0][0][0])
            results_single = "".join([r[1] for r in sorted_results])
        
        # Two-line OCR for square-ish plates
        results_two_line = ""
        if aspect_ratio < 2.0:
            h = processed_img.shape[0]
            mid_y = h // 2
            
            top_results = ocr_reader.readtext(processed_img[:mid_y, :], **ocr_params)
            top_sorted = sorted(top_results, key=lambda r: r[0][0][0]) if top_results else []
            top_text = "".join([r[1] for r in top_sorted])
            
            bottom_results = ocr_reader.readtext(processed_img[mid_y:, :], **ocr_params)
            bottom_sorted = sorted(bottom_results, key=lambda r: r[0][0][0]) if bottom_results else []
            bottom_text = "".join([r[1] for r in bottom_sorted])
            
            results_two_line = top_text + bottom_text
        
        # Clean and pick best result
        single_clean = re.sub(r'[^A-Z0-9]', '', results_single.upper()) if results_single else ""
        two_line_clean = re.sub(r'[^A-Z0-9]', '', results_two_line.upper()) if results_two_line else ""
        
        if len(single_clean) == 10:
            return single_clean
        if len(two_line_clean) == 10:
            return two_line_clean
        
        single_diff = abs(len(single_clean) - 10)
        two_line_diff = abs(len(two_line_clean) - 10)
        
        if two_line_diff < single_diff and two_line_clean:
            return two_line_clean
        return single_clean
        
    except Exception as e:
        return ""


def get_stable_plate(plate_histories, track_id):
    """
    Get stable plate reading using recency-weighted character-level voting.
    Recent readings have more weight than older ones.
    """
    if track_id not in plate_histories or not plate_histories[track_id]:
        return ""
    
    history = plate_histories[track_id]
    
    # If only one reading, return it
    if len(history) == 1:
        return history[0][0]
    
    # Filter to only 10-char texts
    valid_entries = [(t, c) for t, c in history if len(t) == 10]
    if not valid_entries:
        return ""
    
    # Character-level voting with recency weighting
    # More recent = higher weight (exponential decay)
    n = len(valid_entries)
    result = []
    
    for pos in range(10):
        char_weights = {}
        for i, (text, conf) in enumerate(valid_entries):
            char = text[pos]
            # Weight: recent frames get exponentially more weight
            # Frame 0 (oldest) has weight ~0.5, last frame has weight 1.0
            recency_weight = 0.5 + 0.5 * (i / max(n - 1, 1))
            weight = recency_weight * conf
            char_weights[char] = char_weights.get(char, 0) + weight
        
        # Pick character with highest weighted vote
        best_char = max(char_weights, key=char_weights.get)
        result.append(best_char)
    
    return ''.join(result)


def process_plate_text(raw_text):
    """Clean, correct, and validate plate text"""
    if not raw_text:
        return ""
    
    # Clean: uppercase, alphanumeric only
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    # Must be exactly 10 characters
    if len(text) != 10:
        return ""
    
    # Apply position-based corrections
    text = correct_plate_text(text)
    
    # Validate against Indian format
    if not is_valid_plate(text):
        return ""
    
    return text


def main():
    # Load model
    model_path = 'runs/detect/license_plate_detector3/weights/best.pt'
    try:
        model = YOLO(model_path)
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Video source
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    print(f"Using {'video: ' + source if isinstance(source, str) else 'camera'}")
    target_width = 1280
    target_height = 720
    target_fps = 15
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open {source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {width}x{height}")
    print(f"Source Video FPS: {original_fps}")
    print("Press 'q' to quit.")
    frame_skip_interval = int(round(original_fps / target_fps))
    frame_count = 0
    
    cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('License Plate Detection', 1280, 800)
    
    plate_histories = {}  # track_id -> [(text, conf), ...]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        if frame_count % frame_skip_interval != 0:
            continue
        
        frame = cv2.resize(frame, (target_width, target_height))
        # Run YOLO tracking
        results = model.track(frame, conf=0.4, persist=True, verbose=False, imgsz=1280, device=0)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(boxes.id[i])
                
                # Extract plate ROI with padding
                pad = 5
                plate_roi = frame[max(0, y1-pad):min(frame.shape[0], y2+pad),
                                  max(0, x1-pad):min(frame.shape[1], x2+pad)]
                
                if plate_roi.size == 0:
                    continue
                
                # Preprocess and read plate with EasyOCR
                result = preprocess_plate(plate_roi)
                if result is not None:
                    processed, aspect_ratio = result
                    raw_text = read_plate_ocr(processed, aspect_ratio)
                    plate_text = process_plate_text(raw_text)
                else:
                    plate_text = ""
                
                # Track history (keep only last 10 readings)
                if plate_text:
                    if track_id not in plate_histories:
                        plate_histories[track_id] = []
                    plate_histories[track_id].append(plate_text)
                    if len(plate_histories[track_id]) > 10:
                        plate_histories[track_id].pop(0)
                
                # Get stable reading using character-level voting
                stable_text = plate_histories[track_id][-1] if track_id in plate_histories else ""
                
                detections.append((x1, y1, x2, y2, stable_text, track_id))
        
        # Draw detections
        for x1, y1, x2, y2, plate_text, track_id in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{track_id}" + (f" {plate_text}" if plate_text else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.imshow('License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print final results
    print(f"\n=== Final Results (Unique Plates) ===")
    plate_to_tracks = {}
    for track_id, history in plate_histories.items():
        if history:
            text_counts = Counter(history)
            best_text = text_counts.most_common(1)[0][0]
            if best_text not in plate_to_tracks:
                plate_to_tracks[best_text] = []
            plate_to_tracks[best_text].append((track_id, len(history), text_counts[best_text]))
    
    for plate_text, tracks in plate_to_tracks.items():
        total_readings = sum(t[1] for t in tracks)
        total_matches = sum(t[2] for t in tracks)
        track_ids = [t[0] for t in tracks]
        print(f"  {plate_text} ({total_matches}/{total_readings} readings, tracks: {track_ids})")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
