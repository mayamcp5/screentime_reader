import re
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name
from src.ios.chart_utils import extract_simple_hourly_chart, normalize_hourly

# ================================
# OCR PREPROCESSING
# ================================
def preprocess_for_ocr(image_path: str, light_text: bool = False):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.grayscale(img)

    contrast = 2.2 if light_text else 2.0
    brightness = 1.3 if light_text else 1.2
    threshold_val = 150 if light_text else 200

    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    img_np = np.array(img)
    _, img_np = cv2.threshold(img_np, threshold_val, 255, cv2.THRESH_BINARY_INV)

    return Image.fromarray(img_np)

# ================================
# COLOR DETECTION (RED)
# ================================
def is_notification_bar(r, g, b):
    return (
        r > 200 and
        40 <= g <= 120 and
        30 <= b <= 100
    )

# ================================
# EXTRACTION FUNCTIONS
# ================================
import re

def extract_date(lines):
    """Extract clean date like 'July 22' from OCR lines"""
    
    MONTHS = [
        'january','february','march','april','may','june',
        'july','august','september','october','november','december'
    ]
    
    for i, line in enumerate(lines):
        lower = line.lower()
        
        if 'yesterday' in lower or 'today' in lower:
            # Clean symbols like < >
            cleaned = re.sub(r'[<>]', '', line).strip()
            
            # Try to extract "Month Day" from same line
            match = re.search(
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
                cleaned,
                re.IGNORECASE
            )
            if match:
                return match.group(0)
            
            # Otherwise check next line
            if i + 1 < len(lines):
                next_line = re.sub(r'[<>]', '', lines[i + 1]).strip()
                
                match = re.search(
                    r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
                    next_line,
                    re.IGNORECASE
                )
                if match:
                    return match.group(0)
    
    return None

def extract_total_notifications(lines):
    """Extract the big total number"""
    nums = []
    
    for line in lines:
        lower = line.lower()
        
        # Skip these lines
        if any(word in lower for word in ['yesterday', 'today', 'notification', 'updated', 'most']):
            continue
        
        # Skip time patterns
        if re.search(r'\d{1,2}:\d{2}\s*(AM|PM)', line, re.IGNORECASE):
            continue
        
        # Extract standalone numbers
        if re.match(r'^\d+$', line.strip()):
            val = int(line.strip())
            if val > 0:
                nums.append(val)
    
    return max(nums) if nums else 0

def extract_y_axis_max(image_path):
    """Extract the highest y-axis label"""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w = arr.shape[:2]
    
    # Crop left 15% where y-axis is
    left_section = arr[:, :int(img_w * 0.15)]
    left_img = Image.fromarray(left_section)
    text = ocr_image(left_img)
    
    nums = []
    for line in text.split('\n'):
        cleaned = re.sub(r'[^\d]', '', line)
        if cleaned:
            val = int(cleaned)
            if val > 0:
                nums.append(val)
    
    return max(nums) if nums else None

def extract_top_apps(lines):
    """
    Extract top 3 notification apps - found after the chart.
    Format: App name, then count on next line.
    """
    apps = []
    
    for i, line in enumerate(lines):
        lower = line.lower()
        
        # Skip header/metadata lines
        if any(word in lower for word in ['notification', 'yesterday', 'today', 'updated', 'most', 'allowed']):
            continue
        
        # Skip times
        if re.search(r'\d{1,2}:\d{2}\s*(AM|PM)', line, re.IGNORECASE):
            continue
        
        # Check if next line is a number
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            
            if re.match(r'^\d+$', next_line):
                count = int(next_line)
                app_name = clean_app_name(line)
                
                if is_valid_app_name(app_name) and len(app_name) >= 2:
                    apps.append({
                        'name': app_name,
                        'count': count
                    })
    
    apps.sort(key=lambda x: x['count'], reverse=True)
    return apps[:3]

# ================================
# MAIN FUNCTION
# ================================
def process_ios_notifications_screenshot(image_path: str) -> dict:
    # OCR
    normal_text = ocr_image(preprocess_for_ocr(image_path))
    light_text = ocr_image(preprocess_for_ocr(image_path, light_text=True))
    
    all_text = normal_text + "\n" + light_text
    lines = [l.strip() for l in all_text.split("\n") if l.strip()]
    
    print("=== ALL OCR LINES ===")
    for i, line in enumerate(lines):
        print(f"{i}: {repr(line)}")
    
    # Extract data
    date = extract_date(lines)
    total = extract_total_notifications(lines)
    y_max = extract_y_axis_max(image_path)
    top_apps = extract_top_apps(lines)
    
    print(f"\n=== EXTRACTED DATA ===")
    print(f"Date: {date}")
    print(f"Total: {total}")
    print(f"Y-Max: {y_max}")
    print(f"Top Apps: {top_apps}")
    
    # Extract hourly chart
    hourly_pixels = extract_simple_hourly_chart(
        image_path,
        is_notification_bar,
        debug_output_path="debug_notifications.png"
    )
    
    ymax_pixels = hourly_pixels.pop("ymax_pixels", 0)
    
    print(f"\nHourly pixels: {hourly_pixels}")
    print(f"YMax pixels: {ymax_pixels}")
    
    # Normalize
    normalization_value = y_max if y_max else total
    hourly = normalize_hourly(hourly_pixels, normalization_value, ymax_pixels)
    
    return {
        "date": date,
        "total_notifications": total,
        "y_max": y_max,
        "ymax_pixels": ymax_pixels,
        "top_apps": top_apps,
        "hourly_notifications": hourly
    }