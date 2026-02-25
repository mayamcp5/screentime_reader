import re
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2

from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name

# ================================
# OCR PREPROCESSING
# ================================

def preprocess_for_ocr(image_path: str, light_text: bool = False) -> Image.Image:
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
# PIXEL CLASSIFICATION
# ================================

HOURS = [f"{h if h!=0 else 12}am" if h < 12 else f"{h-12 if h>12 else 12}pm" for h in range(24)]

def classify_pixel(r, g, b, mode='dark'):
    r, g, b = int(r), int(g), int(b)
    
    if mode == 'light':
        if r < 100 and 110 <= g <= 170 and b > 210:  # Blue
            return "top1"
        if r < 140 and g > 165 and 180 < b < 235:  # Teal
            return "top2"
        if r > 210 and 135 <= g <= 185 and b < 85:  # Orange
            return "top3"
        if 200 <= r <= 220 and 200 <= g <= 220 and 200 <= b <= 220 and abs(r-g) < 8 and abs(g-b) < 8:
            return "other"
    else:
        if r < 80 and 100 <= g <= 160 and b > 220:
            return "top1"
        if r < 150 and g > 170 and 180 < b < 240:
            return "top2"
        if r > 210 and 140 <= g <= 190 and b < 90:
            return "top3"
        if 48 <= r <= 68 and 48 <= g <= 68 and 48 <= b <= 68 and abs(r-g) < 5 and abs(g-b) < 5:
            return "other"
    
    return None

def is_chart_bg(r, g, b):
    return 20 <= int(r) <= 45 and 20 <= int(g) <= 45 and 20 <= int(b) <= 50

def is_gridline_pixel(r, g, b, mode='dark'):
    r, g, b = int(r), int(g), int(b)
    if abs(r - g) > 8 or abs(g - b) > 8:
        return False
    brightness = (r + g + b) / 3
    if mode == 'light':
        return 190 <= brightness <= 230
    else:
        return 50 <= brightness <= 110

# ================================
# HOURLY CHART EXTRACTION (with debug)
# ================================
def extract_hourly_chart(image_path: str, debug_output_path=None) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    img_h, img_w = arr.shape[:2]

    # Detect light vs dark mode
    sample = arr[100:200, 100:200]
    avg_brightness = np.mean(sample)
    mode = 'light' if avg_brightness > 127 else 'dark'

    # --- Find chart bar region ---
    def find_bar_regions(probe_x):
        regions = []
        in_region = False
        start = None
        for y in range(img_h // 4, img_h):
            has_bar = classify_pixel(*arr[y, probe_x], mode) is not None
            if has_bar and not in_region:
                start = y
                in_region = True
            elif not has_bar and in_region:
                if y - start > 20:
                    regions.append((start, y))
                in_region = False
        return regions

    vertical_probes = range(img_w // 3, 2 * img_w // 3, 40)
    best_regions = max(
        (find_bar_regions(x) for x in vertical_probes),
        key=len,
        default=[]
    )
    if not best_regions:
        return {}
    
    # DEBUG: list all candidate regions
    for i, (top, bottom) in enumerate(best_regions):
        print(f"Candidate region {i}: top={top}, bottom={bottom}, height={bottom-top}")

    # Pick the bottommost region with height >= 20px (filters noise, always
    # gives the hourly chart since it sits below the weekly chart on screen).
    MIN_CHART_HEIGHT = 20
    substantial_regions = [(top, bottom) for top, bottom in best_regions
                           if (bottom - top) >= MIN_CHART_HEIGHT]
    if not substantial_regions:
        return {}
    chart_top, chart_bottom = max(substantial_regions, key=lambda r: r[0])
    chart_bottom += 1

    # --- Detect gridlines ---
    gridlines = []
    for y in range(chart_top - 500, chart_bottom):
        if y < 0 or y >= img_h:
            continue
        row = arr[y]
        gray_count = sum(is_gridline_pixel(*px, mode) for px in row)
        if gray_count > 0.4 * img_w:
            gridlines.append(y)

    collapsed = []
    if gridlines:
        group = [gridlines[0]]
        for y in gridlines[1:]:
            if y - group[-1] <= 2:
                group.append(y)
            else:
                collapsed.append(int(sum(group)/len(group)))
                group = [y]
        collapsed.append(int(sum(group)/len(group)))
    collapsed.sort()

    NUM_GRIDLINES = 5
    if len(collapsed) > NUM_GRIDLINES:
        collapsed = collapsed[-NUM_GRIDLINES:]

    chart_top_line = collapsed[0]
    chart_bottom_line = collapsed[-1]
    ymax = chart_bottom_line - chart_top_line

    # --- Find vertical axes ---
    def find_vertical_axis(search_from_left=True):
        x_range = range(img_w) if search_from_left else range(img_w - 1, -1, -1)
        for x in x_range:
            col = arr[chart_top_line:chart_bottom_line, x]
            count = sum(is_gridline_pixel(*px, mode) for px in col)
            if count > 0.35 * (chart_bottom_line - chart_top_line):
                return x
        return None

    chart_left = find_vertical_axis(True)
    chart_right = find_vertical_axis(False)

    if chart_left is None or chart_right is None or chart_right <= chart_left:
        return {}

    total_width = chart_right - chart_left
    slot_width = total_width / 24.0
    slot_centers = []
    for i in range(24):
        start = chart_left + slot_width * i
        end   = chart_left + slot_width * (i+1)
        center = (start + end) / 2
        slot_centers.append(center)

    # --- Detect bars ---
    bar_segments = []
    in_bar = False
    seg_start = None

    for x in range(chart_left, chart_right):
        col = arr[chart_top_line:chart_bottom_line, x]
        vertical_run = 0
        max_vertical_run = 0

        for px in col:
            if classify_pixel(*px, mode) is not None:
                vertical_run += 1
                max_vertical_run = max(max_vertical_run, vertical_run)
            else:
                vertical_run = 0

        has_bar = max_vertical_run >= 2

        if has_bar and not in_bar:
            seg_start = x
            in_bar = True
        elif not has_bar and in_bar:
            if x - seg_start >= 2:
                bar_segments.append((seg_start, x - 1))
            in_bar = False

    if in_bar and chart_right - seg_start >= 2:
        bar_segments.append((seg_start, chart_right - 1))

    result = {hour: {"overall":0, "top1":0,"top2":0,"top3":0,"other":0} for hour in HOURS}
    result['ymax_pixels'] = ymax

    # --- Debug image ---
    if debug_output_path:
        debug_img = img.copy()
        debug_draw = cv2.cvtColor(np.array(debug_img), cv2.COLOR_RGB2BGR)

        # Show detected "other" (gray) pixels in bright green
        for y in range(chart_top_line, chart_bottom_line):
            for x in range(chart_left, chart_right):
                cat = classify_pixel(*arr[y, x], mode)
                if cat == "other":
                    debug_draw[y, x] = (0, 255, 0)  # Bright green

        # Draw gridlines in RED
        for y in collapsed:
            cv2.line(debug_draw, (0, y), (img_w-1, y), (0, 0, 255), 1)

        # Draw top line in BLUE
        cv2.line(debug_draw, (0, chart_top_line), (img_w-1, chart_top_line), (255, 0, 0), 2)

        # Draw bottom line in PURPLE
        cv2.line(debug_draw, (0, chart_bottom_line), (img_w-1, chart_bottom_line), (128, 0, 128), 2)

        cv2.imwrite(debug_output_path, debug_draw)


    # --- Process bars and overlay debug heights ---
    for x1, x2 in bar_segments:
        bar_center = (x1 + x2) / 2.0

        distances = [abs(bar_center - c) for c in slot_centers]
        slot_idx = distances.index(min(distances))
        hour = HOURS[slot_idx]

        best = {"overall":0,"top1":0,"top2":0,"top3":0,"other":0}

        for x in range(x1, x2+1):
            col = arr[chart_top_line:chart_bottom_line, x]
            cats = [classify_pixel(*px, mode) for px in col]
            bar_rows = [y for y,c in enumerate(cats) if c is not None]

            if len(bar_rows) < 2:
                continue

            # Find the LOWEST classified pixel (true bottom of this bar)
            bar_bottom_idx = max(bar_rows)

            # Walk upward from bottom until classification stops
            bar_top_idx = bar_bottom_idx
            for y in range(bar_bottom_idx, -1, -1):
                if cats[y] is not None:
                    bar_top_idx = y
                else:
                    break

            bar_height = bar_bottom_idx - bar_top_idx + 1
            bar_top = chart_top_line + bar_top_idx


            if bar_height < 1:
                continue

            if bar_height > best['overall']:
                best['overall'] = bar_height
                best.update({cat: sum(1 for c in cats if c==cat) for cat in ["top1","top2","top3","other"]})

        result[hour] = best

        # --- Draw colored bar heights for debug ---
        if debug_output_path and best['overall'] > 0:
            for cat,color in zip(["top1","top2","top3","other"],[(0,0,255),(255,0,0),(0,140,255),(128,128,128)]):
                h = best.get(cat,0)
                if h>0:
                    y_start = chart_bottom_line
                    y_end = chart_bottom_line - h
                    cv2.line(debug_draw, (x1, y_start), (x1, y_end), color, 1)
                    cv2.line(debug_draw, (x2, y_start), (x2, y_end), color, 1)

    if debug_output_path:
        cv2.imwrite(debug_output_path, debug_draw)

    return result

# ================================
# IOS SCREEN TIME PROCESSING
# ================================
def process_ios_overall_screenshot(image_path: str) -> dict:
    normal_text = ocr_image(preprocess_for_ocr(image_path))
    light_text = ocr_image(preprocess_for_ocr(image_path, light_text=True))
    lines = [l.strip() for l in (normal_text + "\n" + light_text).split("\n") if l.strip()]

    print("Light OCR lines:", light_text)
    print("ALL LINES:")
    for l in lines:
        print(repr(l))


    result = {
        "date": None, "is_yesterday": False, "total_time":"0h 0m",
        "categories": [], "top_apps": [], "ymax_pixels": None, "hourly_usage": {}
    }

    for line in lines:
        if "yesterday" in line.lower():
            result["is_yesterday"]=True
            result["date"]=line.split(",")[-1].strip() if "," in line else line.strip()
            break

    in_total_section=False
    for line in lines:
        if "screen time" in line.lower():
            in_total_section=True
            continue
        if in_total_section:
            parsed=parse_time_fragment(line)
            if parsed:
                h,m=parsed
                if h+m>0:
                    result["total_time"]=f"{h}h {m}m"
                    break

    # ================================
    # ROBUST CATEGORY EXTRACTION
    # ================================

    CANONICAL_CATEGORIES = {
        "social": ["social"],
        "entertainment": ["entertainment"],
        "education": ["education"],
        "games": ["games", "game"],
        "productivity": ["productivity"],
        "creativity": ["creativity"],
        "utilities": ["utilities", "utility"],
        "shopping & food": ["shopping", "food"],
        "travel": ["travel"],
        "health & fitness": ["health", "fitness"],
        "information & reading": ["information", "reading"],
        "finance": ["finance"],
        "other": ["other"]
    }

    # Build a flat keyword -> canonical map for order-preserving detection
    word_to_canonical = {}
    for canonical, keywords in CANONICAL_CATEGORIES.items():
        for kw in keywords:
            word_to_canonical[kw] = canonical

    def normalize_line(text):
        text = text.lower()
        text = text.replace("&", " ")
        text = re.sub(r"[^a-z ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    for i, line in enumerate(lines):
        norm = normalize_line(line)
        words = norm.split()

        # FIX: detect categories in the order they appear in the line,
        # not in dict-definition order, so "Games Entertainment" won't
        # become "Entertainment Games".
        detected_categories = []
        seen_canonicals = set()
        for word in words:
            if word in word_to_canonical:
                canonical = word_to_canonical[word]
                if canonical not in seen_canonicals:
                    seen_canonicals.add(canonical)
                    detected_categories.append(canonical)

        if detected_categories and i + 1 < len(lines):
            times_line = lines[i + 1]
            time_matches = re.findall(r"(\d+h\s*\d*m|\d+h|\d+m)", times_line)

            for cat_name, tstr in zip(detected_categories, time_matches):
                parsed = parse_time_fragment(tstr)
                if not parsed:
                    continue
                h, m = parsed
                result["categories"].append({
                    "name": cat_name.title(),
                    "time": f"{h}h {m}m"
                })

            break

    color_to_category={}
    for idx,cat in enumerate(result["categories"]):
        color_to_category[f"top{idx+1}"]=cat["name"].lower()

    # FIX: normalise time strings before parsing so that OCR errors like
    # "th 26m" (OCR misreads '1' as 't') are corrected to "1h 26m".
    def robust_parse_time(text: str):
        fixed = re.sub(r'\b[tTlL]h\b', '1h', text)
        return parse_time_fragment(fixed)

    def clean_candidate(raw: str):
        candidate = re.sub(r'^[a-zA-Z&\d]{1,2}[\.\s]\s*', '', raw)
        candidate = re.sub(r"['\-,\.]+$", "", candidate)
        candidate = re.sub(r"^['\"]|['\"]$", "", candidate)
        candidate = re.sub(r'\s+\d+$', '', candidate)
        return candidate.strip()

    STOP_PATTERNS = [
        r'\d{1,2}:\d{2}\s*(AM|PM)',
        r'\d+%',
        r'screen time', r'all devices', r'week day', r'yesterday',
        r'updated today', r'most used', r'social', r'entertainment',
        r'education', r'shopping', r'utilities', r'limits'
    ]

    def is_stop_line(text: str) -> bool:
        tl = text.lower()
        return any(re.search(pat, tl, re.IGNORECASE) for pat in STOP_PATTERNS)

    def extract_apps_from_lines(source_lines: list) -> list:
        """
        Walk source_lines after 'MOST USED', pairing app names with the time
        that immediately follows them (name on one line, time on the next).
        Returns list of {"name": ..., "time": ...} dicts.
        """
        apps = []
        in_most_used = False
        pending_app = None

        for line in source_lines:
            if "most used" in line.lower():
                in_most_used = True
                continue
            if not in_most_used:
                continue
            if is_stop_line(line):
                break

            parsed = robust_parse_time(line)
            if parsed:
                h, m = parsed
                if h + m > 0 and pending_app:
                    apps.append({"name": pending_app, "time": f"{h}h {m}m"})
                    pending_app = None
                    continue
                # time line but no pending app — skip
                continue

            # Not a time line — treat as potential app name
            candidate = clean_candidate(line)
            name = clean_app_name(candidate)
            if is_valid_app_name(name) and len(name) >= 3:
                normalized = re.sub(r'\W+$', '', name).strip()
                if len(normalized) >= 3:
                    pending_app = normalized

        return apps

    # Strategy: try light_text lines first (times reliably appear on separate
    # lines there for both dark AND light mode screenshots), fall back to
    # merged lines if light_text yields nothing.
    light_lines = [l.strip() for l in light_text.split("\n") if l.strip()]
    top_apps = extract_apps_from_lines(light_lines)

    if not top_apps:
        top_apps = extract_apps_from_lines(lines)

    def minutes(t):
        parsed = parse_time_fragment(t)
        if not parsed:
            return 0
        h, m = parsed
        return h * 60 + m

    top_apps.sort(key=lambda x: minutes(x["time"]), reverse=True)
    result["top_apps"] = top_apps[:3]

    hourly_raw = extract_hourly_chart(image_path, debug_output_path="debug_output.png")
    result["ymax_pixels"]=hourly_raw.pop("ymax_pixels", None)

    color_to_category={}
    if len(result["categories"])>=1:
        color_to_category["top1"]=result["categories"][0]["name"].lower()
    if len(result["categories"])>=2:
        color_to_category["top2"]=result["categories"][1]["name"].lower()
    if len(result["categories"])>=3:
        color_to_category["top3"]=result["categories"][2]["name"].lower()

    hourly_cleaned={}
    for hour in HOURS:
        raw=hourly_raw.get(hour,{})
        hour_data={"overall":raw.get("overall",0),"social":0,"entertainment":0}
        for color_key,category_name in color_to_category.items():
            if category_name=="social":
                hour_data["social"]=raw.get(color_key,0)
            if category_name=="entertainment":
                hour_data["entertainment"]=raw.get(color_key,0)
        hourly_cleaned[hour]=hour_data

    result["hourly_usage"]=hourly_cleaned
    return result