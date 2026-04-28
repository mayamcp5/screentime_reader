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
        if r > 190 and 135 <= g <= 185 and b < 85:  # Orange
            return "top3"
        if 205 <= r <= 220 and 205 <= g <= 220 and 210 <= b <= 225:
            if b - r >= 3 and b - g >= 3:  # Blue tint = bar
                return "other"
    else:
        if 100 <= r and 190 <= g and 215 <= b:
            return "top2"
        if r > 190 and 140 <= g <= 190 and b < 90:
            return "top3"
        if (b - r) > 35 and b > 90:
            return "top1"
        if 40 <= r <= 90 and 40 <= g <= 90 and 40 <= b <= 90 and abs(r-g) < 10 and abs(g-b) < 10:
            return "other"
    
    return None

def is_chart_bg(r, g, b, mode='dark'):
    r, g, b = int(r), int(g), int(b)
    if mode == 'light':
        # near-white background
        return r > 240 and g > 240 and b > 240
    else:
        # dark mode background
        return (
            15 <= r <= 50 and
            15 <= g <= 50 and
            15 <= b <= 60 and
            abs(r - g) < 8 and
            abs(g - b) < 8
        )


def is_gridline_pixel(r, g, b, mode='dark'):
    r, g, b = int(r), int(g), int(b)
    if abs(r - g) > 8 or abs(g - b) > 8:
        return False
    brightness = (r + g + b) / 3
    if mode == 'light':
        return 200 <= brightness <= 225
    else:
        return 40 <= brightness <= 120
    
def is_gridline_pixel_safe(r, g, b, arr=None, y=None, x=None, mode='dark'):
    """Return True only if the pixel is part of a proper horizontal gridline."""
    if not is_gridline_pixel(r, g, b, mode):
        return False

    if arr is None or y is None or x is None:
        return True

    # Require at least 10 contiguous pixels horizontally to be considered a gridline
    run = 1
    for dx in range(1, 10):
        if x - dx >= 0 and is_gridline_pixel(*arr[y, x-dx], mode):
            run += 1
        else:
            break
    for dx in range(1, 10):
        if x + dx < arr.shape[1] and is_gridline_pixel(*arr[y, x+dx], mode):
            run += 1
        else:
            break
    return run >= 10

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

    # ============================================================
    # STEP 1: Detect horizontal gridlines across entire image
    # ============================================================

    gridlines = []

    for y in range(img_h):
        row = arr[y]
        gray_count = sum(is_gridline_pixel(*px, mode) for px in row)

        if gray_count > 0.15 * img_w:
            gridlines.append(y)

    # Collapse adjacent rows into single gridlines
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

    # ============================================================
    # STEP 2: Search for 5 evenly spaced gridlines (chart pattern)
    # ============================================================

    chart_top_line = None
    chart_bottom_line = None

    for i in range(len(collapsed) - 4):
        candidate = collapsed[i:i+5]

        spacings = [
            candidate[j+1] - candidate[j]
            for j in range(4)
        ]

        avg = sum(spacings) / len(spacings)

        # check spacing consistency
        if all(abs(s - avg) < avg * 0.35 for s in spacings):
            chart_top_line = candidate[0]
            chart_bottom_line = candidate[-1]

    if chart_top_line is None:
        return {}

    ymax = chart_bottom_line - chart_top_line

    # ============================================================
    # STEP 3: Detect vertical axes
    # ============================================================

    def find_vertical_axis(search_from_left=True):

        x_range = range(img_w) if search_from_left else range(img_w-1, -1, -1)

        for x in x_range:

            col = arr[chart_top_line:chart_bottom_line, x]

            count = sum(is_gridline_pixel(*px, mode) for px in col)

            if count > 0.35 * (chart_bottom_line-chart_top_line):
                return x

        return None


    chart_left = find_vertical_axis(True)
    chart_right = find_vertical_axis(False)

    if chart_left is None or chart_right is None:
        return {}

    total_width = chart_right - chart_left
    slot_width = total_width / 24.0

    slot_centers = []

    for i in range(24):

        start = chart_left + slot_width * i
        end   = chart_left + slot_width * (i+1)

        slot_centers.append((start+end)/2)

    # ============================================================
    # STEP 4: Detect bar heights per hour slot
    # ============================================================
    # within each hour slot:
    #   scan for bar pixels in that section
    #       start by checking at the y value immediately above the x-axis. if it is not a background pixel, then start a bar!
    #       count the overall height of the bar using the longest segment detected in that section (detect background pixel ==> end vertical run/bar)
    #   count the height of each colored section of the bar and map to the correct category
    result = {
        hour: {"overall": 0, "top1": 0, "top2": 0, "top3": 0, "other": 0}
        for hour in HOURS
    }
    result["ymax_pixels"] = ymax

    for i in range(24):
        hour = HOURS[i]

        x_start = int(chart_left + slot_width * i)
        x_end   = int(chart_left + slot_width * (i + 1))

        column_data = []

        bottom_idx = (chart_bottom_line - chart_top_line) - 1

        # =========================================================
        # PASS 1: Analyze every column normally (same as before)
        # =========================================================
        for x in range(x_start, x_end):

            r, g, b = arr[chart_top_line + bottom_idx, x]

            if is_chart_bg(r, g, b, mode):
                continue

            bar_top_idx = bottom_idx
            gap_count = 0

            for y in range(bottom_idx, -1, -1):
                r, g, b = arr[chart_top_line + y, x]
                cat = classify_pixel(r, g, b, mode)

                if is_chart_bg(r, g, b, mode):
                    gap_count += 1
                    if gap_count > 2:
                        break
                else:
                    bar_top_idx = y
                    gap_count = 0

            bar_height = bottom_idx - bar_top_idx + 1

            if bar_height <= 0:
                continue

            breakdown = {"top1": 0, "top2": 0, "top3": 0, "other": 0}

            for y in range(bar_top_idx, bottom_idx + 1):
                r, g, b = arr[chart_top_line + y, x]
                cat = classify_pixel(r, g, b, mode)
                if cat in breakdown:
                    breakdown[cat] += 1

            column_data.append({
                "x": x,
                "height": bar_height,
                "bottom": bottom_idx,
                "breakdown": breakdown
            })

        # =========================================================
        # PASS 2: WIDTH VALIDATION (THIS IS THE FIX)
        # =========================================================
        valid_columns = []

        for j in range(len(column_data) - 2):
            c1 = column_data[j]
            c2 = column_data[j + 1]
            c3 = column_data[j + 2]

            # must be adjacent columns
            if c2["x"] == c1["x"] + 1 and c3["x"] == c2["x"] + 1:

                # must all touch the bottom (same baseline)
                if abs(c1["bottom"] - c2["bottom"]) <= 1 and abs(c2["bottom"] - c3["bottom"]) <= 1:

                    valid_columns.extend([c1, c2, c3])

        # =========================================================
        # PASS 3: PICK TALLEST VALID COLUMN
        # =========================================================
        if valid_columns:
            tallest = max(valid_columns, key=lambda c: c["height"])

            result[hour] = {
                "overall": tallest["height"],
                "top1": tallest["breakdown"]["top1"],
                "top2": tallest["breakdown"]["top2"],
                "top3": tallest["breakdown"]["top3"],
                "other": tallest["breakdown"]["other"]
            }
        else:
            result[hour] = {
                "overall": 0,
                "top1": 0,
                "top2": 0,
                "top3": 0,
                "other": 0
            }

    # ============================================================
    # DEBUG IMAGE (CLEAN BAR VISUALIZATION)
    # ============================================================
    if debug_output_path:

        debug_img = img.copy()
        debug_draw = cv2.cvtColor(np.array(debug_img), cv2.COLOR_RGB2BGR)

        # ------------------------------------------------------------
        # 1. Draw chart bounds
        # ------------------------------------------------------------
        for y in collapsed:
            cv2.line(debug_draw, (0, y), (img_w - 1, y), (180, 180, 0), 1)

        cv2.line(debug_draw, (0, chart_top_line),
                (img_w - 1, chart_top_line), (255, 0, 0), 2)

        cv2.line(debug_draw, (0, chart_bottom_line),
                (img_w - 1, chart_bottom_line), (128, 0, 128), 2)

        # ------------------------------------------------------------
        # 2. Draw hour slots
        # ------------------------------------------------------------
        for i in range(24):
            x1 = int(chart_left + slot_width * i)
            x2 = int(chart_left + slot_width * (i + 1))

            cv2.rectangle(
                debug_draw,
                (x1, chart_top_line),
                (x2, chart_bottom_line),
                (0, 140, 255),
                1
            )

        cv2.imwrite(debug_output_path, debug_draw)

    print(result)
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
        "productivity & finance": ["productivity", "finance"],
        "creativity": ["creativity"],
        "utilities": ["utilities", "utility"],
        "shopping & food": ["shopping", "food"],
        "travel": ["travel"],
        "health & fitness": ["health", "fitness"],
        "information & reading": ["information", "reading"],
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

        # detect categories in the order they appear in the line,
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
        # Remove leading junk including numbers and punctuation like 'd', '8)', '|'
        candidate = re.sub(r'^[\d\|\)\]\s\.]+', '', raw)
        # Remove common OCR prefixes  
        candidate = re.sub(r'^[a-zA-Z&]{1,2}[\.\s)\]]\s*', '', candidate)
        # Remove trailing punctuation
        candidate = re.sub(r"['\-,\.]+$", "", candidate)
        # Remove quotes
        candidate = re.sub(r"^['\"]|['\"]$", "", candidate)
        # Remove trailing numbers
        candidate = re.sub(r'\s+\d+$', '', candidate)
        return candidate.strip()

    STOP_PATTERNS = [
        r'\d{1,2}:\d{2}\s*(AM|PM)',
        r'\d+%',
        r'screen time', r'all devices', r'week day', r'yesterday',
        r'updated today', r'social', r'entertainment',
        r'education', r'shopping', r'utilities', r'limits'
    ]

    def is_stop_line(text: str) -> bool:
        tl = text.lower()
        return any(re.search(pat, tl, re.IGNORECASE) for pat in STOP_PATTERNS)

    def extract_apps_from_lines(source_lines: list) -> list:
        all_apps = []
        all_times = []
        in_most_used = False

        # 1. Collect everything in order
        for line in source_lines:
            lower = line.lower()
            
            if "screen time" in lower or "yesterday" in lower or "show today" in lower:
                in_most_used = False
                continue

            if "most used" in lower or "show categories" in lower:
                in_most_used = True
                continue
            
            if "updated today" in lower:
                # We don't break immediately because in Dark Mode, 
                # times sometimes appear AFTER this line in OCR
                continue
            if not in_most_used:
                continue

            # Try to parse as time first
            parsed_time = parse_time_fragment(line)
            if parsed_time:
                h, m = parsed_time
                if h + m > 0:
                    all_times.append(f"{h}h {m}m")
                    continue # If it's a time, it's not an app name

            # If not a time, check if it's a valid app name
            candidate = clean_candidate(line)
            name = clean_app_name(candidate)
            
            # Filter out junk like ">", "|", single letters, or stop words
            if is_valid_app_name(name) and len(name) >= 2 and not is_stop_line(line):
                all_apps.append(name)

        # 2. Matching Logic
        results = []
        
        # CASE: We found a list of apps and a list of times
        # This works for Dark Mode (Apps... then Times) 
        # and Light Mode (App, Time, App, Time) because the relative order is preserved.
        num_to_match = min(len(all_apps), len(all_times))
        
        for i in range(num_to_match):
            results.append({
                "name": all_apps[i],
                "time": all_times[i]
            })

        # If we found apps but NO times, return apps with 0m
        if not results and all_apps:
            for app in all_apps:
                results.append({"name": app, "time": "0h 0m"})

        return results
    
    def is_category_view(source_lines: list) -> bool:
        for line in source_lines:
            lower = line.lower()
            if "show apps and websites" in lower:
                return True
            if "show apps" in lower:
                return True
        return False

    # Strategy: try light_text lines first (times reliably appear on separate
    # lines there for both dark AND light mode screenshots), fall back to
    # merged lines if light_text yields nothing.
    light_lines = [l.strip() for l in light_text.split("\n") if l.strip()]

    if is_category_view(light_lines):
        top_apps = []
    else:
        top_apps = extract_apps_from_lines(light_lines)

    if not is_category_view(light_lines):
        has_no_apps = len(top_apps) == 0
        missing_all_times = all(app.get("time") == "0h 0m" for app in top_apps)

        if has_no_apps or missing_all_times:
            print("\n--- FALLING BACK TO ALL LINES ---")
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