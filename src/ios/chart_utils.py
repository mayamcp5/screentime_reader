import numpy as np
from PIL import Image
import cv2

HOURS = [f"{h if h!=0 else 12}am" if h < 12 else f"{h-12 if h>12 else 12}pm" for h in range(24)]

# ================================
# GRIDLINE DETECTION
# ================================
def is_gridline_pixel(r, g, b, mode='dark'):
    r, g, b = int(r), int(g), int(b)

    if abs(r - g) > 8 or abs(g - b) > 8:
        return False

    brightness = (r + g + b) / 3

    if mode == 'light':
        return 200 <= brightness <= 225
    else:
        return 40 <= brightness <= 120


# ================================
# CORE SIMPLE CHART EXTRACTION
# ================================
def extract_simple_hourly_chart(image_path, is_bar_pixel_fn, debug_output_path=None):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    img_h, img_w = arr.shape[:2]

    # --- Detect light vs dark mode ---
    sample = arr[100:200, 100:200]
    mode = 'light' if np.mean(sample) > 127 else 'dark'

    # ============================================================
    # STEP 1: Detect horizontal gridlines
    # ============================================================
    gridlines = []

    for y in range(img_h):
        row = arr[y]
        gray_count = sum(is_gridline_pixel(*px, mode) for px in row)

        if gray_count > 0.15 * img_w:
            gridlines.append(y)

    # Collapse adjacent rows
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
    # STEP 2: Find chart bounds (5 evenly spaced lines)
    # ============================================================
    chart_top = None
    chart_bottom = None

    for i in range(len(collapsed) - 4):
        candidate = collapsed[i:i+5]

        spacings = [candidate[j+1] - candidate[j] for j in range(4)]
        avg = sum(spacings) / len(spacings)

        if all(abs(s - avg) < avg * 0.35 for s in spacings):
            chart_top = candidate[0]
            chart_bottom = candidate[-1]

    if chart_top is None:
        return {}

    ymax = chart_bottom - chart_top

    # ============================================================
    # STEP 3: Find vertical bounds
    # ============================================================
    def find_axis(left=True):
        xs = range(img_w) if left else range(img_w-1, -1, -1)

        for x in xs:
            col = arr[chart_top:chart_bottom, x]
            count = sum(is_gridline_pixel(*px, mode) for px in col)

            if count > 0.35 * (chart_bottom - chart_top):
                return x

        return None

    chart_left = find_axis(True)
    chart_right = find_axis(False)

    if chart_left is None or chart_right is None:
        return {}

    slot_width = (chart_right - chart_left) / 24.0

    result = {hour: 0 for hour in HOURS}
    result["ymax_pixels"] = ymax

    # ============================================================
    # STEP 4: Scan bars
    # ============================================================
    for i in range(24):
        hour = HOURS[i]

        x_start = int(chart_left + slot_width * i)
        x_end   = int(chart_left + slot_width * (i + 1))

        best_height = 0

        for x in range(x_start, x_end):

            r, g, b = arr[chart_bottom - 1, x]

            if not is_bar_pixel_fn(r, g, b):
                continue

            top_y = chart_bottom - 1
            gap = 0

            for y in range(chart_bottom - 1, chart_top, -1):
                r, g, b = arr[y, x]

                if is_bar_pixel_fn(r, g, b):
                    top_y = y
                    gap = 0
                else:
                    gap += 1
                    if gap > 3:
                        break

            height = (chart_bottom - 1) - top_y + 1

            # remove noise
            if height < 2:
                height = 0

            best_height = max(best_height, height)

        result[hour] = best_height

    # ============================================================
    # DEBUG
    # ============================================================
    if debug_output_path:
        debug = cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)

        cv2.line(debug, (0, chart_top), (img_w, chart_top), (255, 0, 0), 2)
        cv2.line(debug, (0, chart_bottom), (img_w, chart_bottom), (0, 0, 255), 2)

        for i in range(24):
            x = int(chart_left + slot_width * i)
            cv2.line(debug, (x, chart_top), (x, chart_bottom), (0, 255, 255), 1)

        cv2.imwrite(debug_output_path, debug)

    return result


# ================================
# NORMALIZATION
# ================================
def normalize_hourly(hourly_pixels, total, ymax):
    normalized = {}

    for hour, px in hourly_pixels.items():
        if hour == "ymax_pixels":
            continue

        if ymax > 0:
            val = round((px / ymax) * total)
        else:
            val = 0

        normalized[hour] = val

    return normalized