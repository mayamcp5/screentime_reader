import re
from src.utils import ocr_image
from src.ios.chart_utils import extract_simple_hourly_chart, normalize_hourly

# ================================
# COLOR DETECTION (BLUE)
# ================================
def is_pickup_bar(r, g, b):
    return (
        r < 120 and
        150 <= g <= 230 and
        b > 200
    )

# ================================
# TOTAL EXTRACTION
# ================================
def extract_total_pickups(lines):
    nums = []

    for line in lines:
        if ":" in line:
            continue

        matches = re.findall(r"\d+", line)
        for m in matches:
            nums.append(int(m))

    return max(nums) if nums else 0


# ================================
# MAIN FUNCTION
# ================================
def process_ios_pickups_screenshot(image_path: str) -> dict:

    text = ocr_image(image_path)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    total = extract_total_pickups(lines)

    hourly_pixels = extract_simple_hourly_chart(
        image_path,
        is_pickup_bar,
        debug_output_path="debug_pickups.png"
    )

    ymax = hourly_pixels.pop("ymax_pixels", 0)

    hourly = normalize_hourly(hourly_pixels, total, ymax)

    return {
        "total_pickups": total,
        "ymax_pixels": ymax,
        "hourly_pickups": hourly
    }