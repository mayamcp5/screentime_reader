import re
from src.utils import ocr_image
from src.ios.chart_utils import extract_simple_hourly_chart, normalize_hourly

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
# TOTAL EXTRACTION
# ================================
def extract_total_notifications(lines):
    nums = []

    for line in lines:
        if ":" in line:
            continue  # skip times

        matches = re.findall(r"\d+", line)
        for m in matches:
            nums.append(int(m))

    return max(nums) if nums else 0


# ================================
# MAIN FUNCTION
# ================================
def process_ios_notifications_screenshot(image_path: str) -> dict:

    text = ocr_image(image_path)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    total = extract_total_notifications(lines)

    hourly_pixels = extract_simple_hourly_chart(
        image_path,
        is_notification_bar,
        debug_output_path="debug_notifications.png"
    )

    ymax = hourly_pixels.pop("ymax_pixels", 0)

    hourly = normalize_hourly(hourly_pixels, total, ymax)

    return {
        "total_notifications": total,
        "ymax_pixels": ymax,
        "hourly_notifications": hourly
    }