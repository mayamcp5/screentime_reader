# src/ios/activity.py

import re
from src.utils import ocr_image
from src.parsing.time_parsing import parse_time_fragment
from src.parsing.app_name_parsing import clean_app_name, is_valid_app_name


def process_ios_category_screenshot(image_path: str, include_seconds=False):
    from src.ios.overall import preprocess_for_ocr
    from PIL import Image
    import numpy as np
    
    # Detect dark vs light mode by sampling background
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    sample = arr[100:200, 100:200]  # Sample a region
    avg_brightness = np.mean(sample)
    
    is_dark_mode = avg_brightness < 127
    
    print(f"Detected mode: {'DARK' if is_dark_mode else 'LIGHT'} (brightness: {avg_brightness:.1f})")
    
    # Use appropriate preprocessing
    if is_dark_mode:
        # Dark mode - use light_text preprocessing for white text
        text = ocr_image(preprocess_for_ocr(image_path, light_text=True))
    else:
        # Light mode - use normal preprocessing for dark text
        text = ocr_image(image_path)
    
    print("\n================ RAW OCR TEXT ================\n")
    print(text)
    print("\n=============================================\n")
    
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    category = None
    total_time = "0h 0m"
    apps = []

    # ======================================================
    # 1️⃣ Detect category
    # ======================================================
    for line in lines:
        lower = line.lower()
        if "entertainment" in lower:
            category = "Entertainment"
        if "social" in lower:
            category = "Social"
        if not category:
            category = "Cut-off Category"

    # ======================================================
    # 2️⃣ Extract TOTAL TIME
    # ======================================================
    in_total_section = False
    for line in lines:
        lower = line.lower()
        if "screen time" in lower:
            in_total_section = True
            continue
        if "apps & websites" in lower:
            break
        if not in_total_section or "daily" in lower:
            continue

        line = line.replace("Th", "1h").replace("th", "1h").replace("lh", "1h").replace("Ih", "1h")
        parsed = parse_time_fragment(line)
        if parsed:
            h, m = parsed
            if h > 0 or m > 0:
                total_time = f"{h}h {m}m"
                break

    # ======================================================
    # 3️⃣ Slice APPS section and TIMES section cleanly
    # ======================================================
    apps_section, times_section = [], []
    in_apps = False
    in_times = False
    for line in lines:
        lower = line.lower()
        if "apps & websites" in lower:
            in_apps = True
            continue
        if "limits" in lower:
            in_apps = False
            in_times = True
            continue
        if in_apps:
            apps_section.append(line)
        elif in_times:
            times_section.append(line)

    # ======================================================
    # 4️⃣ Detect layout type
    # ======================================================
    # Check if ANY line has BOTH a name AND time on same line
    same_line_layout = False
    for line in apps_section:
        has_time = parse_time_fragment(line) is not None
        # Remove time and check if substantial text remains
        line_without_time = re.sub(r'\d+h|\d+m|\d+s', '', line).strip()
        has_name = len(line_without_time) >= 3
        
        if has_time and has_name:
            same_line_layout = True
            break

    # ======================================================
    # 5️⃣ Parse SAME-LINE layout
    # ======================================================
    if same_line_layout:
        for line in apps_section:
            line = line.replace("Th", "1h").replace("th", "1h").replace("lh", "1h").replace("Ih", "1h")

            # Extract time from line
            time_match = re.search(r'(\d+)h|(\d+)m|(\d+)s', line)
            h = m = s = 0
            if time_match:
                h = int(time_match.group(1) or 0)
                m = int(time_match.group(2) or 0)
                s = int(time_match.group(3) or 0)

            if not include_seconds and h == 0 and m == 0:
                continue  # skip apps under 1 min

            # Clean app name
            name_part = re.sub(r'(\d+h|\d+m|\d+s)', '', line)  # remove time
            name_part = re.sub(r'^[^a-zA-Z]+', '', name_part)  # remove junk prefix
            # Clean app name, but only remove junk up to first capital letter if there is one
            if any(c.isupper() for c in name_part):
                # Strip everything before first capital letter
                first_cap_idx = next(i for i, c in enumerate(name_part) if c.isupper())
                name_part = name_part[first_cap_idx:]
            else:
                # Keep lowercase-starting names as-is
                name_part = name_part.strip()
            name_part = clean_app_name(name_part)

            if not is_valid_app_name(name_part):
                continue

            apps.append({
                "name": name_part,
                "time": f"{h}h {m}m" if h+m > 0 else f"{s}s"
            })

    # ======================================================
    # 6️⃣ Parse SPLIT layout
    # ======================================================
    else:
        # Collect names AND times from apps_section
        names = []
        times = []
        
        for line in apps_section:
            line = line.replace("Th", "1h").replace("lh", "1h").replace("Ih", "1h")
            
            # Check if this is a time line
            parsed = parse_time_fragment(line)
            if parsed:
                h, m = parsed
                if include_seconds or h > 0 or m > 0:
                    times.append((h, m))
                print(f"  Time line: {repr(line)} → ({h}h {m}m)")
                continue
            
            # Not a time - must be app name
            original = line
            cleaned = re.sub(r'^[^a-zA-Z]+', '', line)
            name = clean_app_name(cleaned)
            valid = is_valid_app_name(name)
            print(f"  Name line: {repr(original)} → {repr(name)} → valid: {valid}")
            
            if valid and len(name) >= 3:
                names.append(name)

        # Also collect times from times_section if it exists
        for line in times_section:
            parsed = parse_time_fragment(line)
            if not parsed:
                continue
            h, m = parsed
            if not include_seconds and h == 0 and m == 0:
                continue
            times.append((h, m))

        # If more times than names, skip first time (likely Daily Average)
        if len(times) > len(names):
            print(f"  Skipping first time (Daily Average): {times[0]}")
            times = times[1:]

        for name, (h, m) in zip(names, times):
            apps.append({
                "name": name,
                "time": f"{h}h {m}m"
            })

    # ======================================================
    # 7️⃣ Sort apps descending by minutes
    # ======================================================
    apps.sort(
        key=lambda x: int(x["time"].split("h")[0]) * 60 +
                      int(x["time"].split("h")[1].replace("m", "").strip()),
        reverse=True
    )

    return {
        "category": category,
        "total_time": total_time,
        "apps": apps
    }