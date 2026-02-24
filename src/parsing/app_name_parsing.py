import re

def clean_app_name(text: str) -> str:
    # Remove leading/trailing non-alphanumeric
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"[^A-Za-z0-9]+$", "", text)

    # Remove leading single letter followed by symbol (e.g., "E) Discord" -> "Discord")
    text = re.sub(r"^[A-Za-z](?=[^A-Za-z0-9\s])\s*", "", text)

    # Remove OCR artifacts: symbols like £©®(){}[]•◆►→ etc.
    text = re.sub(r"[£©®(){}[\]•◆►→]+", "", text)

    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_valid_app_name(name: str) -> bool:
    name = name.strip()

    # Too short
    if len(name) < 2:
        return False

    # Reject pure digits
    if name.isdigit():
        return False

    # Reject anything containing time patterns OR numbers mixed with symbols
    if re.search(r'\d', name):
        return False  # catch all numbers, not just "3h" or "5m"

    # Reject common OCR misreads (single letters with symbols)
    if re.match(r'^[A-Z]{1,2}[\)\]]?$', name):
        return False

    # Must contain at least 2 letters
    if not re.search(r'[A-Za-z].*[A-Za-z]', name):
        return False

    # Reject if too many non-letter characters
    non_letter_ratio = len(re.findall(r'[^A-Za-z\s]', name)) / max(1, len(name))
    if non_letter_ratio > 0.3:  # more than 30% garbage
        return False

    return True