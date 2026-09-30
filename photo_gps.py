from __future__ import annotations

from typing import Optional, Tuple
import io
import exifread


def _ratio_to_float(r) -> float:
    return float(r.num) / float(r.den)


def _dms_to_deg(dms, ref: str) -> float:
    deg = _ratio_to_float(dms[0])
    minutes = _ratio_to_float(dms[1])
    seconds = _ratio_to_float(dms[2])
    value = deg + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ("S", "W"):
        value = -value
    return value


def extract_gps_from_image_bytes(image_bytes: bytes) -> Optional[Tuple[float, float]]:
    """
    Returns (lat, lon) if GPS tags exist, else None.
    """
    bio = io.BytesIO(image_bytes)
    tags = exifread.process_file(bio, details=False)

    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")

    if not (lat_tag and lat_ref and lon_tag and lon_ref):
        return None

    lat = _dms_to_deg(lat_tag.values, str(lat_ref.values))
    lon = _dms_to_deg(lon_tag.values, str(lon_ref.values))
    return lat, lon