import time
import requests

class GeocodeError(RuntimeError):
    pass

_LAST_CALL_TS = 0.0

def geocode_place(query: str, country_codes: str = "in", timeout_s: int = 20) -> dict:
    """
    Nominatim geocoder with:
    - strong User-Agent + From header
    - 1 request/sec throttle (OSM policy)
    - India-only restriction
    """
    global _LAST_CALL_TS

    q = (query or "").strip()
    if not q:
        raise GeocodeError("Place name is empty.")

    # Throttle to 1 request/second
    now = time.time()
    wait = 1.1 - (now - _LAST_CALL_TS)
    if wait > 0:
        time.sleep(wait)

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": q,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": country_codes,
    }

    # IMPORTANT: Provide a descriptive UA + a real contact email (policy requirement)
    headers = {
        "User-Agent": "farm-zoning-ui/1.0 (ShreyaA2105; contact: ammalajerishreya@gmail.com)",
        "From": "ammalajerishreya@gmail.com",
        "Accept-Language": "en",
    }

    r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
    _LAST_CALL_TS = time.time()

    if r.status_code != 200:
        raise GeocodeError(f"Nominatim error HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    if not data:
        raise GeocodeError("No results found. Try adding district/state, e.g., 'Mandya, Karnataka'.")

    item = data[0]
    lat = float(item["lat"])
    lon = float(item["lon"])
    bb = item.get("boundingbox")  # [south, north, west, east]
    bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])] if bb else None

    return {
        "display_name": item.get("display_name", q),
        "lat": lat,
        "lon": lon,
        "bbox": bbox,
    }