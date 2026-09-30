import os
import io
import re
import base64

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import folium
from folium.raster_layers import ImageOverlay
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import box as shapely_box

from PIL import Image

from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from config import CLIENT_ID, CLIENT_SECRET
from sarvam_client import sarvam_translate
from geocode import geocode_place, GeocodeError

from groq_client import groq_chat, GroqError
from rag_govt import build_or_load_govt_index, govt_rag_answer

from sarvam_speech import (
    speech_to_text_bytes,
    text_to_speech_bytes,
    SarvamSpeechError,
)


# ---------------- Helpers: load secrets into environment ----------------
def load_secrets_into_env():
    for k in ("GROQ_API_KEY", "SARVAM_API_KEY", "HF_TOKEN"):
        try:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = st.secrets[k]
        except Exception:
            pass


# ---------------- Language selector (sidebar) ----------------
def language_selector():
    with st.sidebar:
        st.markdown("### 🌐 Language")
        language = st.selectbox(
            "Select language",
            [
                "English",
                "Hindi",
                "Marathi",
                "Kannada",
                "Tamil",
                "Telugu",
                "Gujarati",
                "Bengali",
                "Punjabi",
                "Malayalam",
            ],
            index=0,
        )
        st.caption("Note: We try to answer in the selected language.")

    lang_code = {
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Marathi": "mr-IN",
        "Kannada": "kn-IN",
        "Tamil": "ta-IN",
        "Telugu": "te-IN",
        "Gujarati": "gu-IN",
        "Bengali": "bn-IN",
        "Punjabi": "pa-IN",
        "Malayalam": "ml-IN",
    }
    return language, lang_code[language]


@st.cache_data(show_spinner=False)
def translate_cached(text: str, target_lang: str) -> str:
    if not text:
        return text
    if target_lang == "en-IN":
        return text
    return sarvam_translate(text=text, target_lang=target_lang, source_lang="en-IN")


def tr(text: str, target_lang: str) -> str:
    if target_lang == "en-IN":
        return text
    try:
        return translate_cached(text, target_lang)
    except Exception:
        return text


# ---------------- Crop planning context ----------------
def crop_context_ui() -> dict:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌱 Crop planning context")

        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"], index=0)
        irrigation = st.selectbox(
            "Irrigation type",
            ["Rainfed", "Drip", "Sprinkler", "Flood", "Other/Unknown"],
            index=4,
        )
        preference = st.text_input(
            "Crop preference (optional)",
            value="",
            help="Example: sugarcane / cotton / vegetables / pulses",
        )

    return {
        "season": season,
        "irrigation_type": irrigation,
        "crop_preference": preference.strip() or None,
    }


# ---------------- Groq helpers ----------------
LANG_NAME = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
    "kn-IN": "Kannada",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "gu-IN": "Gujarati",
    "bn-IN": "Bengali",
    "pa-IN": "Punjabi",
    "ml-IN": "Malayalam",
}


def groq_model_selector_ui() -> str:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 LLM (Groq)")
        groq_model = st.selectbox(
            "Choose Groq model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ],
            index=0,
        )
    st.session_state["groq_model"] = groq_model
    return groq_model


@st.cache_data(show_spinner=False)
def llm_recommendations_cached(payload: dict, target_code: str, model: str) -> str:
    lang = LANG_NAME.get(target_code, "English")
    prompt = f"""
You are an agriculture advisory assistant for Indian farmers.

Task:
- Suggest crops to plant AND give zone-wise management actions based on NDVI/NDWI micro-zones. 
- You are a soil nutrient expert. Tell about what nutrients are lacking in the soil and what action needs to be taken.

Rules:
- Use ONLY the data in INPUT.
- Do not claim exact soil type/rainfall/prices/yields.
- Output must be in {lang}.

INPUT (JSON):
{payload}
"""
    return groq_chat(prompt, model=model)


# ---------------- Govt RAG index ----------------
@st.cache_resource(show_spinner=False)
def govt_index_cached():
    return build_or_load_govt_index(pdf_dir="data/govt_pdfs", persist_dir="storage/govt_index")


# ---------------- Sarvam cached wrappers ----------------
@st.cache_data(show_spinner=False)
def stt_cached(audio_bytes: bytes, filename: str, mode: str) -> str:
    return speech_to_text_bytes(
        audio_bytes=audio_bytes,
        filename=filename or "audio.wav",
        model="saaras:v3",
        mode=mode,
        debug=False,
    )


@st.cache_data(show_spinner=False)
def tts_cached(text: str, target_language_code: str, debug: bool) -> bytes:
    return text_to_speech_bytes(
        text=text,
        target_language_code=target_language_code,
        debug=debug,
    )


# ---------------- Place search (sidebar) ----------------
@st.cache_data(show_spinner=False)
def geocode_cached(query: str) -> dict:
    return geocode_place(query, country_codes="in")


def place_search_ui():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📍 Place search (India)")
        place_query = st.text_input(
            "Type place name",
            value=st.session_state.get("place_query", "Baramati, Pune"),
        )
        st.session_state["place_query"] = place_query

        st.session_state.setdefault("place", None)

        col_a, col_b = st.columns([1, 1])
        find = col_a.button("🔎 Find", use_container_width=True)
        clear = col_b.button("🧹 Clear", use_container_width=True)

        if clear:
            st.session_state["place"] = None

        if find:
            try:
                place = geocode_cached(place_query)
                st.session_state["place"] = place
                st.success(f"Found: {place['display_name']}")
            except GeocodeError as e:
                st.session_state["place"] = None
                st.error(str(e))


# ---------------- Polygon drawing helpers ----------------
def extract_polygon_from_folium(st_folium_output: dict) -> list[list[float]] | None:
    if not st_folium_output:
        return None
    feat = st_folium_output.get("last_active_drawing")
    if not feat:
        return None
    geom = feat.get("geometry") or {}
    if geom.get("type") != "Polygon":
        return None
    coords = geom.get("coordinates")
    if not coords or not coords[0]:
        return None
    ring = coords[0]
    return [[float(lat), float(lon)] for lon, lat in ring]


def polygon_to_bbox(poly_latlon: list[list[float]]) -> tuple[float, float, float, float]:
    lats = [p[0] for p in poly_latlon]
    lons = [p[1] for p in poly_latlon]
    return min(lons), min(lats), max(lons), max(lats)


# ---------------- India-only validation ----------------
INDIA_BBOX = shapely_box(68.0, 6.0, 97.0, 36.0)


def ensure_aoi_in_india(min_lon, min_lat, max_lon, max_lat) -> bool:
    return INDIA_BBOX.intersects(shapely_box(min_lon, min_lat, max_lon, max_lat))


# ---------------- Sentinel Hub config ----------------
def get_sh_config():
    config = SHConfig()
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return config


# ---------------- EvalScripts ----------------
EVALSCRIPT_NDVI_SCL = """
//VERSION=3
function setup() {
  return { input: [{ bands: ["B04", "B08", "SCL"] }], output: { bands: 2, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi, sample.SCL];
}
"""

EVALSCRIPT_NDWI_SCL = """
//VERSION=3
function setup() {
  return { input: [{ bands: ["B03", "B08", "SCL"] }], output: { bands: 2, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
  return [ndwi, sample.SCL];
}
"""


@st.cache_data(show_spinner=False)
def fetch_index(evalscript: str, min_lon, min_lat, max_lon, max_lat, start_date, end_date, maxcc, resolution=10):
    config = get_sh_config()
    bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                time_interval=(start_date, end_date),
                maxcc=maxcc / 100.0,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    data = req.get_data()[0]
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[2] == 1:
        return data[:, :, 0]
    return data


def apply_scl_mask(index: np.ndarray, scl: np.ndarray, bad_classes={0, 1, 2, 3, 8, 9, 10, 11}) -> np.ndarray:
    masked = index.copy()
    masked[np.isin(scl, list(bad_classes))] = np.nan
    return masked


def split_index_and_scl(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"Expected (H,W,2) array, got shape={arr.shape}")
    return arr[:, :, 0].astype(np.float32), arr[:, :, 1].astype(np.int16)


def valid_pixel_percent(*arrays: np.ndarray) -> float:
    if not arrays:
        return 0.0
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return float(mask.mean() * 100.0)


def stats(arr):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def make_zones_from_two_features(ndvi, ndwi, k=3):
    h, w = ndvi.shape
    X = np.stack([ndvi.reshape(-1), ndwi.reshape(-1)], axis=1)
    valid_mask = np.isfinite(X).all(axis=1)
    valid = X[valid_mask]
    if valid.shape[0] == 0:
        raise ValueError("No valid pixels after masking. Try different dates/cloud cover.")

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = np.full(X.shape[0], -1, dtype=int)
    labels[valid_mask] = km.fit_predict(valid)

    centers = km.cluster_centers_
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap.get(x, -1) if x != -1 else -1 for x in labels], dtype=int)
    return labels.reshape(h, w), centers[order]


def zone_stats(zones):
    total = zones.size
    return {
        "Poor": float(np.sum(zones == 0) / total * 100.0),
        "Medium": float(np.sum(zones == 1) / total * 100.0),
        "Good": float(np.sum(zones == 2) / total * 100.0),
    }


def zones_to_png_data_url(zones: np.ndarray, alpha: float = 0.55) -> str:
    h, w = zones.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    poor = (220, 20, 60, int(alpha * 255))
    medium = (255, 193, 7, int(alpha * 255))
    good = (46, 125, 50, int(alpha * 255))

    rgba[zones == 0] = poor
    rgba[zones == 1] = medium
    rgba[zones == 2] = good

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def add_zone_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="
        position: fixed; bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(255,255,255,0.92);
        padding: 10px 12px; border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.15);
        font-family: Arial; font-size: 12px;">
      <b>Zone Legend</b><br>
      <div style="margin-top:6px;">
        <span style="display:inline-block;width:12px;height:12px;background:#dc143c;border-radius:3px;"></span> Poor
      </div>
      <div>
        <span style="display:inline-block;width:12px;height:12px;background:#ffc107;border-radius:3px;"></span> Medium
      </div>
      <div>
        <span style="display:inline-block;width:12px;height:12px;background:#2e7d32;border-radius:3px;"></span> Good
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def inject_css():
    st.markdown(
        """
        <style>
        .app-title {font-size: 2.0rem; font-weight: 750; margin-bottom: 0.25rem;}
        .app-subtitle {opacity: 0.85; margin-top: 0; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header():
    st.markdown('<div class="app-title">🌾 Hyperlocal Field Zoning (MVP+)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Micro-zones + AI crop planning + Govt schemes (RAG) + Voice</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Field Zoning MVP+", layout="wide")
    load_secrets_into_env()
    inject_css()

    with st.sidebar:
        st.caption(f"GROQ key set: {bool(os.getenv('GROQ_API_KEY'))}")
        st.caption(f"SARVAM key set: {bool(os.getenv('SARVAM_API_KEY'))}")
        tts_debug = st.toggle("TTS_DEBUG (show Sarvam TTS response)", value=False)

    st.session_state.setdefault("field_polygon", None)
    st.session_state.setdefault("last_rag_answer", None)
    st.session_state.setdefault("last_rag_sources", None)
    st.session_state.setdefault("last_transcript", None)
    st.session_state.setdefault("last_tts_audio", None)

    language, target_code = language_selector()
    groq_model = groq_model_selector_ui()
    crop_ctx = crop_context_ui()
    place_search_ui()

    header()

    left, right = st.columns([0.38, 0.62], gap="large")

    with left:
        st.markdown("### Field & dates (India only)")

        default_min_lon = 74.566782
        default_min_lat = 18.140663
        default_max_lon = 74.586782
        default_max_lat = 18.160663

        if st.session_state.get("field_polygon"):
            default_min_lon, default_min_lat, default_max_lon, default_max_lat = polygon_to_bbox(
                st.session_state["field_polygon"]
            )
        elif st.session_state.get("place"):
            lat0 = st.session_state["place"]["lat"]
            lon0 = st.session_state["place"]["lon"]
            dlat = 1.0 / 111.0
            dlon = 1.0 / (111.0 * np.cos(np.deg2rad(lat0)))
            default_min_lat = lat0 - dlat
            default_max_lat = lat0 + dlat
            default_min_lon = lon0 - dlon
            default_max_lon = lon0 + dlon

        min_lon = st.number_input("Min Lon", value=float(default_min_lon), format="%.6f")
        min_lat = st.number_input("Min Lat", value=float(default_min_lat), format="%.6f")
        max_lon = st.number_input("Max Lon", value=float(default_max_lon), format="%.6f")
        max_lat = st.number_input("Max Lat", value=float(default_max_lat), format="%.6f")

        c1, c2 = st.columns(2)
        start_date = c1.text_input("Start (YYYY-MM-DD)", "2025-10-01")
        end_date = c2.text_input("End (YYYY-MM-DD)", "2025-11-30")

        maxcc = st.slider("Max cloud cover (%)", 0, 100, 60)
        resolution = st.selectbox("Resolution (m/pixel)", [10, 20, 30], index=0)

        run = st.button("🚀 Analyze field", type="primary", use_container_width=True)

    st.markdown("---")
    tabs = st.tabs(
        ["📈 NDVI", "💧 NDWI", "🧩 Zones", f"✅ Crop plan ({language})", "🏛️ Govt Help (RAG + Voice Note)"]
    )

    if run:
        if not ensure_aoi_in_india(min_lon, min_lat, max_lon, max_lat):
            st.error("This demo is restricted to India. Please choose a field inside India.")
            return

        with st.spinner("Downloading Sentinel‑2 and computing NDVI + NDWI..."):
            ndvi_raw = fetch_index(
                EVALSCRIPT_NDVI_SCL,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                start_date,
                end_date,
                maxcc,
                resolution=resolution,
            )
            ndwi_raw = fetch_index(
                EVALSCRIPT_NDWI_SCL,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                start_date,
                end_date,
                maxcc,
                resolution=resolution,
            )

        ndvi, scl1 = split_index_and_scl(ndvi_raw)
        ndwi, scl2 = split_index_and_scl(ndwi_raw)
        ndvi = apply_scl_mask(ndvi, scl1)
        ndwi = apply_scl_mask(ndwi, scl2)

        quality_pct = valid_pixel_percent(ndvi, ndwi)
        ndvi_s = stats(ndvi)
        ndwi_s = stats(ndwi)

        zones, centers = make_zones_from_two_features(ndvi, ndwi, k=3)
        zs = zone_stats(zones)

        st.session_state["last_zones"] = zones
        st.session_state["last_bounds"] = [[min_lat, min_lon], [max_lat, max_lon]]

        place = st.session_state.get("place") or {}
        place_name = place.get("display_name")

        payload = {
            "location": place_name,
            "aoi_center": {"lat": float((min_lat + max_lat) / 2), "lon": float((min_lon + max_lon) / 2)},
            "season": crop_ctx["season"],
            "irrigation_type": crop_ctx["irrigation_type"],
            "crop_preference": crop_ctx["crop_preference"],
            "date_range": {"start": start_date, "end": end_date},
            "resolution_m": resolution,
            "cloud_limit_percent": maxcc,
            "data_quality_valid_pixels_percent": quality_pct,
            "zone_area_percent": zs,
            "cluster_centers_ordered_worst_to_best": np.round(centers, 3).tolist(),
            "ndvi_summary": ndvi_s,
            "ndwi_summary": ndwi_s,
        }
        st.session_state["last_payload"] = payload

        with tabs[0]:
            st.caption(f"Valid pixels after cloud mask: {quality_pct:.1f}%")
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDVI")
            st.pyplot(fig)

        with tabs[1]:
            st.caption(f"Valid pixels after cloud mask: {quality_pct:.1f}%")
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(ndwi, cmap="PuBuGn", vmin=-1.0, vmax=1.0)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDWI")
            st.pyplot(fig)

        with tabs[2]:
            c = st.columns(3)
            c[0].metric("Poor", f"{zs['Poor']:.1f}%")
            c[1].metric("Medium", f"{zs['Medium']:.1f}%")
            c[2].metric("Good", f"{zs['Good']:.1f}%")

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(zones, cmap="viridis", vmin=0, vmax=2)
            ax.axis("off")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_ticks([0, 1, 2])
            cb.set_ticklabels(["Poor", "Medium", "Good"])
            st.pyplot(fig)

        with tabs[3]:
            st.markdown("### ✅ Crop plan + actions (Groq + Llama)")
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is not set.")
            else:
                with st.spinner("Generating crop plan..."):
                    try:
                        st.markdown(llm_recommendations_cached(payload, target_code, groq_model))
                    except GroqError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"LLM error: {e}")

    with tabs[4]:
        st.markdown("### 🏛️ Government Schemes & Policies (RAG + Voice Note)")
        st.caption("Record a voice note → transcribe → RAG → voice answer.")

        col1, col2 = st.columns([0.55, 0.45], gap="large")

        with col1:
            stt_mode = st.selectbox(
                "STT mode",
                ["transcribe", "translate", "verbatim", "translit", "codemix"],
                index=0,
                key="govt_stt_mode",
            )
            audio_file = st.audio_input("Tap to record (voice note)", sample_rate=16000, key="govt_audio_input")
            top_k = st.slider("Number of sources", 3, 10, 5, key="govt_topk")
            run_voice = st.button("Answer my voice note", type="primary", use_container_width=True, key="govt_run_voice")

        with col2:
            if st.session_state.get("last_transcript"):
                st.markdown("**Transcript:**")
                st.write(st.session_state["last_transcript"])

            if st.session_state.get("last_rag_answer"):
                st.markdown("**Answer (with citations):**")
                st.markdown(st.session_state["last_rag_answer"])

            if st.session_state.get("last_rag_sources"):
                st.markdown("**Sources used:**")
                for s in st.session_state["last_rag_sources"]:
                    st.markdown(f"- **{s['id']}**: `{s['source']}` page={s['page']}")

            if st.session_state.get("last_tts_audio"):
                st.markdown("**Voice answer:**")
                st.audio(st.session_state["last_tts_audio"], format="audio/wav")

        if run_voice:
            if not os.getenv("SARVAM_API_KEY"):
                st.error("SARVAM_API_KEY is not set (needed for STT/TTS).")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is not set (needed for RAG answer).")
            elif audio_file is None:
                st.warning("Please record a voice note first.")
            else:
                try:
                    audio_bytes = audio_file.getvalue()

                    with st.spinner("1/3 Transcribing..."):
                        transcript = stt_cached(audio_bytes, getattr(audio_file, "name", "voice.wav"), stt_mode)
                        st.session_state["last_transcript"] = transcript

                    with st.spinner("2/3 RAG answering..."):
                        index = govt_index_cached()
                        result = govt_rag_answer(transcript, index=index, model=groq_model, top_k=top_k)
                        st.session_state["last_rag_answer"] = result["answer"]
                        st.session_state["last_rag_sources"] = result["sources"]

                    with st.spinner("3/3 TTS..."):
                        answer_text = st.session_state["last_rag_answer"]
                        if target_code != "en-IN":
                            answer_text = tr(answer_text, target_code)
                        speech_text = re.sub(r"\[S\d+\]", "", answer_text).strip()
                        st.session_state["last_tts_audio"] = tts_cached(speech_text, target_code, debug=tts_debug)

                    st.rerun()

                except (SarvamSpeechError, GroqError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Voice RAG error: {e}")

    with right:
        st.markdown("### Field preview (draw your field polygon)")

        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="CartoDB positron")

        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "rectangle": True,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        folium.Rectangle(bounds=bounds, color="#2E7D32", fill=True, fill_opacity=0.10).add_to(m)

        if st.session_state.get("last_zones") is not None and st.session_state.get("last_bounds") is not None:
            zones_overlay = st.session_state["last_zones"]
            overlay_bounds = st.session_state["last_bounds"]
            img_url = zones_to_png_data_url(zones_overlay, alpha=0.55)

            ImageOverlay(
                image=img_url,
                bounds=overlay_bounds,
                opacity=0.85,
                name="Micro-zones",
                interactive=False,
                cross_origin=False,
                zindex=2,
            ).add_to(m)

            add_zone_legend(m)
            folium.LayerControl(collapsed=True).add_to(m)

        map_out = st_folium(m, height=420, width=None)

        poly = extract_polygon_from_folium(map_out)
        if poly:
            st.session_state["field_polygon"] = poly

        if st.session_state.get("field_polygon"):
            st.success("Field boundary captured ✅")
        else:
            st.info("Draw your field boundary (polygon) on the map.")


if __name__ == "__main__":
    main()