# Satellite Farm AI

An AI-powered agricultural field analysis system that combines Sentinel-2 satellite imagery, vegetation indices, machine learning, RAG, and multilingual voice AI to provide field-level agricultural insights.

## Features

- 🛰️ Sentinel-2 satellite imagery analysis
- 🌱 NDVI and NDWI computation
- 🗺️ K-Means based agricultural field micro-zoning
- 📊 Zone-wise vegetation and water analysis
- 📚 RAG-based agricultural and government-scheme information retrieval
- 🤖 Groq LLM-powered agricultural recommendations
- 🎙️ Multilingual speech-to-text and text-to-speech
- 📍 Location-based field selection and geospatial visualization

## Tech Stack

**Python · Streamlit · Sentinel Hub · NumPy · Scikit-learn · LlamaIndex · Chroma · Groq · Sarvam AI · Folium**

## Workflow

```text
Field Location
      ↓
Sentinel-2 Satellite Data
      ↓
Quality & Cloud Masking
      ↓
NDVI / NDWI Analysis
      ↓
K-Means Field Zoning
      ↓
Zone-wise Insights
      ↓
RAG + LLM
      ↓
Agricultural Recommendations
```

## Run Locally

```bash
git clone https://github.com/Shreya21052004/satellite-farm-ai.git
cd satellite-farm-ai

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Configuration

Configure the required API credentials using environment variables or Streamlit secrets:

```text
SENTINEL_CLIENT_ID
SENTINEL_CLIENT_SECRET
GROQ_API_KEY
SARVAM_API_KEY
```
