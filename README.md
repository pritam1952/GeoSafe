<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0.3-EC6C00?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-5.22.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge"/>

<br/><br/>

# 🚦 GeoSafe — Road Accident Hotspot Prediction System

**An AI-powered platform that detects accident hotspots across Indian roads and predicts crash severity before incidents happen.**

<br/>

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-geosafe.streamlit.app-FF4B4B?style=for-the-badge)](https://geosafe.streamlit.app)

<br/>

[📓 Notebooks](./notebooks) &nbsp;·&nbsp;
[📊 Dataset](./data/indian_roads_dataset.csv) &nbsp;·&nbsp;
[🐛 Report Bug](https://github.com/pritam1952/GeoSafe/issues) &nbsp;·&nbsp;
[⭐ Star this repo](https://github.com/pritam1952/GeoSafe)

</div>

---

## 📌 What is GeoSafe?

GeoSafe is an end-to-end machine learning application built for **road safety intelligence in India**. It combines geospatial clustering and gradient-boosted classification to answer two core questions:

- 🗺️ **Where** are accidents most likely to happen? → Hotspot Map
- ⚠️ **How severe** will an accident be given current conditions? → Severity Predictor

Built as a portfolio/capstone project using Indian road accident data.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏠 **Dashboard** | Live KPI cards, severity breakdown, city rankings, hourly trends, weather distribution |
| 🗺️ **Hotspot Map** | Interactive Folium map on dark tiles with clustered risk zones (High / Medium / Low) |
| ⚠️ **Severity Predictor** | Input road + weather + time → get Fatal / Major / Minor prediction with confidence gauge |
| 📊 **Plotly Charts** | Interactive bar, line, donut, and gauge charts throughout |
| 🎨 **Professional UI** | Custom dark-navy + signal-amber design system with Inter & JetBrains Mono fonts |

---

## 🧠 ML Pipeline

```
Raw Data  →  Preprocessing  →  EDA & Feature Engineering
    →  DBSCAN Hotspot Detection  →  XGBoost Severity Model  →  Streamlit App
```

### Model Details
- **Algorithm:** XGBoost Classifier
- **Target:** Accident Severity (`fatal` / `major` / `minor`)
- **Features (17):** lanes, traffic signal, temperature, peak hour flag, night flag, weekend flag, hour, city, state, road type, weather, visibility, traffic density, cause, day of week, vehicles involved, casualties
- **Hotspot Detection:** DBSCAN (Density-Based Spatial Clustering) on lat/lon coordinates

---

## 🗂️ Project Structure

```
GeoSafe/
│
├── 📁 data/
│   ├── indian_roads_dataset.csv      ← Raw dataset
│   ├── cleaned_accidents.csv         ← Preprocessed data
│   ├── hotspots.csv                  ← DBSCAN cluster output
│   ├── severity_model.pkl            ← Trained XGBoost model
│   ├── label_encoders.pkl            ← Categorical encoders
│   └── target_encoder.pkl            ← Target label encoder
│
├── 📁 notebooks/
│   ├── EDA.ipynb                     ← Exploratory Data Analysis
│   ├── Hotspot_Detection.ipynb       ← DBSCAN clustering
│   └── hotspot_map.html              ← Standalone map output
│
├── 🐍 app.py                         ← Main Streamlit application
├── 🤖 Model_Training.ipynb           ← XGBoost training notebook
├── 📋 requirements.txt               ← Python dependencies
├── 🐍 runtime.txt                    ← Python 3.11 pin
└── 📖 README.md                      ← You are here
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/pritam1952/GeoSafe.git
cd GeoSafe
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data files (if not present)
```bash
# Run notebooks in this order:
# 1. notebooks/EDA.ipynb
# 2. notebooks/Hotspot_Detection.ipynb
# 3. Model_Training.ipynb
```

### 4. Launch the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser 🎉

---

## 📦 Requirements

```txt
streamlit==1.45.0
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
xgboost==2.0.3
folium==0.16.0
streamlit-folium==0.20.0
plotly==5.22.0
joblib==1.4.2
pillow==10.3.0
```

---

## ☁️ Deployment

App is live on **Streamlit Community Cloud**:

👉 **[https://geosafe.streamlit.app](https://geosafe.streamlit.app)**

### To redeploy yourself:
1. Fork this repo
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app** → select your fork → set main file to `app.py`
4. Click **Deploy** → live in ~3 minutes

---

## ⚠️ Known Limitations

- The severity model includes `casualties` and `vehicles_involved` as training features — both are post-accident outcomes (**data leakage**). The Predictor fills these with dataset medians at inference time. For a production-clean model, retrain without these two columns.
- Dataset represents Indian road conditions — not official government data.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Model** | XGBoost, scikit-learn |
| **Clustering** | DBSCAN (scikit-learn) |
| **Data** | Pandas, NumPy |
| **Visualisation** | Plotly, Folium |
| **App Framework** | Streamlit |
| **Serialisation** | Joblib |
| **Language** | Python 3.11 |
| **Deployment** | Streamlit Community Cloud |

---

## 👤 Author

**Pritam**
- GitHub: [@pritam1952](https://github.com/pritam1952)
- Live App: [geosafe.streamlit.app](https://geosafe.streamlit.app)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ for Road Safety in India 🇮🇳

**⭐ If you found this useful, please star the repo!**

</div>
