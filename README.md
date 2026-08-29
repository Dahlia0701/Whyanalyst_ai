# 📊 WhyAnalyst AI

A full-stack, natural language Explainable AI (XAI) and data analytics platform. WhyAnalyst AI allows users to upload tabular CSV datasets, ask conversational questions, execute statistical aggregations, generate dynamic visualizations, train predictive machine learning pipelines, and interpret model decisions using SHAP value feature attributions alongside LLM-synthesized narrative explanations.

---

## ✨ Key Features

### 🧠 Explainable AI (XAI) & Natural Language Insights

* **Granular Model Attribution:** Generates SHAP waterfall force plots for record-level and dataset-level attributions.
* **Automated Label Sanitization:** Post-processes Scikit-Learn transformer output strings into human-readable business labels (e.g., converting `cat__Product_Type_Office Supplies` to `Product Type: Office Supplies`).
* **Executive Narrative Generation:** Synthesizes raw numeric SHAP feature values into clear, plain-English "What this chart means" text explanations, rendered directly beneath each generated chart.

### 📈 Interactive Visualizations & Statistical Analytics

* **Natural Language Aggregations:** Evaluates key statistical operations including `mean`, `sum`, `max`, and `min` directly from freeform queries.
* **Dynamic Plotly Rendering:** Outputs responsive interactive visualizations using client-side JavaScript without requiring full-page reloads.

### 🔮 Dual-Dataset Predictive Pipeline

* **Out-of-Sample Prediction:** Accepts a primary dataset for training alongside an optional secondary dataset for model testing and inference.
* **Automated Model Metrics:** Generates model validation metrics (such as MAE score) alongside predicted output rows.

### ⚙️ System Design Highlights

* **Zero-Shot Query Intent Parsing:** Maps unstructured natural language prompts into standardized JSON intent payloads.
* **Decoupled Architecture:** Separates data ingestion, NLP parsing, statistical computation, and machine learning pipelines into distinct modular packages.

---

## 📺 Demo

Check out the **Whyanalyst_AI** in action:

<div align="center">
  <video src=https://github.com/user-attachments/assets/5bed3396-aace-415e-a909-0a6f7c09ebb1 width="100%" controls>
</video>
</div>

## 🧠 How It Works (Pipeline)

```
[ User Data Upload (.csv) ] 
            │
            ▼
   [ core/loader.py ] ────────► Standardizes & cleans raw dataset
            │
            ▼
 [ core/inspector.py ] ──────► Extracts dataset schema & metadata
            │
            ▼
[ User Query Input ]
            │
            ▼
  [ nlp/llm_parser.py ] ─────► Tokenizes input & classifies Intent
            │
   ┌────────┴─────────────────┬──────────────────────┬────────────────────────┐
   ▼                          ▼                      ▼                        ▼
[ Aggregation ]         [ Visualization ]      [ Explainability ]       [ Prediction ]
(analytics/engine.py)   (analytics/plotter.py)  (ml/explainer.py)        (ml/predictor.py)
   │                          │                      │                        │
   └────────┬─────────────────┴──────────────────────┴────────────────────────┘
            │
            ▼
 [ FastApi Output Endpoint ] ──► Async payload streaming (JSON / Charts / Narratives)
            │
            ▼
 [ Frontend Dashboard ] ──────► Renders interactive Plotly visual charts, summary tables & AI narrative boxes

```

---

## 🧰 Tech Stack

### 🖥️ Frontend

* **HTML5 / CSS3:** Modern dark-slate aesthetic responsive layout.
* **Vanilla JavaScript (ES6+):** Dynamic fetch API integration and asynchronous DOM manipulation.
* **Plotly.js:** Dynamic client-side visualization library.

### ⚙️ Backend

* **Python:** `3.9.11`
* **FastAPI:** `0.128.8`
* **Uvicorn:** ASGI web server worker.

### 🤖 AI / ML

* **Google Gemini API:** `google-genai` for intent parsing and natural language narrative generation.
* **Scikit-Learn & XGBoost:** Data preprocessing pipelines and predictive machine learning execution.
* **SHAP (SHapley Additive exPlanations):** Model interpretability and feature contribution extraction.
* **Pandas & NumPy:** In-memory data manipulation and statistical compute routines.

---

## 🚀 Installation & Setup

### 1. Prerequisites

Ensure Python `3.9.11` is installed on your local environment.

### 2. Backend & Environment Setup

Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone https://github.com/Dahlia0701/Whyanalyst_ai.git
cd Whyanalyst_ai

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

Create a `.env` file in the project root directory and supply your Gemini API Key:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

### 3. Running the Server

Launch the FastAPI development server using Uvicorn:

```bash
uvicorn src.app:app --reload

```

The API server will run locally at `http://127.0.0.1:8000`.

Uploaded datasets are stored at runtime under `storage/datasets/` (auto-created on first upload).

### 4. Frontend Launch

Serve the static files using your preferred web server setup (e.g., VS Code Live Server targeting `frontend/index.html`).

---

## 💡 How to Use

### 🔍 Asking Natural Language Queries

Upload your CSV dataset through the top dropzone interface. Enter standard data queries directly into the console prompt:

* *"What is the mean of Sales?"*
* *"Sum of Profit by Product_Type"*

### 📊 Interpreting SHAP & Narrative Output

Ask specific explanatory queries to trigger the Explainable AI (XAI) engine:

* *"Why is the profit high for electronics in the east region?"*
* The system constructs a SHAP waterfall chart detailing local feature impacts alongside a **"What this chart means"** natural language executive commentary rendered directly beneath the chart.

### 🎯 Running Out-of-Sample Predictions

1. Upload your primary training dataset (`data.csv`).
2. Upload your secondary prediction target dataset (`data3.csv`).
3. Query the model: *"Predict the sales of data3.csv"*.
4. Review the generated row-level predicted values and model validation scores (e.g., MAE Score).

---

## 📁 Project Structure

```text
whyanalyst_ai/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── work.js
├── src/
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── plotter.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── inspector.py
│   │   └── loader.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── explainer.py
│   │   ├── pipeline.py
│   │   └── predictor.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── intent.py
│   │   ├── llm_parser.py
│   │   └── re_parser.py
│   ├── app.py
│   └── main.py
├── storage/
│   └── datasets/        # auto-created at runtime; holds user-uploaded CSVs
├── tests/
│   ├── test_llmt.py
│   ├── test_llm.py
│   ├── test_ml.py
│   └── test_reg.py
├── .env
├── .gitignore
└── requirements.txt

```

> **Note:** `storage/datasets/` is generated automatically by `app.py` on first upload and is git-ignored — you won't see it in a fresh clone until you run the server and upload a file.

---

## ⚠️ Limitations & Future Scope

### Current Limitations

* **Supported File Formats:** Limited to tabular CSV input files.
* **In-Memory Compute:** Execution scales based on local RAM capacity during SHAP calculation runs.
* **Dataset Registry:** Uploaded dataset metadata is currently held in an in-memory Python dict (`dataset_registry`) rather than a persistent database, so it resets on server restart.

### Future Scope

* **Extended File Format Ingestion:** Native ingestion support for Excel (`.xlsx`) and Parquet (`.parquet`) file formats.
* **Automated Model Optimization:** Integrated hyperparameter tuning and automated model selection pipelines.
* **Database Connectors:** Live query execution against external SQL databases (PostgreSQL, MySQL, Snowflake), and replacing the in-memory dataset registry with MongoDB.
* **Exportable Reports:** One-click automated PDF export summarizing key analytical outputs, charts, and model explainability metrics.

---

## 🤝 Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.
