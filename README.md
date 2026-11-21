# AutoValuate AI — Second-Hand Car Price Prediction Web App

A full-stack Machine Learning web application built with **HTML, CSS, JavaScript, Python (Flask), and scikit-learn**.

## Features
- **Multi-Model Selection:** Benchmarks 7 regression algorithms (`Ridge`, `K-Nearest Neighbors`, `Decision Tree`, `Random Forest`, `Extra Trees`, `Gradient Boosting`, and `HistGradientBoostingRegressor`) and automatically deploys the highest-scoring model (`HistGradientBoostingRegressor`, `R² = 0.9904`).
- **Confidence Range:** Estimates fair low/high market bounds using 15th/85th percentile quantile regression.
- **5-Year Resale Forecast:** Interactive SVG chart projecting future value over the next 5 years.
- **Price Drivers (Explainability):** Marginal sensitivity analysis showing how age, mileage, fuel type, gearbox, ownership, and condition impact the estimate.
- **Currency Toggle & Comparison History:** Switch between INR (`₹`) and USD (`$`), use live slider prediction, and save comparisons in browser `localStorage`.

## Project Structure
- `car_catalog.py` — Shared vehicle catalog, valid powertrain options, and default specs.
- `generate_dataset.py` — Generates `data/used_cars.csv` using domain depreciation curves.
- `train_model.py` — Benchmarks all 7 algorithms, selects the best model, and saves `model/car_price_model.joblib` & `model/metadata.json`.
- `test.py` — Implements and tests only the winning model (`HistGradientBoostingRegressor`).
- `app.py` — Flask server serving `static/` assets and JSON REST endpoints (`/api/catalog`, `/api/predict`, `/api/health`).
- `static/` — `index.html`, `style.css`, and `app.js` frontend.
- `test_app.py` — Flask API and integration tests.
- `Dockerfile` & `.dockerignore` — Multi-stage non-root production container with Gunicorn.

## Quick Start

```powershell
# 1. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset and benchmark all models
python generate_dataset.py
python train_model.py

# 4. Run the winning model script and API tests
python test.py
python -m unittest test_app.py -v

# 5. Start the web application
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Docker

```bash
# Build the multi-stage production image (generates data, trains & verifies model)
docker build -t autovaluate-ai .

# Run the container on port 5000
docker run --rm -p 5000:5000 --name autovaluate autovaluate-ai
```

Then open `http://127.0.0.1:5000`. To map another host port (e.g. `8080`), run `docker run --rm -p 8080:5000 autovaluate-ai`.
