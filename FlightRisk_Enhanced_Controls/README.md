# FlightRisk AI — web app (Streamlit)

A clean customer-facing website connected to **the actual Layer 2 CatBoost models** exported from the user's Google Colab notebook. This is not a sample model or fabricated prediction page.

## What customers see

- **I'm flying** — choose a full airline name, departure/arrival airports, date and scheduled departure time. See a route map, estimated chance of being 15+ minutes late, and an approximate arrival time in the **destination's local time**.
- **I'm picking someone up** — same risk estimate, with an arrival estimate. Optionally enter the **scheduled arrival from the ticket** for a more useful time estimate.
- Calendars and time pickers instead of free-form date/time text; a minimal, dark-navy / mint interface. No model evaluation plots or ML terminology in the customer view.

## Run locally

1. Extract the ZIP. Keep `app.py`, `inference.py`, `requirements.txt`, `models/`, and `.streamlit/` together.
2. Open a terminal **inside the extracted folder**.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Start: `python -m streamlit run app.py`
5. Open the local URL displayed in the terminal (usually `http://localhost:8501`).

**Do not run this from Google Colab.** Colab was used to train the models; the website uses those saved models independently.

## Give your professor a persistent URL — Streamlit Community Cloud

1. Create or sign in to GitHub and create a repository, for example `flightrisk-ai`.
2. Upload **all the files from this folder**, including the `models` subfolder and `requirements.txt`. Make sure `.streamlit/config.toml` is included (a hidden folder on some systems).
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Choose **Create app → Deploy a public app from GitHub**; select your repository and branch.
5. Choose `app.py` as your **entrypoint** and deploy.
6. Test the resulting `*.streamlit.app` link from an incognito browser and send that link to your professor. Free apps can sleep while idle, but the project URL does not automatically expire after seven days.

You need to complete the GitHub and Streamlit login/deployment steps yourself. **This ZIP is a ready-to-deploy app; no website has been published by the assistant.**

## Exact trained artifacts

- `models/flightrisk_layer2_delay_classifier.cbm`: trained **CatBoost classifier** (arriving 15+ minutes late)
- `models/flightrisk_layer2_arrival_delay_regressor.cbm`: trained **CatBoost regression** model (arrival delay in minutes)
- `models/flightrisk_layer2_metadata.json`: exact 44 features, preprocessing medians, available airports, historical lookups and date range
- `models/supported_routes.csv`: supported airline + route combinations (7,394 combinations in the export)

The model file and metadata are read directly. The website **does not silently re-train** a different estimator.

## Results from the uploaded Colab report (not claims of live accuracy)

- Selected classifier: **Engineered + weather forecasts**
- Final test ROC-AUC: **0.7172** (original nine-feature version: 0.7046)
- Final test PR-AUC: **0.4675**
- Arrival-delay regression test MAE: **28.59 minutes** (constant-median baseline 29.97 minutes)
- Weather coverage in training source: **29.2% origin**, **28.8% destination**. The weather model **does not have complete weather coverage**.
- Training period: **2025-01-01 through 2026-06-30**; unseen test month: **2026-08**.

These results are **moderate**, not evidence of highly accurate flight-specific arrival predictions. The customer website uses cautious language.

## Sources and feature construction

- Flights, airlines, routes, dates, scheduled times and outcomes: [Kaggle dataset selected by the user](https://www.kaggle.com/datasets/samuelpombourcq/us-flight-delays-and-cancellations-2025-2026).
- Airport names, coordinates and approximate geographic timezone identification: [OurAirports](https://github.com/davidmegginson/ourairports-data), already exported in `models/flightrisk_layer2_metadata.json`.
- Federal holiday flags: derived at prediction time from `pandas.tseries.holiday.USFederalHolidayCalendar`.
- Historical on-time rates / sampled airport-hour volumes: training-period lookups frozen in the metadata. These are **not live congestion** readings.
- Optional weather: [Open-Meteo forecast API](https://open-meteo.com/en/docs) for near-future flights. The model was trained on its **24-hour-ahead historical forecast product**, not observed hindsight weather. Live forecasts are a useful but **not perfectly equivalent** lead-time input. Weather is used only for the eight airports represented in the training weather subset; otherwise the trained missing-weather fallback is used. Predictions still work if Open-Meteo is unreachable.

## Important limitations

- **US domestic flights only.** International routes have not been trained or validated.
- **No realtime flight status or cancellation prediction.** The outcome classifier is conditioned on **completed, non-diverted flights**; cancelled or diverted flights were excluded from training.
- The form doesn't identify a specific published flight number or access an airline's current schedule. The **scheduled arrival shown by default is approximated from the median historical scheduled duration** of that airline + route. In pickup mode, users can enter the scheduled arrival **from their ticket** to improve that reference time.
- The estimated arrival is **not guaranteed**, and the average held-out error is nearly 29 minutes. Use airline/airport live status for actual pickup decisions.
- The historical group lookup is frozen through June 2026; we do not claim it includes new flights since then.
- For flights further in the future, weather is unavailable and the model falls back to historical features and learned weather-missing indicators.
- The model was evaluated on an August 2026 holdout; accuracy in other periods, routes and unusual weather may differ.

## Testing

`python test_inference.py`

This checks bundle loading, the 44 exact inference features, supported route constraints, real CatBoost predictions, and departure/arrival time zone handling. The Streamlit UI itself requires Streamlit to be installed and must be visually checked after deployment.


## Updated route map and feature context (v2)

- The route map is drawn as an embedded SVG using public-domain Natural Earth boundaries. It does not depend on Leaflet, external map tiles, JavaScript or network access to render. Routes are indicative arcs, not actual aircraft tracks.
- The HTML `Design_Preview_Only.html` is an offline-compatible design demo and **does not generate predictions**. To run a true inference, deploy or start the `app.py` Streamlit application.
- The customer results now include a concise “Your flight context” section for historical route trends, travel-day features, nearby US federal holidays and **available** short-range weather forecasts. It deliberately omits model diagnostics and invented feature contributions.
- US domestic flights only. Models are statistical historical estimates, not live flight tracking; cancellation risk is not modelled.

## v3 — Explicit flight outlook and context controls

- **Choose prediction type:** *Arrival delay* (actual trained CatBoost model) or *Departure delay* (scheduled departure details only; **no departure-delay model has been trained**). The latter is explicitly marked unavailable for ML prediction so the website does not mislabel an arrival prediction.
- **Choose travel purpose:** *I'm flying* or *I'm picking someone up*. The pickup flow optionally accepts scheduled arrival date/time from a ticket and estimates arrival based on the arrival-delay regression.
- **Weather & travel conditions:** customer-facing cards for departure/arrival weather (when supported, short-range forecasts only), US federal holiday proximity and route/time context. Missing weather is shown as unavailable, never fabricated.
- **Preview:** `Design_Preview_Only.html` demonstrates the form controls and map but does **not** generate real flight predictions. Actual ML predictions only work in `app.py` after running/deploying Streamlit.
- Current trained Layer 2 assets only predict **arrival delay on completed US domestic flights**. To add genuine departure-delay predictions in future, train/export a separate `DepDelay`-targeted model with the same pre-departure no-leakage discipline.
