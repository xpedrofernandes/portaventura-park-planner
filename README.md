# Park Planner

Turns a plain-language request ("arriving at 10am, my daughter is 115cm, avoid big drops") into a same-day ride schedule for PortAventura World, ordered to minimize predicted queue time.

## What it does

1. **Understand the request.** A natural-language visit description is parsed into structured constraints — arrival/end time, a max rider height, and ride tags to avoid or prefer.
2. **Forecast queue waits.** A model trained on historical wait-time data predicts how long the queue will be for any ride at any hour, day of week, and month.
3. **Build a schedule.** Starting from the visitor's arrival time, the planner repeatedly picks whichever eligible ride currently has the shortest predicted wait, until the visitor's end time is reached.

A Streamlit app (`app.py`) wraps this end to end for interactive use; `src/evaluate.py` scores the whole pipeline against a hand-written set of example requests.

## Architecture

```
natural-language request
        |
        v
  LLM extraction (src/extract.py)
  Claude Haiku + forced tool call -> validated Constraints
  (arrival_time, end_time, max_height_cm, exclude_tags, prefer_tags)
        |
        v
  Greedy planner (src/planner.py)
  filters data/rides.json by height & excluded tags
        |
        v
  LightGBM forecaster (models/wait_time_lgbm.txt)
  predicts WAIT_TIME_MAX for (ride, hour, day_of_week, month,
  attendance, temp, rain)
        |
        v
  preference-tiered greedy ordering
  at each step: prefer_tags rides are scheduled first (if any are
  still reachable), then whichever remaining ride has the lowest
  predicted wait; +5min ride, +5/10min zone-walk between stops
        |
        v
  schedule: [{time, ride, zone, predicted_wait_min}, ...]
```

The forecaster is trained separately, offline, from raw historical data:

```
data/raw/waiting_times.csv (~365MB, chunked read)
  -> scripts/build_portaventura_dataset.py -> data/processed/portaventura_wait_times.parquet
  -> scripts/build_baseline_lookup.py      -> data/processed/baseline_wait_lookup.parquet (median lookup, for comparison)
  -> scripts/train_wait_time_model.py      -> models/wait_time_lgbm.txt (LightGBM, time-split train/test)
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Create a `.env` file in the project root with:

```
ANTHROPIC_API_KEY=sk-ant-...
```

You'll also need the raw CSVs (`attendance.csv`, `entity_schedule.csv`, `link_attraction_park.csv`, `waiting_times.csv`, `weather_data.csv`, plus two `.xlsx` files) under `data/raw/` — these aren't included in the repo (git-ignored, large). With them in place, build the derived data and model:

```
.venv\Scripts\python scripts\build_portaventura_dataset.py
.venv\Scripts\python scripts\build_baseline_lookup.py
.venv\Scripts\python scripts\train_wait_time_model.py
```

This populates `data/processed/` (git-ignored, regenerate it locally) and `models/wait_time_lgbm.txt`, which is committed to the repo (so it deploys with it to Streamlit Cloud) -- if you regenerate it, commit the updated file so the deployed app stays in sync.

## Running it

**Interactive app:**
```
.venv\Scripts\python -m streamlit run app.py
```

**Command line:**
```
.venv\Scripts\python src\extract.py "we arrive at 10am, no water rides, kid is 120cm"
.venv\Scripts\python src\planner.py     # runs a built-in example
```

**Evaluation:**
```
.venv\Scripts\python src\evaluate.py             # single run
.venv\Scripts\python src\evaluate.py --runs 3     # repeat 3x, report mean/min/max + per-case consistency
```

## Results

Evaluated against `data/eval_set.json` (25 hand-written natural-language requests with expected constraints) and a 2022 time-based holdout for the forecaster.

| Metric | Result |
|---|---|
| Forecaster MAE (2022 holdout) | **9.89**, vs. **11.49** for a median-lookup baseline fit on the same 2018-2021 training data (time-based split: train 2018-2021, test 2022 — never randomly split, since this is a forecasting problem) |
| Extraction accuracy | **76-80%** across repeated runs (exact match of every field a test case specifies; list fields compared as sets) |
| Plan constraint-satisfaction | **100%** — every schedule built from extracted constraints honors its own height/tag/time-window limits |
| Prefer-tags satisfaction | **86.7%** — mean fraction of each schedule's first 5 rides that match a preferred tag, across the 12 eval cases with non-empty `prefer_tags` |

## Known limitations

- **Ride metadata is fabricated.** `data/rides.json`'s heights, tags, and 5 invented zones (Thrill Peak, Coaster Canyon, Lagoon Cove, Kids Kingdom, Mystic Quarter) are plausible guesses, not sourced from any real PortAventura map or ride specs.
- **Weather is geolocated wrong.** `weather_data.csv` covers a single fixed lat/lon near Paris, not PortAventura's actual location in Spain — it's the only weather source available, so the forecaster uses it as a proxy anyway.
- **Data ends 2022-08-18.** There's no more recent wait-time, attendance, or weather data to train or plan against.
- **The planner is greedy, not optimal.** It picks the cheapest reachable ride at each step (with `prefer_tags` as a hard priority tier); it does not search for the schedule that minimizes total wait over the whole day.
- **LLM extraction isn't fully deterministic**, even at `temperature=0` — repeated `evaluate.py` runs on the same 25 cases have scored anywhere from 76% to 80%, and even runs with identical aggregate scores can differ on individual unchecked fields.
- **Two real user intents the schema can't express:**
  - *"the most popular rides"* — there's no popularity/ranking concept anywhere in `Constraints` or `rides.json`; the planner only optimizes for short waits, which is a different (if correlated) thing.
  - *"we don't have express passes"* — there's no skip-the-line/fast-pass concept modeled at all; every predicted wait assumes the standard queue.
- **Most consistent extraction failures are defensible, not wrong.** Of the 5-6 cases that fail on every run (the exact count varies slightly with sampling), 4 are the model being *more* cautious or inclusive than the hand-written expected answer, rather than a genuine misread: "the most intense stuff" additionally prefers `heights` beyond the expected `thrill`; "keep it gentle" additionally excludes `heights` and prefers `kids` beyond `family`; "nothing rough" (paired with a dinner reservation) additionally excludes `water`; "two kids under 8, no water, no big drops" additionally excludes `thrill` beyond the expected `water`/`heights`. These fail exact-match scoring against `eval_set.json` but arguably reflect reasonable extrapolation. The remaining 1-2 are more clear-cut gaps: "hit all the big coasters" drops the expected `thrill` tag entirely, and "got the little one with us... gates open when?" infers no height limit at all, since "little one" isn't covered by the toddler/small-child conventions in the extraction prompt.
