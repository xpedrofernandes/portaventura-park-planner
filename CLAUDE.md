# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is an early-stage Python data project — only a `.gitignore`, `requirements.txt`, and one data-prep script are committed so far. There is no README, no `pyproject.toml`, and no tests. Treat this as a greenfield project: there is no established architecture to preserve, so when adding code, establish structure deliberately rather than searching for conventions that don't exist yet.

## Environment

- Python 3.12 virtual environment at `.venv/` (Windows, created with the standard `venv` module). Activate with `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`).
- Dependencies are frozen in `requirements.txt` (install with `.venv\Scripts\pip.exe install -r requirements.txt`). Re-freeze with `.venv\Scripts\pip.exe freeze > requirements.txt` after adding packages so the environment stays reproducible.
- Key packages: `pandas`, `numpy`, `scipy`, `pyarrow`, `lightgbm` (data/ML), `matplotlib`, `streamlit`, `altair`, `pydeck`, `pillow` (viz/dashboard), `anthropic`, `python-dotenv` (AI integration/config), `openpyxl` (reading the `.xlsx` sources).
- `.env` holds an `ANTHROPIC_API_KEY` and is git-ignored — never commit it or print its contents.

## Data

Raw source data lives in `data/raw/` (git-ignored — not committed to the repo, so don't assume it's present in a fresh clone):

- `attendance.csv` — daily park attendance: `USAGE_DATE, FACILITY_NAME, attendance`
- `entity_schedule.csv` — opening schedules and closures for parks/attractions: `ENTITY_DESCRIPTION_SHORT, ENTITY_TYPE (PARK/ATTR), DEB_TIME, FIN_TIME, WORK_DATE, REF_CLOSING_DESCRIPTION`
- `link_attraction_park.csv` — semicolon-delimited mapping of `ATTRACTION;PARK`
- `waiting_times.csv` — per-attraction, per-time-slot wait times and throughput: `WORK_DATE, DEB_TIME/FIN_TIME, WAIT_TIME_MAX, NB_UNITS, GUEST_CARRIED, CAPACITY, ADJUST_CAPACITY, OPEN_TIME, UP_TIME, DOWNTIME, NB_MAX_UNIT`
- `weather_data.csv` — hourly weather observations (OpenWeather-style schema): `dt, dt_iso, temp, humidity, wind_speed, rain_*, snow_*, weather_main, weather_description, ...`
- `glossary.xlsx`, `parade_night_show.xlsx` — Excel workbooks (reading them requires `openpyxl`, which is not currently installed in `.venv`)

Note the inconsistent delimiters/date formats across files (`link_attraction_park.csv` uses `;`, others use `,`; timestamp formats vary between files) — normalize explicitly rather than assuming a single parsing path works for all of them. Two parks appear in the data: PortAventura World and Tivoli Gardens.

`waiting_times.csv` is ~365MB spanning 2018-01-01 to 2022-08-18 (~3.5M rows) — never load it whole. Read it with `usecols` and `chunksize`, filter per chunk, then concatenate.

A row is "closed" (ride not operating) when `OPEN_TIME == 0` and `CAPACITY == 0`; `WAIT_TIME_MAX` is ~0 in that state. `DOWNTIME > 0` is a narrower, separate signal (ride breakdown) and only applies to a small fraction of rows — don't conflate the two.

Derived/processed datasets go in `data/processed/` (git-ignored, like `data/raw/` — regenerate via the scripts below rather than expecting it to exist after a fresh clone).

`data/rides.json` (committed — small, hand-curatable) has one entry per PortAventura ride (`name`, `min_height_cm`, `tags`, `zone`). `tags` are drawn from `src/tags.ALLOWED_TAGS`; `zone` is one of 5 invented zones (Thrill Peak, Coaster Canyon, Lagoon Cove, Kids Kingdom, Mystic Quarter) grouping rides that would plausibly sit near each other. **All of this — heights, tags, and zone layout — is fabricated/plausible-guess data, not sourced from a real PortAventura map or ride specs**; treat it as placeholder content to replace if real ride data ever becomes available, not as ground truth.

## Scripts

- `scripts/build_portaventura_dataset.py` — streams `data/raw/waiting_times.csv` in chunks, keeps only PortAventura World rides with the ride actually open, and writes `data/processed/portaventura_wait_times.parquet`. Run with `.venv\Scripts\python.exe scripts\build_portaventura_dataset.py`. Use this as the template for any similar chunked extraction (e.g. a Tivoli Gardens equivalent) rather than re-reading the raw CSV from scratch each time.
- `scripts/build_baseline_lookup.py` — reads `portaventura_wait_times.parquet` and writes `data/processed/baseline_wait_lookup.parquet`: median `WAIT_TIME_MAX` grouped by `ride, hour, day_of_week, month`. Intended as the baseline/reference table for later modeling work (e.g. an anomaly or uplift model would compare live wait against this).
- `scripts/train_wait_time_model.py` — trains a LightGBM model (native `lgb.train` API — scikit-learn is not installed, so don't use `lgb.LGBMRegressor`) on `ride, hour, day_of_week, month, attendance, temp, rain`, time-split train=2018-2021/test=2022 (never random-split this data — it's a forecasting problem). Attendance comes from `attendance.csv` filtered to `FACILITY_NAME == "PortAventura World"`; weather is daily-aggregated (mean temp, summed `rain_1h` with NaN treated as 0) from `weather_data.csv`, which covers a single fixed lat/lon ("Custom location", ~Paris coordinates — not actually PortAventura's location, but it's the only weather source available) rather than being per-park. Attendance only starts 2018-06-01, so rows before that have NaN attendance; left as NaN and handled natively by LightGBM's missing-value splitting rather than imputed. Saves the model to `models/` (git-ignored, like `data/processed/`).

**Important finding**: the pre-built baseline lookup as saved is not a fair comparison for the time-split model, because it was fit on all years 2018-2022 (including the test period) — it has already seen 2022's actual medians. `train_wait_time_model.py` therefore also recomputes a train-only version of the same lookup (fit on 2018-2021 only) for the fair comparison. Last run: model MAE 9.89 vs. fair (train-only) baseline MAE 11.49 — the model wins — vs. the leaky all-years baseline MAE 9.08, which beats the model only because of the leakage. Use the train-only number as the real baseline to beat.

One ride (Vertical Drop) only has 2 months of data and 1,623 rows total — likely opened/closed mid-period. Excluding it from monthly aggregates barely changes results, so ride-mix isn't a meaningful confound for the by-month seasonality numbers. Individual rides also don't have data on every calendar day in their date range (~75-80% coverage is typical) even though they report across the full range of operating hours (9am-10pm) — treat missing ride-days as no data, not as zero wait.

## LLM layer

- `src/tags.py` — the single source of truth for the ride-tag vocabulary: `ALLOWED_TAGS = ["water", "thrill", "family", "kids", "coaster", "spinning", "heights", "indoor"]`. Anything tagging rides or parsing tag-like constraints (e.g. `rides.json`'s `tags` field once populated) should import from here rather than hardcoding the list, so the vocabulary only ever changes in one place.
- `src/extract.py` — turns a natural-language planning request into validated constraints (`arrival_time`, `end_time`, `max_height_cm`, `exclude_tags`, `prefer_tags`) by calling Claude Haiku (`claude-haiku-4-5-20251001`) with a forced tool call (`tool_choice={"type": "tool", ...}`) so the response is JSON matching a fixed schema, then re-validates it in a plain-dataclass `Constraints.from_dict` (format/type checks — there's no pydantic in this project). `exclude_tags`/`prefer_tags` are constrained to `tags.ALLOWED_TAGS` both in the tool's JSON schema (`enum`) and again defensively in `from_dict` (any tag outside the vocabulary — from a model that ignores the enum, or from hand-built input — is silently dropped, and duplicates are collapsed, rather than raising). Reads `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`. Run `python src/extract.py` for 3 built-in test cases (no pytest installed — this is a plain runnable script, not a test suite), or `python src/extract.py "<request text>"` for ad hoc input. Since `src/extract.py` does `from tags import ALLOWED_TAGS` (not a relative/package import), always run scripts in `src/` directly (`python src/extract.py`), not as a `-m src.extract` module.
- `src/evaluate.py` — runs every case in `data/eval_set.json` through `extract.py` then `planner.py` and reports two independent metrics: **extraction accuracy** (does every field named in a case's `expected` dict exactly match extraction output — list fields compared as sets, fields the case doesn't mention aren't checked) both overall and per-field, and **plan constraint-satisfaction rate** (does the schedule built from the *extracted* constraints actually honor them — no ride over `max_height_cm`, none with an excluded tag, nothing outside `[arrival_time, end_time)` — this checks `planner.py`'s own correctness, independent of extraction accuracy). Uses a fixed `EVAL_DATE = 2022-07-15` (not `date.today()`) so results are reproducible run to run in everything *except* the LLM call itself. `extract.py` sets `temperature=0` (passed via `extra_body`, since this SDK build's typed `messages.create` signature doesn't expose `temperature` directly — checked with `inspect.signature`), but that only *reduces*, not eliminates, run-to-run noise: 3 consecutive evaluate.py runs scored 76.0%/80.0%/80.0%, and even the two 80.0% runs differed in raw extracted output on an unchecked field (case 14's `arrival_time`: `None` vs `"09:00"`). Don't read a 1-2 case swing as a regression without re-running a few times first. Catches `ConstraintValidationError` per-case (Haiku occasionally returns something extract.py's own validation rejects, e.g. a literal `"<UNKNOWN>"` for a time field) so one bad case doesn't abort the whole run — that case is scored as a failure instead. Run `python src/evaluate.py`. `data/eval_set.json` (25 hand-written cases plus a `_decisions` block spelling out the height/time conventions used to write `expected`, e.g. "until close" → `22:00`) is committed.

Extraction accuracy history (same 25 cases, tracks prompt changes in `extract.py`'s `SYSTEM_PROMPT`):
| | extraction accuracy | plan constraint-satisfaction |
|---|---|---|
| Before adding `_decisions` time/height conventions | 68.0% (17/25) | 96.0% (24/25) |
| After (park-closes/morning/after-dinner/toddler/small-child conventions + "reservation time is end_time, not arrival_time") | 76.0% (19/25) | 100.0% (25/25) |

Remaining failures are mostly `prefer_tags`/`exclude_tags` under-/over-inference on vague requests (e.g. "keep it gentle" → model adds `kids` beyond expected `family`; "most intense stuff" → model adds `coaster` beyond expected `thrill`) rather than a conventions gap — read `python src/evaluate.py`'s failure table for the current specifics before trying to fix more of these, since exact numbers drift between runs.
- `src/planner.py` — greedy same-day scheduler: `build_schedule(constraints, date=...)` filters `data/rides.json` by `Constraints` (drops rides taller than `max_height_cm`, or tagged with any `exclude_tags`), then repeatedly picks whichever remaining ride has the lowest predicted wait *at the time the visitor would actually reach it* (via `models/wait_time_lgbm.txt`), applying a 5-minute intra-zone / 10-minute inter-zone walk and a flat 5-minute ride duration; `prefer_tags` rides get a `PREFERENCE_BONUS_MIN=5`-minute head start in the ranking rather than being force-picked. It's a greedy cheapest-next-ride heuristic, not a globally optimal tour. Requires `models/wait_time_lgbm.txt` to already exist (`python scripts/train_wait_time_model.py` first — the model is git-ignored). Since there's no real future attendance/weather to predict on, unset `attendance`/`temp`/`rain` default to that target date's historical monthly average (`_monthly_defaults()`, computed from `attendance.csv`/`weather_data.csv`); pass a `date` to control which month/day-of-week is used, since `Constraints` itself carries no date. The model's `ride` categorical codes were fixed at training time to the sorted order of the *full* 26-ride list — `_predict_wait` always builds that categorical against all of `data/rides.json`, never just the filtered candidates, or predictions would silently be wrong. Run `python src/planner.py` for a built-in example.

## Working in this repo

- Since there's no lint/test tooling yet, when you introduce the first tests also set up the minimal commands needed to run them, and document those commands here.
