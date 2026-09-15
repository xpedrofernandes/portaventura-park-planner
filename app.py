"""Streamlit UI for the natural-language ride planner.

Run with:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import base64
import dataclasses
import pathlib
import sys

import lightgbm as lgb
import pandas as pd
import streamlit as st

# extract.py/planner.py use plain sibling imports (e.g. "from tags import
# ALLOWED_TAGS") that only resolve when src/ itself is on sys.path -- add it
# before importing, since this file lives at the project root.
APP_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extract import ConstraintValidationError, extract_constraints  # noqa: E402
from planner import build_schedule, load_model, load_rides  # noqa: E402

st.set_page_config(page_title="Park Planner", page_icon="🎢")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_rides():
    return load_rides()


@st.cache_data
def get_background_b64() -> str:
    img_bytes = (APP_DIR / "assets" / "background.jpg").read_bytes()
    return base64.b64encode(img_bytes).decode("ascii")


def _no_meaningful_constraints(constraints) -> bool:
    return (
        constraints.arrival_time is None
        and constraints.end_time is None
        and constraints.max_height_cm is None
        and not constraints.exclude_tags
        and not constraints.prefer_tags
    )


# Fixed, cover-sized, dark-overlaid background photo plus "glass" panels for
# content. Colors here are hardcoded rather than Streamlit theme variables so
# the look is identical regardless of the user's light/dark theme setting --
# this page always uses light text on a dark panel over the photo.
#
# The image+overlay lives on its own `position: fixed` div (z-index: -1)
# rather than as `.stApp`'s own background-image. `.stApp` isn't reliably the
# actual scrolling container across Streamlit versions/layouts, so a
# background painted directly on it can end up sized/positioned against its
# full (possibly page-tall) box instead of the viewport -- on a page with
# enough content to scroll, that let a brighter, less-covered part of the
# image show through further down. A dedicated fixed, inset:0 div is pinned
# to the viewport unconditionally, so the same darkened crop shows no matter
# how far the page scrolls or how tall the content is.
st.markdown(
    f"""
    <div class="app-bg"></div>
    <style>
    .stApp {{
        background: transparent;
    }}
    .app-bg {{
        position: fixed;
        inset: 0;
        z-index: -1;
        background-image:
            linear-gradient(
                to bottom,
                rgba(4, 8, 14, 0.58) 0%,
                rgba(4, 8, 14, 0.72) 55%,
                rgba(4, 8, 14, 0.85) 100%
            ),
            url("data:image/jpeg;base64,{get_background_b64()}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    .block-container {{
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }}
    .st-key-hero_panel, .st-key-results_panel {{
        background: rgba(10, 15, 24, 0.68);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.75rem 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
    }}
    .st-key-results_panel {{
        background: rgba(6, 9, 15, 0.9);
    }}
    .st-key-hero_panel h1, .st-key-hero_panel h2, .st-key-hero_panel h3,
    .st-key-hero_panel p, .st-key-hero_panel span, .st-key-hero_panel label,
    .st-key-hero_panel small,
    .st-key-results_panel h1, .st-key-results_panel h2, .st-key-results_panel h3,
    .st-key-results_panel p, .st-key-results_panel span, .st-key-results_panel label,
    .st-key-results_panel small {{
        color: #f2f4f8 !important;
    }}
    .st-key-hero_panel [data-testid="stTextInputRootElement"] {{
        background-color: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }}
    .st-key-hero_panel input {{
        color: #f2f4f8 !important;
        background: transparent !important;
        caret-color: #f2f4f8 !important;
    }}
    .st-key-hero_panel input::placeholder {{
        color: rgba(242, 244, 248, 0.55) !important;
    }}
    @media (max-width: 600px) {{
        .st-key-hero_panel, .st-key-results_panel {{
            padding: 1.1rem 1.25rem;
            border-radius: 14px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="hero_panel"):
    st.title("PortAventura Ride Planner")
    st.caption(
        "Describe your visit in plain language -- arrival/end time, rider heights, "
        "and what you want to avoid or prioritize."
    )

    request_text = st.text_input(
        "Your request",
        placeholder='e.g. "We arrive at 10am, leave by 4pm, no water rides, my kid is 120cm"',
    )
    st.caption(
        "Good requests mention a time window, rider heights, or ride types to avoid/prefer -- "
        'e.g. "family day from opening, nothing scary, my son is 105cm."'
    )
    run_clicked = st.button("Plan my day", type="primary")

if run_clicked:
    with st.container(key="results_panel"):
        if not request_text.strip():
            st.warning("Enter a request first.")
            st.stop()

        with st.spinner("Reading your request..."):
            try:
                constraints = extract_constraints(request_text)
            except ConstraintValidationError as e:
                st.error(f"Couldn't parse that request into valid constraints: {e}")
                st.stop()
            except RuntimeError as e:
                st.error(f"Configuration error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong contacting the extraction model: {e}")
                st.stop()

        is_fallback = not constraints.is_planning_request or _no_meaningful_constraints(constraints)
        if is_fallback:
            st.info(
                "Didn't find any scheduling details in that request (an arrival/end time, "
                "rider heights, or ride preferences). Try something like:\n\n"
                '- "We arrive at 10am, leave by 4pm, no water rides."\n'
                '- "My daughter is 115cm, avoid big drops, we love coasters."\n'
                '- "Family day from opening, keep it gentle."\n\n'
                "Showing a default full-day plan below instead."
            )

        with st.expander("Extracted constraints"):
            st.json(dataclasses.asdict(constraints))

        try:
            rides = get_rides()
            model = get_model()
        except lgb.basic.LightGBMError:
            st.error(
                "Model file not found at models/wait_time_lgbm.txt -- "
                "run `python scripts/train_wait_time_model.py` first."
            )
            st.stop()

        with st.spinner("Building your schedule..."):
            try:
                schedule = build_schedule(constraints, rides=rides, booster=model)
            except Exception as e:
                st.error(f"Couldn't build a schedule: {e}")
                st.stop()

        st.subheader("Default full-day plan" if is_fallback else "Your schedule")

        if not schedule:
            st.warning("No rides fit those constraints within the time window. Try relaxing something.")
        else:
            table = pd.DataFrame(
                [
                    {
                        "Time": stop["time"],
                        "Ride": stop["ride"],
                        "Zone": stop["zone"],
                        "Predicted wait (min)": stop["predicted_wait_min"],
                    }
                    for stop in schedule
                ]
            )
            st.dataframe(table, hide_index=True, use_container_width=True)

            total_wait = sum(stop["predicted_wait_min"] for stop in schedule)
            st.metric("Total predicted queue minutes", f"{total_wait:.1f}")
