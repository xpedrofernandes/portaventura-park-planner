"""Streamlit UI for the natural-language ride planner.

Run with:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import dataclasses
import pathlib
import sys

import lightgbm as lgb
import pandas as pd
import streamlit as st

# extract.py/planner.py use plain sibling imports (e.g. "from tags import
# ALLOWED_TAGS") that only resolve when src/ itself is on sys.path -- add it
# before importing, since this file lives at the project root.
SRC_DIR = pathlib.Path(__file__).resolve().parent / "src"
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


st.title("PortAventura Ride Planner")
st.caption(
    "Describe your visit in plain language -- arrival/end time, rider heights, "
    "and what you want to avoid or prioritize."
)

request_text = st.text_input(
    "Your request",
    placeholder='e.g. "We arrive at 10am, leave by 4pm, no water rides, my kid is 120cm"',
)
run_clicked = st.button("Plan my day", type="primary")

if run_clicked:
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
