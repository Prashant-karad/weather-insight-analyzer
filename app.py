"""
Weather Insight Analyzer  -  DAA Mini Project (UI)
Upload a weather CSV -> C++ engine computes covariance with
sequential / multithreaded matrix multiplication -> this app
turns the results into insights, a predictor and a benchmark.

Run:  streamlit run app.py
"""
import os
import subprocess
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

CPP_SOURCE = "weather_analyzer.cpp"
BINARY = "./weather_analyzer.exe" if os.name == "nt" else "./weather_analyzer"
OUT_DIR = "results"

st.set_page_config(page_title="Weather Insight Analyzer", page_icon="🌦️", layout="wide")


# ---------------- Backend helpers ----------------
def compile_engine():
    """Compile the C++ engine once if the binary does not exist."""
    if os.path.exists(BINARY):
        return True, ""
    cmd = ["g++", "-O2", "-std=c++17", "-pthread", CPP_SOURCE, "-o", BINARY]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def run_engine(csv_path):
    r = subprocess.run([BINARY, csv_path, OUT_DIR], capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


def load(name, **kw):
    path = os.path.join(OUT_DIR, name)
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def strength(r):
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "no clear"


# ---------------- Sidebar: input ----------------
st.sidebar.title("🌦️ Weather Insight Analyzer")
st.sidebar.write("Upload a weather CSV. Numeric columns are analysed; text columns like Date or City are ignored.")
uploaded = st.sidebar.file_uploader("Weather CSV file", type=["csv"])
sample = st.sidebar.selectbox(
    "...or use a sample file",
    ["sample_weather.csv  (1 city, 365 days - small)", "big_weather.csv  (10 cities, 20,000 hours - big)"],
)
sample_file = sample.split()[0]

if st.sidebar.button("▶ Analyze", type="primary", use_container_width=True):
    ok, err = compile_engine()
    if not ok:
        st.sidebar.error("Could not compile the C++ engine:\n" + err)
    else:
        if uploaded is not None:
            tmp = os.path.join(tempfile.gettempdir(), "uploaded_weather.csv")
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            csv_path, st.session_state["source"] = tmp, uploaded.name
        elif os.path.exists(sample_file):
            csv_path, st.session_state["source"] = sample_file, sample_file
        else:
            csv_path = None
            st.sidebar.warning("Please upload a CSV file first.")
        if csv_path:
            with st.spinner("Running C++ engine..."):
                ok, log = run_engine(csv_path)
            st.session_state["log"] = log
            st.session_state["raw"] = pd.read_csv(csv_path)
            st.session_state["done"] = ok
            if not ok:
                st.sidebar.error(log)

# ---------------- Main page ----------------
st.title("🌦️ Weather Insight Analyzer")
if not st.session_state.get("done"):
    st.info("👈 Upload a weather CSV (or choose a sample file) and click **Analyze**.")
    st.markdown(
        "**What you get:** a summary of your weather data, which factors move together, "
        "a what-if predictor (e.g. *if humidity is 85%, how much rain?*), and a speed "
        "comparison of sequential vs multithreaded matrix multiplication."
    )
    st.stop()

info = load("info.csv").iloc[0]
stats = load("statistics.csv")
corr = load("correlation.csv", index_col=0)
cov = load("covariance.csv", index_col=0)
reg = load("regression.csv")
timing = load("timing.csv")
raw = st.session_state["raw"]
features = list(corr.columns)

st.caption(f"File: **{st.session_state['source']}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records analysed", int(info["Rows"]))
c2.metric("Weather factors", int(info["Columns"]))
c3.metric("Invalid rows skipped", int(info["SkippedRows"]))
c4.metric("CPU threads used", int(info["CPUThreads"]))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Summary", "🔗 Relationships", "🔮 What-if Predictor", "⚡ Performance", "📁 Data & Downloads"]
)

# ---- Tab 1: Summary ----
with tab1:
    st.subheader("Weather summary")
    show = stats.copy()
    show["Range"] = show["Max"] - show["Min"]
    show["Variability %"] = (show["StdDev"] / show["Mean"].abs().replace(0, pd.NA) * 100).round(1)
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)

    st.subheader("Key insights")
    most_var = show.dropna(subset=["Variability %"]).sort_values("Variability %").iloc[-1]
    most_stable = show.dropna(subset=["Variability %"]).sort_values("Variability %").iloc[0]
    st.markdown(f"- 📈 **{most_var['Feature']}** changes the most (variability {most_var['Variability %']}%) — hardest to plan around.")
    st.markdown(f"- 📉 **{most_stable['Feature']}** is the most stable (variability {most_stable['Variability %']}%).")

    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            pairs.append((a, b, corr.loc[a, b]))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    for a, b, r in pairs[:3]:
        direction = "goes up" if r > 0 else "goes down"
        st.markdown(f"- 🔗 When **{a}** rises, **{b}** usually {direction} ({strength(r)} link, r = {r:.2f}).")

    pick = st.selectbox("See how a factor changes across records", features)
    st.line_chart(raw[pick].reset_index(drop=True))

# ---- Tab 2: Relationships ----
with tab2:
    st.subheader("Correlation heatmap")
    st.caption("+1 = always rise together, −1 = one rises when the other falls, 0 = unrelated.")
    long = corr.reset_index().melt(id_vars="Feature", var_name="With", value_name="r")
    heat = alt.Chart(long).mark_rect().encode(
        x=alt.X("With:N", title=None), y=alt.Y("Feature:N", title=None),
        color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
        tooltip=["Feature", "With", alt.Tooltip("r:Q", format=".2f")],
    )
    text = alt.Chart(long).mark_text(fontSize=12).encode(
        x="With:N", y="Feature:N", text=alt.Text("r:Q", format=".2f")
    )
    chart = heat + text if len(features) <= 12 else heat   # numbers only fit on small matrices
    st.altair_chart(chart.properties(height=420 if len(features) <= 12 else 900), use_container_width=True)

    st.subheader("All relationships in plain words")
    rel = pd.DataFrame(
        [{"Factor A": a, "Factor B": b, "Correlation": round(r, 2),
          "Meaning": f"{strength(r).capitalize()} {'positive' if r > 0 else 'negative'} link"}
         for a, b, r in pairs]
    )
    st.dataframe(rel, use_container_width=True, hide_index=True)

    with st.expander("Covariance matrix (computed by C++ engine)"):
        st.dataframe(cov.round(3), use_container_width=True)

# ---- Tab 3: Predictor ----
with tab3:
    st.subheader("What-if predictor")
    st.caption("Uses the covariance matrix: slope = Cov(X,Y) / Var(X). Best when the two factors are strongly linked.")
    a, b = st.columns(2)
    x_feat = a.selectbox("I know this value", features, index=0)
    y_choices = [f for f in features if f != x_feat]
    best = reg[reg["X"] == x_feat].sort_values("Correlation", key=abs, ascending=False)["Y"].iloc[0]
    y_feat = b.selectbox("Predict this", y_choices, index=y_choices.index(best))

    row = reg[(reg["X"] == x_feat) & (reg["Y"] == y_feat)].iloc[0]
    xs = stats.set_index("Feature").loc[x_feat]
    val = st.slider(f"{x_feat} value", float(xs["Min"]), float(xs["Max"]), float(xs["Mean"]))
    pred = row["Slope"] * val + row["Intercept"]
    r = row["Correlation"]

    m1, m2 = st.columns(2)
    m1.metric(f"Predicted {y_feat}", f"{pred:.2f}")
    m2.metric("Reliability", f"{strength(r).capitalize()} (|r| = {abs(r):.2f})")
    if abs(r) < 0.4:
        st.warning("These two factors are only weakly linked, so treat this prediction as a rough guess.")

    pts = alt.Chart(raw).mark_circle(opacity=0.5).encode(x=x_feat, y=y_feat)
    line_df = pd.DataFrame({x_feat: [xs["Min"], xs["Max"]]})
    line_df[y_feat] = row["Slope"] * line_df[x_feat] + row["Intercept"]
    line = alt.Chart(line_df).mark_line(color="red").encode(x=x_feat, y=y_feat)
    you = alt.Chart(pd.DataFrame({x_feat: [val], y_feat: [pred]})).mark_point(
        size=200, color="orange", filled=True).encode(x=x_feat, y=y_feat)
    st.altair_chart((pts + line + you).properties(height=380), use_container_width=True)
    st.caption(f"Formula: {y_feat} = {row['Slope']:.4f} × {x_feat} + {row['Intercept']:.4f}")

# ---- Tab 4: Performance ----
with tab4:
    st.subheader("Sequential vs multithreaded matrix multiplication")
    st.caption("Operation: Xcᵀ × Xc (the core of the covariance calculation). Complexity O(m² · n).")
    seq = timing.loc[timing["Method"] == "Sequential", "Time_ms"].iloc[0]
    timing["Speedup (x)"] = (seq / timing["Time_ms"]).round(2)
    st.dataframe(timing.round(4), use_container_width=True, hide_index=True)
    st.bar_chart(timing.set_index("Method")["Time_ms"])
    if info["ResultsMatch"] == "Yes":
        st.success("✅ All three methods produced the same matrix — the parallel versions are correct.")
    else:
        st.error("❌ Results differ between methods.")
    fastest = timing.sort_values("Time_ms").iloc[0]["Method"]
    st.info(
        f"Fastest on this file: **{fastest.replace('_', ' ')}**. For small files, creating threads costs "
        "more time than it saves, so sequential can win. Upload a file with thousands of rows "
        "and more columns to see threading pull ahead."
    )

# ---- Tab 5: Data & downloads ----
with tab5:
    st.subheader("Uploaded data (first 100 rows)")
    st.dataframe(raw.head(100), use_container_width=True)
    st.subheader("Download results")
    for f in ["statistics.csv", "correlation.csv", "covariance.csv", "regression.csv", "timing.csv"]:
        path = os.path.join(OUT_DIR, f)
        with open(path, "rb") as fh:
            st.download_button(f"⬇ {f}", fh, file_name=f, key=f)
    with st.expander("C++ engine log"):
        st.code(st.session_state.get("log", ""))
