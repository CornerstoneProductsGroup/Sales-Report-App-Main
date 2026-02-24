import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta

APP_TITLE = "Sales Report App"

# Default Paths
DATA_DIR = Path("data")
DEFAULT_SALES_STORE = DATA_DIR / "sales_store.csv"
DEFAULT_VENDOR_MAP = Path("vendor_map.xlsx")
DEFAULT_PRICE_HISTORY = DATA_DIR / "price_history.csv"

st.set_page_config(page_title=APP_TITLE, layout="wide")

# -------------------------
# Helpers
# -------------------------
def fmt_int(v):
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return "—"

def fmt_currency(v):
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "—"

def fmt_currency_signed(v):
    try:
        v = float(v)
        sign = "-" if v < 0 else ""
        return f"{sign}${abs(v):,.2f}"
    except Exception:
        return "—"

def _color(v):
    try:
        v = float(v)
    except Exception:
        return "#666666"
    if v > 0:
        return "#0b7a0b"
    if v < 0:
        return "#b00020"
    return "#666666"

def _table_height(df: pd.DataFrame, row_px=28, header_px=38, max_px=950, min_px=180):
    if df is None:
        return min_px
    return int(max(min_px, min(max_px, header_px + len(df) * row_px)))

def add_totals_row(df: pd.DataFrame, label_col: str):
    if df.empty:
        return df
    num_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    total = {label_col: "TOTAL"}
    for c in df.columns:
        if c in num_cols:
            total[c] = float(df[c].sum())
        elif c != label_col:
            total[c] = ""
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)

def style_totals_row(styler, label_col):
    def _bold_total(row):
        if str(row[label_col]) == "TOTAL":
            return ["font-weight:700;"] * len(row)
        return [""] * len(row)
    return styler.apply(_bold_total, axis=1)

@st.cache_data(show_spinner=False)
def load_sales_store(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"])
    df = pd.read_csv(p)
    # normalize columns
    for c in ["Retailer","SKU","SourceFile"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["Units","UnitPrice"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["StartDate","EndDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "UnitPrice" not in df.columns:
        df["UnitPrice"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def load_vendor_map(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["SKU","Vendor","Retailer"])
    # Expect tabs per retailer; start with first sheet if unknown
    try:
        xls = pd.ExcelFile(p)
        sheet = xls.sheet_names[0]
        vm = pd.read_excel(p, sheet_name=sheet)
    except Exception:
        vm = pd.read_excel(p)
    # Try to find SKU and Vendor columns
    cols = {c.lower().strip(): c for c in vm.columns}
    sku_col = cols.get("sku") or cols.get("sku #") or cols.get("sku#") or cols.get("sku number") or cols.get("sku_num")
    vendor_col = cols.get("vendor") or cols.get("vendor name") or cols.get("vendorname")
    if not sku_col:
        # fallback: first column
        sku_col = vm.columns[0]
    if not vendor_col:
        vendor_col = "Vendor"
        vm[vendor_col] = ""
    out = vm[[sku_col, vendor_col]].copy()
    out.columns = ["SKU","Vendor"]
    out["SKU"] = out["SKU"].astype(str).str.strip()
    out["Vendor"] = out["Vendor"].astype(str).str.strip()
    out = out.dropna(subset=["SKU"]).drop_duplicates()
    return out

def attach_vendor(df: pd.DataFrame, vendor_map: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["Vendor"] = ""
        return df
    if vendor_map.empty:
        df["Vendor"] = ""
        return df
    return df.merge(vendor_map, on="SKU", how="left")

def compute_sales(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Sales" not in d.columns:
        d["Sales"] = np.nan
    # If UnitPrice missing, keep Sales as NaN -> treated as 0 unless price history is added later.
    d["Sales"] = d["Units"].fillna(0) * d["UnitPrice"].fillna(0)
    return d

def week_label(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")

def get_all_weeks(df: pd.DataFrame):
    if df.empty:
        return []
    w = sorted(df["StartDate"].dropna().dt.date.unique().tolist())
    return w

def select_weeks(df: pd.DataFrame, n_weeks: int):
    weeks = get_all_weeks(df)
    if not weeks:
        return []
    n = int(n_weeks)
    return weeks[-n:] if len(weeks) >= n else weeks

def month_year_options(df: pd.DataFrame):
    if df.empty:
        return []
    p = df["StartDate"].dropna().dt.to_period("M")
    months = sorted(p.unique().tolist())
    return [m.to_timestamp().strftime("%B %Y") for m in months]

def filter_month_year(df: pd.DataFrame, label: str):
    # label like "January 2026"
    try:
        dt = datetime.strptime(label, "%B %Y")
    except Exception:
        return df.iloc[0:0].copy()
    m, y = dt.month, dt.year
    return df[(df["StartDate"].dt.year == y) & (df["StartDate"].dt.month == m)].copy()

def safe_pct(a, b):
    # (a-b)/b
    if b == 0:
        return np.nan
    return (a-b)/b

# -------------------------
# Load data
# -------------------------
st.title(APP_TITLE)

with st.sidebar:
    st.header("Data")
    sales_path = st.text_input("Sales store file", value=str(DEFAULT_SALES_STORE))
    vendor_path = st.text_input("Vendor map file", value=str(DEFAULT_VENDOR_MAP))
    weeks_shown = st.selectbox("Weeks shown", options=[4,6,8,13,26,52], index=2)
    st.caption("Weeks shown controls most tables (latest N weeks).")

df_raw = load_sales_store(sales_path)
vm = load_vendor_map(vendor_path)
df = attach_vendor(df_raw, vm)
df = compute_sales(df)

# core time fields
if not df.empty:
    df["Year"] = df["StartDate"].dt.year.astype("Int64")
    df["Week"] = df["StartDate"].dt.date
else:
    df["Year"] = pd.Series(dtype="Int64")
    df["Week"] = pd.Series(dtype="object")

latest_weeks = select_weeks(df, weeks_shown)
df_recent = df[df["Week"].isin(latest_weeks)].copy() if latest_weeks else df.iloc[0:0].copy()

# Tabs (kept aligned with your workflow)
tabs = st.tabs([
    "Summary",
    "Totals Dashboard",
    "Top SKUs",
    "Executive Summary",
    "Comparison",
    "WoW Exceptions",
    "SKU Intelligence",
    "Lost Sales Detector",
    "Year Summary",
    "Alerts",
    "Data Management",
])

# -------------------------
# Summary
# -------------------------
with tabs[0]:
    st.subheader("Summary")

    if df_recent.empty:
        st.info("No recent data found. Check your sales_store.csv path.")
    else:
        total_units = float(df_recent["Units"].sum())
        total_sales = float(df_recent["Sales"].sum())
        c1, c2 = st.columns(2)
        c1.metric("Units (shown weeks)", fmt_int(total_units))
        c2.metric("Sales (shown weeks)", fmt_currency(total_sales))

        # by retailer
        by_retailer = df_recent.groupby("Retailer", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
        by_retailer = by_retailer.sort_values("Sales", ascending=False)
        by_retailer = add_totals_row(by_retailer, "Retailer")
        sty = by_retailer.style.format({"Units": fmt_int, "Sales": fmt_currency})
        sty = style_totals_row(sty, "Retailer")
        st.markdown("### Retailer totals")
        st.dataframe(sty, use_container_width=True, hide_index=True, height=_table_height(by_retailer))

# -------------------------
# Totals Dashboard
# -------------------------
with tabs[1]:
    st.subheader("Totals Dashboard")

    if df_recent.empty:
        st.info("No data.")
    else:
        # Average window options: rolling + month-year
        month_year_list = month_year_options(df)
        avg_options = ["4 weeks","6 weeks","8 weeks","13 weeks","26 weeks","52 weeks"] + month_year_list

        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            view = st.selectbox("View", ["Summary totals","Retailer totals","Vendor totals","SKU totals"], index=0)
        with c2:
            avg_window = st.selectbox("Average window", options=avg_options, index=0)
        with c3:
            metric = st.selectbox("Metric", ["Units","Sales"], index=1)

        # current period = shown weeks
        cur = df_recent.copy()

        def _avg_frame():
            if avg_window.endswith("weeks"):
                n = int(avg_window.split()[0])
                w = select_weeks(df, n)
                return df[df["Week"].isin(w)].copy()
            # Month Year
            return filter_month_year(df, avg_window)

        avg_df = _avg_frame()

        def build_table(group_col):
            g_cur = cur.groupby(group_col, as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
            g_avg = avg_df.groupby(group_col, as_index=False).agg(AvgUnits=("Units","mean"), AvgSales=("Sales","mean"))
            out = g_cur.merge(g_avg, on=group_col, how="left").fillna(0.0)
            out["Δ Units"] = out["Units"] - out["AvgUnits"]
            out["Δ Sales"] = out["Sales"] - out["AvgSales"]
            out["% Units"] = out.apply(lambda r: (r["Δ Units"]/r["AvgUnits"]) if r["AvgUnits"] else np.nan, axis=1)
            out["% Sales"] = out.apply(lambda r: (r["Δ Sales"]/r["AvgSales"]) if r["AvgSales"] else np.nan, axis=1)
            # totals row
            out = out.sort_values(metric, ascending=False, kind="mergesort")
            out = add_totals_row(out, group_col)
            return out

        if view == "Summary totals":
            st.markdown("### Summary totals")
            rows = [("Retailer","Retailer"),("Vendor","Vendor"),("SKU","SKU")]
            for label, col in rows:
                st.markdown(f"**{label}**")
                t = build_table(col)
                sty = t.style.format({
                    "Units": fmt_int, "Sales": fmt_currency,
                    "AvgUnits": lambda v: f"{float(v):,.1f}" if pd.notna(v) else "—",
                    "AvgSales": fmt_currency,
                    "Δ Units": fmt_int, "Δ Sales": fmt_currency,
                    "% Units": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                    "% Sales": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                }).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Units","Δ Sales"])
                sty = style_totals_row(sty, col)
                st.dataframe(sty, use_container_width=True, hide_index=True, height=_table_height(t))
        else:
            group_col = {"Retailer totals":"Retailer","Vendor totals":"Vendor","SKU totals":"SKU"}[view]
            t = build_table(group_col)
            st.dataframe(
                style_totals_row(
                    t.style.format({
                        "Units": fmt_int, "Sales": fmt_currency,
                        "AvgUnits": lambda v: f"{float(v):,.1f}" if pd.notna(v) else "—",
                        "AvgSales": fmt_currency,
                        "Δ Units": fmt_int, "Δ Sales": fmt_currency,
                        "% Units": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                        "% Sales": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                    }).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Units","Δ Sales"]),
                    group_col
                ),
                use_container_width=True,
                hide_index=True,
                height=_table_height(t)
            )

# -------------------------
# Top SKUs
# -------------------------
with tabs[2]:
    st.subheader("Top SKUs")

    if df_recent.empty:
        st.info("No data.")
    else:
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            basis = st.selectbox("Rank by", ["Units","Sales"], index=0)
        with c2:
            top_n = st.number_input("Top N", min_value=5, max_value=500, value=50, step=5)
        with c3:
            min_threshold = st.number_input(f"Minimum {basis}", min_value=0.0, value=0.0, step=1.0)

        g = df_recent.groupby(["SKU","Vendor"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"), Retailers=("Retailer","nunique"))
        g = g.sort_values(basis, ascending=False, kind="mergesort")
        # threshold filter requested
        if min_threshold > 0:
            g = g[g[basis] >= float(min_threshold)]
        g = g.head(int(top_n)).copy()

        st.caption("Top N means: show the top N rows after sorting (and after applying your minimum threshold).")
        st.dataframe(
            g.style.format({"Units": fmt_int, "Sales": fmt_currency}),
            use_container_width=True,
            hide_index=True,
            height=_table_height(g)
        )

        st.markdown("### SKU lookup")
        q = st.text_input("Search SKU contains", value="")
        if q.strip():
            d2 = df_recent[df_recent["SKU"].astype(str).str.contains(q.strip(), case=False, na=False)].copy()
            if d2.empty:
                st.info("No matches in shown weeks.")
            else:
                g2 = d2.groupby(["Retailer","SKU","Vendor"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                g2 = g2.sort_values("Sales", ascending=False)
                st.dataframe(g2.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True)

# -------------------------
# Executive Summary
# -------------------------
with tabs[3]:
    st.subheader("Executive Summary")

    if df.empty:
        st.info("No data.")
    else:
        scope = st.selectbox("Scope", ["All","Retailer","Vendor"], index=0)
        d_all = df.copy()

        if scope == "Retailer":
            opts = sorted([x for x in d_all["Retailer"].dropna().unique().tolist() if str(x).strip()])
            pick = st.selectbox("Retailer", options=opts)
            d_base = d_all[d_all["Retailer"] == pick].copy()
        elif scope == "Vendor":
            opts = sorted([x for x in d_all["Vendor"].dropna().unique().tolist() if str(x).strip()])
            pick = st.selectbox("Vendor", options=opts)
            d_base = d_all[d_all["Vendor"] == pick].copy()
        else:
            d_base = d_all.copy()

        years = sorted(d_base["Year"].dropna().unique().tolist())
        if not years:
            st.info("No dated rows.")
        else:
            pick_year = st.selectbox("Year", options=years, index=len(years)-1)

            d = d_base[d_base["Year"] == int(pick_year)].copy()

            # When scope = All: show SKU totals across all platforms (not broken out)
            if scope == "All":
                st.markdown("### SKU totals (all retailers)")
                sku = d.groupby(["SKU","Vendor"], as_index=False).agg(
                    Retailers=("Retailer","nunique"),
                    Units=("Units","sum"),
                    Sales=("Sales","sum")
                ).sort_values("Sales", ascending=False)
                st.dataframe(
                    sku.style.format({"Units": fmt_int, "Sales": fmt_currency}),
                    use_container_width=True,
                    hide_index=True,
                    height=_table_height(sku, max_px=950)
                )
            else:
                # monthly totals
                st.markdown("### Monthly totals")
                d["Month"] = d["StartDate"].dt.to_period("M").dt.to_timestamp()
                mt = d.groupby("Month", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Month")
                mt["Month"] = mt["Month"].dt.strftime("%b %Y")
                st.dataframe(mt.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True)

                st.markdown("### Vendor mix" if scope == "Retailer" else "### Retailer mix")
                mix_col = "Vendor" if scope == "Retailer" else "Retailer"
                mix = d.groupby(mix_col, as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Sales", ascending=False)
                st.dataframe(mix.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True, height=_table_height(mix))

                # top/bottom skus
                st.markdown("### Top / Bottom SKUs")
                g = d.groupby(["Retailer","SKU","Vendor"], as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum"))
                top10 = g.sort_values("Sales", ascending=False).head(10)
                bot10 = g.sort_values("Sales", ascending=True).head(10)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top 10**")
                    st.dataframe(top10.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Bottom 10**")
                    st.dataframe(bot10.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True)

            # Always show yearly totals at bottom (all years for this scope)
            st.divider()
            st.markdown("### Yearly totals (all years)")
            yt = d_base.groupby("Year", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Year")
            st.dataframe(yt.style.format({"Units": fmt_int, "Sales": fmt_currency}), use_container_width=True, hide_index=True)

# -------------------------
# Comparison
# -------------------------
with tabs[4]:
    st.subheader("Comparison")

    if df.empty:
        st.info("No data.")
    else:
        compare_area = st.radio("Comparison area", ["Retailer / Vendor", "SKU Comparison"], horizontal=True)
        mode = st.radio("Compare type", ["A vs B (Months)", "A vs B (Years)", "Multi-year (high/low highlight)"], horizontal=True)

        d = df.copy()

        # selections
        d["MonthP"] = d["StartDate"].dt.to_period("M")
        months = sorted(d["MonthP"].dropna().unique().tolist())
        month_labels = [m.to_timestamp().strftime("%B %Y") for m in months]
        label_to_period = dict(zip(month_labels, months))
        years = sorted(d["Year"].dropna().unique().tolist())

        if compare_area == "Retailer / Vendor":
            by = st.selectbox("Compare by", ["Retailer","Vendor"], index=0)
            options = sorted([x for x in d[by].dropna().unique().tolist() if str(x).strip()])
            sel = st.multiselect(f"Limit to {by}(s) (optional)", options=options)
        else:
            by = "SKU"
            sel = []

        if mode == "A vs B (Months)":
            c1, c2 = st.columns(2)
            with c1:
                a_pick = st.multiselect("Selection A", options=month_labels, default=month_labels[-1:] if month_labels else [])
            with c2:
                b_pick = st.multiselect("Selection B", options=month_labels, default=month_labels[-2:-1] if len(month_labels)>=2 else [])
            a_periods = [label_to_period[x] for x in a_pick if x in label_to_period]
            b_periods = [label_to_period[x] for x in b_pick if x in label_to_period]
            da = d[d["MonthP"].isin(a_periods)].copy()
            db = d[d["MonthP"].isin(b_periods)].copy()
            label_a = " + ".join(a_pick) if a_pick else "A"
            label_b = " + ".join(b_pick) if b_pick else "B"

        elif mode == "A vs B (Years)":
            c1, c2 = st.columns(2)
            with c1:
                years_a = st.multiselect("Selection A (years)", options=years, default=years[-2:-1] if len(years)>=2 else years)
            with c2:
                years_b = st.multiselect("Selection B (years)", options=years, default=years[-1:] if years else [])
            da = d[d["Year"].isin([int(y) for y in years_a])].copy()
            db = d[d["Year"].isin([int(y) for y in years_b])].copy()
            label_a = " + ".join([str(y) for y in years_a]) if years_a else "A"
            label_b = " + ".join([str(y) for y in years_b]) if years_b else "B"
        else:
            years_pick = st.multiselect("Years to view (2..5)", options=years, default=years[-3:] if len(years)>=3 else years)
            years_pick = [int(y) for y in years_pick][:5]
            metric = st.selectbox("Highlight based on", ["Sales","Units"], index=0)
            if len(years_pick) < 2:
                st.info("Pick at least 2 years.")
                st.stop()
            dd = d[d["Year"].isin(years_pick)].copy()
            if sel and by != "SKU":
                dd = dd[dd[by].isin(sel)]
            group_col = by
            pieces = []
            for y in years_pick:
                gy = dd[dd["Year"] == int(y)].groupby(group_col, as_index=False).agg(**{
                    f"Units_{y}": ("Units","sum"),
                    f"Sales_{y}": ("Sales","sum"),
                })
                pieces.append(gy)
            out = pieces[0]
            for p in pieces[1:]:
                out = out.merge(p, on=group_col, how="outer")
            out = out.fillna(0.0)
            # totals row
            total = {group_col: "TOTAL"}
            for c in out.columns:
                if c == group_col: continue
                total[c] = float(out[c].sum())
            out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

            cols = [group_col]
            for y in years_pick:
                cols += [f"Units_{y}", f"Sales_{y}"]
            disp = out[cols].copy()

            metric_cols = [f"{metric}_{y}" for y in years_pick]
            spark_chars = ["▁","▂","▃","▄","▅","▆","▇","█"]

            def _spark(vals):
                vals = [float(v) if pd.notna(v) else np.nan for v in vals]
                if len(vals) == 0 or all(pd.isna(v) for v in vals):
                    return ""
                vmin = np.nanmin(vals); vmax = np.nanmax(vals)
                if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                    return "▁" * len(vals)
                out_s = []
                for v in vals:
                    if pd.isna(v):
                        out_s.append(" ")
                    else:
                        t = (v - vmin) / (vmax - vmin)
                        idx = int(round(t * (len(spark_chars) - 1)))
                        idx = max(0, min(len(spark_chars) - 1, idx))
                        out_s.append(spark_chars[idx])
                return "".join(out_s)

            def _pct_change(a, b):
                try:
                    a=float(a); b=float(b)
                except Exception:
                    return np.nan
                if a == 0:
                    return np.nan
                return (b-a)/a

            def _cagr(a, b, periods):
                try:
                    a=float(a); b=float(b)
                except Exception:
                    return np.nan
                if a <= 0 or b <= 0 or periods <= 0:
                    return np.nan
                return (b/a)**(1.0/periods) - 1.0

            # insights on metric columns (exclude TOTAL)
            mcols = [f"{metric}_{y}" for y in years_pick]
            if all(c in disp.columns for c in mcols):
                series_vals = disp[mcols].copy()
                disp["Spark"] = series_vals.apply(lambda r: _spark(r.tolist()), axis=1)
                first_col, last_col = mcols[0], mcols[-1]
                pct = series_vals.apply(lambda r: _pct_change(r[first_col], r[last_col]), axis=1)
                disp["Trend"] = np.where(pct.isna(), "—", np.where(pct > 0, "↑", np.where(pct < 0, "↓", "→")))
                disp["Trend %"] = pct
                periods = max(1, len(mcols)-1)
                disp["CAGR %"] = series_vals.apply(lambda r: _cagr(r[first_col], r[last_col], periods), axis=1)
                is_total = disp[group_col].astype(str) == "TOTAL"
                for c in ["Spark","Trend","Trend %","CAGR %"]:
                    disp.loc[is_total, c] = ""

            def _hl_minmax(row):
                styles = [""] * len(row)
                if str(row.iloc[0]) == "TOTAL":
                    return styles
                vals = []
                idxs = []
                for j, c in enumerate(disp.columns):
                    if c in metric_cols:
                        try:
                            v = float(row[c])
                        except Exception:
                            v = np.nan
                        vals.append(v); idxs.append(j)
                if not vals:
                    return styles
                vmin = np.nanmin(vals); vmax=np.nanmax(vals)
                if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
                    return styles
                for v, j in zip(vals, idxs):
                    if np.isclose(v, vmax):
                        styles[j] = "background-color: rgba(0,200,0,0.18); font-weight:600;"
                    elif np.isclose(v, vmin):
                        styles[j] = "background-color: rgba(220,0,0,0.14);"
                return styles

            fmt = {}
            for c in disp.columns:
                if c.startswith("Units_"):
                    fmt[c] = fmt_int
                if c.startswith("Sales_"):
                    fmt[c] = fmt_currency
            if "Trend %" in disp.columns:
                fmt["Trend %"] = lambda v: (f"{float(v)*100:.1f}%" if (v is not None and pd.notna(v) and str(v).strip() not in {"","—"}) else "—")
            if "CAGR %" in disp.columns:
                fmt["CAGR %"] = lambda v: (f"{float(v)*100:.1f}%" if (v is not None and pd.notna(v) and str(v).strip() not in {"","—"}) else "—")

            st.dataframe(disp.style.format(fmt).apply(_hl_minmax, axis=1), use_container_width=True, hide_index=True, height=_table_height(disp, max_px=1100))

            # movers between earliest and latest year for SKU totals (always SKU-based)
            st.divider()
            y0, y1 = years_pick[0], years_pick[-1]
            da = d[d["Year"] == int(y1)].copy()
            db = d[d["Year"] == int(y0)].copy()
            # show movers by SKU
            sa = da.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
            sb = db.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
            mv = sa.merge(sb, on="SKU", how="outer").fillna(0.0)
            mv["Sales_Diff"] = mv["Sales_A"] - mv["Sales_B"]
            mv["Units_Diff"] = mv["Units_A"] - mv["Units_B"]
            top_up = mv.sort_values("Sales_Diff", ascending=False).head(10)
            top_dn = mv.sort_values("Sales_Diff", ascending=True).head(10)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Top 10 Increased SKUs ({y1} vs {y0})**")
                st.dataframe(top_up.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Units_Diff":fmt_int,"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Sales_Diff":fmt_currency}), use_container_width=True, hide_index=True)
            with c2:
                st.markdown(f"**Top 10 Decreased SKUs ({y1} vs {y0})**")
                st.dataframe(top_dn.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Units_Diff":fmt_int,"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Sales_Diff":fmt_currency}), use_container_width=True, hide_index=True)
            st.stop()

        # Apply limits
        if sel and by != "SKU":
            da = da[da[by].isin(sel)]
            db = db[db[by].isin(sel)]

        # A vs B table builder
        ga = da.groupby(by, as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
        gb = db.groupby(by, as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
        out = ga.merge(gb, on=by, how="outer").fillna(0.0)
        out["Units_Diff"] = out["Units_A"] - out["Units_B"]
        out["Sales_Diff"] = out["Sales_A"] - out["Sales_B"]
        out["Units_%"] = out["Units_Diff"] / out["Units_B"].replace(0, np.nan)
        out["Sales_%"] = out["Sales_Diff"] / out["Sales_B"].replace(0, np.nan)

        total = {by:"TOTAL",
                 "Units_A": float(out["Units_A"].sum()),
                 "Sales_A": float(out["Sales_A"].sum()),
                 "Units_B": float(out["Units_B"].sum()),
                 "Sales_B": float(out["Sales_B"].sum())}
        total["Units_Diff"] = total["Units_A"] - total["Units_B"]
        total["Sales_Diff"] = total["Sales_A"] - total["Sales_B"]
        total["Units_%"] = (total["Units_Diff"]/total["Units_B"]) if total["Units_B"] else np.nan
        total["Sales_%"] = (total["Sales_Diff"]/total["Sales_B"]) if total["Sales_B"] else np.nan
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

        disp = out[[by,"Units_A","Sales_A","Units_B","Sales_B","Units_Diff","Units_%","Sales_Diff","Sales_%"]]
        disp = disp.rename(columns={
            "Units_A": f"Units ({label_a})",
            "Sales_A": f"Sales ({label_a})",
            "Units_B": f"Units ({label_b})",
            "Sales_B": f"Sales ({label_b})",
        })

        sty = disp.style.format({
            f"Units ({label_a})": fmt_int,
            f"Units ({label_b})": fmt_int,
            "Units_Diff": fmt_int,
            "Units_%": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
            f"Sales ({label_a})": fmt_currency,
            f"Sales ({label_b})": fmt_currency,
            "Sales_Diff": fmt_currency,
            "Sales_%": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
        }).applymap(lambda v: f"color:{_color(v)};", subset=["Units_Diff","Sales_Diff"])
        sty = style_totals_row(sty, by)

        st.dataframe(sty, use_container_width=True, hide_index=True, height=_table_height(disp, max_px=1100))

        # Top SKU movers always at bottom
        st.divider()
        sa = da.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
        sb = db.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
        mv = sa.merge(sb, on="SKU", how="outer").fillna(0.0)
        mv["Sales_Diff"] = mv["Sales_A"] - mv["Sales_B"]
        mv["Units_Diff"] = mv["Units_A"] - mv["Units_B"]
        top_up = mv.sort_values("Sales_Diff", ascending=False).head(10)
        top_dn = mv.sort_values("Sales_Diff", ascending=True).head(10)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top 10 Increased SKUs ({label_a} vs {label_b})**")
            st.dataframe(top_up.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Units_Diff":fmt_int,"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Sales_Diff":fmt_currency}), use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**Top 10 Decreased SKUs ({label_a} vs {label_b})**")
            st.dataframe(top_dn.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Units_Diff":fmt_int,"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Sales_Diff":fmt_currency}), use_container_width=True, hide_index=True)

# -------------------------
# WoW Exceptions
# -------------------------
with tabs[5]:
    st.subheader("WoW Exceptions")

    if df.empty:
        st.info("No data.")
    else:
        # Scope: total SKUs vs broken out
        view = st.radio("View", ["SKU totals (all retailers)", "SKU by Retailer"], horizontal=True)
        prior_window = st.selectbox("Prior window", options=[4,6,8,13,26,52], index=1)

        # current week = most recent week in ALL data
        all_weeks = get_all_weeks(df)
        if not all_weeks:
            st.info("No weeks.")
        else:
            current_week = all_weeks[-1]
            cur_df = df[df["Week"] == current_week].copy()
            # prior window pulls across years
            n = int(prior_window)
            pos = len(all_weeks)-1
            start = max(0, pos - n)
            prior_weeks = all_weeks[start:pos]
            prior_df = df[df["Week"].isin(prior_weeks)].copy()

            if view == "SKU totals (all retailers)":
                ga = cur_df.groupby(["SKU","Vendor"], as_index=False).agg(CurUnits=("Units","sum"), CurSales=("Sales","sum"))
                gb = prior_df.groupby(["SKU","Vendor"], as_index=False).agg(PriorUnits=("Units","mean"), PriorSales=("Sales","mean"))
                out = ga.merge(gb, on=["SKU","Vendor"], how="outer").fillna(0.0)
            else:
                ga = cur_df.groupby(["Retailer","SKU","Vendor"], as_index=False).agg(CurUnits=("Units","sum"), CurSales=("Sales","sum"))
                gb = prior_df.groupby(["Retailer","SKU","Vendor"], as_index=False).agg(PriorUnits=("Units","mean"), PriorSales=("Sales","mean"))
                out = ga.merge(gb, on=["Retailer","SKU","Vendor"], how="outer").fillna(0.0)

            out["Δ Units"] = out["CurUnits"] - out["PriorUnits"]
            out["Δ Sales"] = out["CurSales"] - out["PriorSales"]
            out["% Units"] = out.apply(lambda r: (r["Δ Units"]/r["PriorUnits"]) if r["PriorUnits"] else np.nan, axis=1)
            out["% Sales"] = out.apply(lambda r: (r["Δ Sales"]/r["PriorSales"]) if r["PriorSales"] else np.nan, axis=1)

            sort_by = st.selectbox("Sort by", ["Δ Sales","Δ Units","% Sales","% Units"], index=0)
            out = out.sort_values(sort_by, ascending=False, kind="mergesort").head(500)

            st.dataframe(
                out.style.format({
                    "CurUnits": fmt_int, "PriorUnits": lambda v: f"{float(v):,.1f}" if pd.notna(v) else "—",
                    "CurSales": fmt_currency, "PriorSales": fmt_currency,
                    "Δ Units": fmt_int, "Δ Sales": fmt_currency,
                    "% Units": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                    "% Sales": lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                }).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Units","Δ Sales"]),
                use_container_width=True,
                hide_index=True,
                height=_table_height(out, max_px=1100)
            )

# -------------------------
# SKU Intelligence (Year vs Year or Month vs Month)
# -------------------------
with tabs[6]:
    st.subheader("SKU Intelligence")

    if df.empty:
        st.info("No data.")
    else:
        mode = st.radio("Compare", ["Year vs Year", "Month vs Month"], horizontal=True)
        years = sorted(df["Year"].dropna().unique().tolist())
        df["MonthLabel"] = df["StartDate"].dt.strftime("%B %Y")
        months = sorted(df["MonthLabel"].dropna().unique().tolist(), key=lambda x: datetime.strptime(x, "%B %Y"))
        metric = st.selectbox("Metric", ["Sales","Units"], index=0)

        if mode == "Year vs Year":
            c1, c2 = st.columns(2)
            with c1:
                yA = st.selectbox("Year A", options=years, index=max(0, len(years)-2))
            with c2:
                yB = st.selectbox("Year B", options=years, index=len(years)-1)
            a = df[df["Year"] == int(yA)].copy()
            b = df[df["Year"] == int(yB)].copy()
            label_a, label_b = str(yA), str(yB)
        else:
            c1, c2 = st.columns(2)
            with c1:
                mA = st.selectbox("Month A", options=months, index=max(0, len(months)-2))
            with c2:
                mB = st.selectbox("Month B", options=months, index=len(months)-1)
            a = df[df["MonthLabel"] == mA].copy()
            b = df[df["MonthLabel"] == mB].copy()
            label_a, label_b = mA, mB

        ga = a.groupby(["SKU","Vendor"], as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"), ActiveWeeks_A=("Week","nunique"))
        gb = b.groupby(["SKU","Vendor"], as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"), ActiveWeeks_B=("Week","nunique"))
        out = ga.merge(gb, on=["SKU","Vendor"], how="outer").fillna(0.0)
        out["Units_Diff"] = out["Units_B"] - out["Units_A"]
        out["Sales_Diff"] = out["Sales_B"] - out["Sales_A"]

        # ActiveWeeks explanation: number of distinct weeks with any records in the period
        out["ActiveWeeks"] = out["ActiveWeeks_B"]  # keep single column for display (current/compare period)

        sort_by = st.selectbox("Sort by", ["Sales_Diff","Units_Diff","Sales_B","Units_B"], index=0)
        out = out.sort_values(sort_by, ascending=False, kind="mergesort").head(500)

        disp = out[["SKU","Vendor","Units_A","Sales_A","Units_B","Sales_B","Units_Diff","Sales_Diff","ActiveWeeks"]]
        disp = disp.rename(columns={
            "Units_A": f"Units ({label_a})",
            "Sales_A": f"Sales ({label_a})",
            "Units_B": f"Units ({label_b})",
            "Sales_B": f"Sales ({label_b})",
        })

        st.caption("ActiveWeeks = number of distinct sales weeks present for the SKU in the selected period.")
        st.dataframe(
            disp.style.format({
                f"Units ({label_a})": fmt_int, f"Units ({label_b})": fmt_int, "Units_Diff": fmt_int,
                f"Sales ({label_a})": fmt_currency, f"Sales ({label_b})": fmt_currency, "Sales_Diff": fmt_currency,
                "ActiveWeeks": fmt_int
            }).applymap(lambda v: f"color:{_color(v)};", subset=["Units_Diff","Sales_Diff"]),
            use_container_width=True,
            hide_index=True,
            height=_table_height(disp, max_px=1100)
        )

# -------------------------
# Lost Sales Detector (Lost + Gained + Net)
# -------------------------
with tabs[7]:
    st.subheader("Lost Sales Detector")

    if df.empty:
        st.info("No data.")
    else:
        years = sorted(df["Year"].dropna().unique().tolist())
        c1, c2 = st.columns(2)
        with c1:
            yA = st.selectbox("Base year", options=years, index=max(0, len(years)-2))
        with c2:
            yB = st.selectbox("Compare year", options=years, index=len(years)-1)

        a = df[df["Year"] == int(yA)].copy()
        b = df[df["Year"] == int(yB)].copy()

        ga = a.groupby(["SKU","Vendor"], as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
        gb = b.groupby(["SKU","Vendor"], as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
        out = ga.merge(gb, on=["SKU","Vendor"], how="outer").fillna(0.0)
        out["Δ Units"] = out["Units_B"] - out["Units_A"]
        out["Δ Sales"] = out["Sales_B"] - out["Sales_A"]

        lost = out[(out["Sales_A"] > 0) & (out["Sales_B"] == 0)].copy()
        gained = out[(out["Sales_A"] == 0) & (out["Sales_B"] > 0)].copy()

        lost_tot_units = float(lost["Δ Units"].sum()) if not lost.empty else 0.0
        lost_tot_sales = float(lost["Δ Sales"].sum()) if not lost.empty else 0.0
        gained_tot_units = float(gained["Δ Units"].sum()) if not gained.empty else 0.0
        gained_tot_sales = float(gained["Δ Sales"].sum()) if not gained.empty else 0.0

        net_units = gained_tot_units + lost_tot_units
        net_sales = gained_tot_sales + lost_tot_sales

        st.markdown(
            f"**Net Units (Gained + Lost):** <span style='color:{_color(net_units)}; font-weight:700;'>{fmt_int(net_units)}</span> &nbsp;&nbsp; "
            f"**Net Sales (Gained + Lost):** <span style='color:{_color(net_sales)}; font-weight:700;'>{fmt_currency_signed(net_sales)}</span>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Lost SKUs (had sales in base, zero in compare)")
            lost = lost.sort_values("Sales_A", ascending=False).head(500)
            st.dataframe(lost.style.format({"Units_A":fmt_int,"Sales_A":fmt_currency,"Units_B":fmt_int,"Sales_B":fmt_currency,"Δ Units":fmt_int,"Δ Sales":fmt_currency}),
                         use_container_width=True, hide_index=True, height=_table_height(lost, max_px=900))
        with c2:
            st.markdown("### Gained SKUs (zero in base, sales in compare)")
            gained = gained.sort_values("Sales_B", ascending=False).head(500)
            st.dataframe(gained.style.format({"Units_A":fmt_int,"Sales_A":fmt_currency,"Units_B":fmt_int,"Sales_B":fmt_currency,"Δ Units":fmt_int,"Δ Sales":fmt_currency}),
                         use_container_width=True, hide_index=True, height=_table_height(gained, max_px=900))

# -------------------------
# Year Summary (clean, all years + accordion)
# -------------------------
with tabs[8]:
    st.subheader("Year Summary")

    if df.empty:
        st.info("No data.")
    else:
        d = df.copy()
        years = sorted(d["Year"].dropna().unique().tolist())
        current_year = int(max(years))
        prior_year = int(years[-2]) if len(years) >= 2 else None

        basis = st.radio("Basis", ["Sales","Units"], horizontal=True, key="ys_basis")
        value_col = "Sales" if basis == "Sales" else "Units"

        cur = d[d["Year"] == current_year].copy()
        prv = d[d["Year"] == prior_year].copy() if prior_year is not None else d.iloc[0:0].copy()

        def _sum(df_, col):
            return float(df_[col].sum()) if df_ is not None and not df_.empty else 0.0

        st.markdown("### KPIs (latest two years)")
        uC, sC = _sum(cur,"Units"), _sum(cur,"Sales")
        uP, sP = (_sum(prv,"Units") if prior_year else 0.0), (_sum(prv,"Sales") if prior_year else 0.0)
        uD, sD = uC-uP, sC-sP
        uPct = (uD/uP) if uP else np.nan
        sPct = (sD/sP) if sP else np.nan

        yr_tot = d.groupby("Year", as_index=False).agg(Units=("Units","sum"), Sales=("Sales","sum")).sort_values("Year")
        avg_units = float(yr_tot["Units"].mean()) if not yr_tot.empty else 0.0
        avg_sales = float(yr_tot["Sales"].mean()) if not yr_tot.empty else 0.0

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric(f"Units ({current_year})", fmt_int(uC))
        k2.metric(f"Units ({prior_year})" if prior_year else "Units (prior)", fmt_int(uP) if prior_year else "—",
                  delta=f"{fmt_int(uD)} ({uPct*100:.1f}%)" if prior_year and pd.notna(uPct) else (fmt_int(uD) if prior_year else None))
        k3.metric(f"Sales ({current_year})", fmt_currency(sC))
        k4.metric(f"Sales ({prior_year})" if prior_year else "Sales (prior)", fmt_currency(sP) if prior_year else "—",
                  delta=f"{fmt_currency_signed(sD)} ({sPct*100:.1f}%)" if prior_year and pd.notna(sPct) else (fmt_currency_signed(sD) if prior_year else None))
        k5.markdown(f"**Δ Units vs all-years avg**<br><span style='color:{_color(uC-avg_units)}; font-weight:700;'>{fmt_int(uC-avg_units)}</span>", unsafe_allow_html=True)
        k6.markdown(f"**Δ Sales vs all-years avg**<br><span style='color:{_color(sC-avg_sales)}; font-weight:700;'>{fmt_currency_signed(sC-avg_sales)}</span>", unsafe_allow_html=True)

        if prior_year is not None:
            st.markdown("### YoY driver breakdown (latest two years)")
            a = prv; b = cur
            sku_a = a.groupby("SKU", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
            sku_b = b.groupby("SKU", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
            sku = sku_a.merge(sku_b, on="SKU", how="outer").fillna(0.0)
            sku["Δ Units"] = sku["Units_B"] - sku["Units_A"]
            sku["Δ Sales"] = sku["Sales_B"] - sku["Sales_A"]
            mover_col = "Δ Sales" if value_col == "Sales" else "Δ Units"
            up = sku.sort_values(mover_col, ascending=False).head(10)
            dn = sku.sort_values(mover_col, ascending=True).head(10)
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Top 10 Increases**")
                st.dataframe(up.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Δ Units":fmt_int,"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Δ Sales":fmt_currency}).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Units","Δ Sales"]),
                             use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Top 10 Decreases**")
                st.dataframe(dn.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Δ Units":fmt_int,"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Δ Sales":fmt_currency}).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Units","Δ Sales"]),
                             use_container_width=True, hide_index=True)

        st.markdown("### Concentration risk (all years)")

        def _top_share(df_year, group_col, topn):
            g = df_year.groupby(group_col, as_index=False).agg(val=(value_col,"sum"))
            total = float(g["val"].sum())
            if total <= 0:
                return 0.0
            return float(g.sort_values("val", ascending=False).head(topn)["val"].sum()) / total

        conc_rows = []
        for y in years:
            dy = d[d["Year"] == int(y)].copy()
            conc_rows.append({
                "Year": int(y),
                "Top 1 Retailer %": _top_share(dy, "Retailer", 1),
                "Top 3 Retailers %": _top_share(dy, "Retailer", 3),
                "Top 5 Retailers %": _top_share(dy, "Retailer", 5),
                "Top 1 Vendor %": _top_share(dy, "Vendor", 1),
                "Top 3 Vendors %": _top_share(dy, "Vendor", 3),
                "Top 5 Vendors %": _top_share(dy, "Vendor", 5),
            })
        conc = pd.DataFrame(conc_rows).sort_values("Year")
        st.dataframe(conc.style.format({c:(lambda v: f"{v*100:.1f}%") for c in conc.columns if c!="Year"}),
                     use_container_width=True, hide_index=True)

        st.caption("Click a year to expand the Top 1 / Top 3 / Top 5 breakdown. Click again to collapse.")

        def _top_list(df_year, group_col, topn):
            g = df_year.groupby(group_col, as_index=False).agg(val=(value_col,"sum"))
            total = float(g["val"].sum())
            if total <= 0:
                return pd.DataFrame(columns=[group_col, value_col, "Share"])
            g = g.sort_values("val", ascending=False).head(topn).copy()
            g["Share"] = g["val"]/total
            g = g.rename(columns={"val": value_col})
            return g[[group_col, value_col, "Share"]]

        for y in years:
            dy = d[d["Year"] == int(y)].copy()
            with st.expander(f"{int(y)} — Top 1 / Top 3 / Top 5 breakdown", expanded=False):
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("**Retailers**")
                    for n in [1,3,5]:
                        t = _top_list(dy, "Retailer", n)
                        st.markdown(f"Top {n} Retailer{'s' if n>1 else ''}")
                        st.dataframe(t.style.format({value_col:(fmt_currency if value_col=="Sales" else fmt_int), "Share": lambda v: f"{v*100:.1f}%"}),
                                     use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Vendors**")
                    for n in [1,3,5]:
                        t = _top_list(dy, "Vendor", n)
                        st.markdown(f"Top {n} Vendor{'s' if n>1 else ''}")
                        st.dataframe(t.style.format({value_col:(fmt_currency if value_col=="Sales" else fmt_int), "Share": lambda v: f"{v*100:.1f}%"}),
                                     use_container_width=True, hide_index=True)

        st.markdown("### Retailer summary (pick years here)")
        r1,r2 = st.columns(2)
        with r1:
            r_base = st.selectbox("Retailer Base Year", options=years, index=max(0, len(years)-2))
        with r2:
            r_comp = st.selectbox("Retailer Comparison Year", options=years, index=len(years)-1)

        ra = d[d["Year"] == int(r_base)].groupby("Retailer", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
        rb = d[d["Year"] == int(r_comp)].groupby("Retailer", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
        r = ra.merge(rb, on="Retailer", how="outer").fillna(0.0)
        r["Units_Diff"] = r["Units_B"] - r["Units_A"]
        r["Sales_Diff"] = r["Sales_B"] - r["Sales_A"]
        r["Units_%"] = r["Units_Diff"] / r["Units_A"].replace(0, np.nan)
        r["Sales_%"] = r["Sales_Diff"] / r["Sales_A"].replace(0, np.nan)
        r = r.sort_values("Sales_Diff", ascending=False)
        r_disp = r[["Retailer","Units_A","Units_B","Units_Diff","Units_%","Sales_A","Sales_B","Sales_Diff","Sales_%"]]
        st.dataframe(r_disp.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Units_Diff":fmt_int,"Units_%":lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                                          "Sales_A":fmt_currency,"Sales_B":fmt_currency,"Sales_Diff":fmt_currency,"Sales_%":lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"}).applymap(lambda v: f"color:{_color(v)};", subset=["Units_Diff","Sales_Diff"]),
                     use_container_width=True, hide_index=True, height=_table_height(r_disp, max_px=1100))

        st.markdown("### Vendor summary (pick years here)")
        v1,v2 = st.columns(2)
        with v1:
            v_base = st.selectbox("Vendor Base Year", options=years, index=max(0, len(years)-2))
        with v2:
            v_comp = st.selectbox("Vendor Comparison Year", options=years, index=len(years)-1)

        va = d[d["Year"] == int(v_base)].groupby("Vendor", as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
        vb = d[d["Year"] == int(v_comp)].groupby("Vendor", as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
        v = va.merge(vb, on="Vendor", how="outer").fillna(0.0)
        v["Units_Diff"] = v["Units_B"] - v["Units_A"]
        v["Sales_Diff"] = v["Sales_B"] - v["Sales_A"]
        v["Units_%"] = v["Units_Diff"] / v["Units_A"].replace(0, np.nan)
        v["Sales_%"] = v["Sales_Diff"] / v["Sales_A"].replace(0, np.nan)
        v = v.sort_values("Sales_Diff", ascending=False)
        v_disp = v[["Vendor","Units_A","Units_B","Units_Diff","Units_%","Sales_A","Sales_B","Sales_Diff","Sales_%"]]
        st.dataframe(v_disp.style.format({"Units_A":fmt_int,"Units_B":fmt_int,"Units_Diff":fmt_int,"Units_%":lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—",
                                          "Sales_A":fmt_currency,"Sales_B":fmt_currency,"Sales_Diff":fmt_currency,"Sales_%":lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"}).applymap(lambda v: f"color:{_color(v)};", subset=["Units_Diff","Sales_Diff"]),
                     use_container_width=True, hide_index=True, height=_table_height(v_disp, max_px=1100))

# -------------------------
# Alerts (month selection)
# -------------------------
with tabs[9]:
    st.subheader("Alerts")

    if df.empty:
        st.info("No data.")
    else:
        months = month_year_options(df)
        if not months:
            st.info("No months available.")
        else:
            c1,c2 = st.columns(2)
            with c1:
                a = st.selectbox("Month A", options=months, index=max(0, len(months)-2))
            with c2:
                b = st.selectbox("Month B", options=months, index=len(months)-1)
            da = filter_month_year(df, a)
            db = filter_month_year(df, b)
            ga = da.groupby(["SKU","Vendor"], as_index=False).agg(Units_A=("Units","sum"), Sales_A=("Sales","sum"))
            gb = db.groupby(["SKU","Vendor"], as_index=False).agg(Units_B=("Units","sum"), Sales_B=("Sales","sum"))
            out = ga.merge(gb, on=["SKU","Vendor"], how="outer").fillna(0.0)
            out["Δ Sales"] = out["Sales_B"] - out["Sales_A"]
            out["Δ Units"] = out["Units_B"] - out["Units_A"]
            out["% Sales"] = out["Δ Sales"] / out["Sales_A"].replace(0, np.nan)
            out["% Units"] = out["Δ Units"] / out["Units_A"].replace(0, np.nan)

            st.markdown("### Biggest changes (by Sales)")
            top_up = out.sort_values("Δ Sales", ascending=False).head(50)
            top_dn = out.sort_values("Δ Sales", ascending=True).head(50)
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Increases**")
                st.dataframe(top_up.style.format({"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Δ Sales":fmt_currency,"Units_A":fmt_int,"Units_B":fmt_int,"Δ Units":fmt_int,
                                                  "% Sales":lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"}).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Sales","Δ Units"]),
                             use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Decreases**")
                st.dataframe(top_dn.style.format({"Sales_A":fmt_currency,"Sales_B":fmt_currency,"Δ Sales":fmt_currency,"Units_A":fmt_int,"Units_B":fmt_int,"Δ Units":fmt_int,
                                                  "% Sales":lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—"}).applymap(lambda v: f"color:{_color(v)};", subset=["Δ Sales","Δ Units"]),
                             use_container_width=True, hide_index=True)

# -------------------------
# Data Management
# -------------------------
with tabs[10]:
    st.subheader("Data Management")

    st.markdown("### Files bundled with the app")
    st.write(f"Sales store: `{DEFAULT_SALES_STORE}` ({'found' if DEFAULT_SALES_STORE.exists() else 'missing'})")
    st.write(f"Vendor map: `{DEFAULT_VENDOR_MAP}` ({'found' if DEFAULT_VENDOR_MAP.exists() else 'missing'})")

    st.markdown("### Upload a new sales_store.csv to replace data")
    up = st.file_uploader("Upload sales_store.csv", type=["csv"])
    if up is not None:
        try:
            new_df = pd.read_csv(up)
            DATA_DIR.mkdir(exist_ok=True, parents=True)
            new_df.to_csv(DEFAULT_SALES_STORE, index=False)
            st.success("Saved to data/sales_store.csv. Refresh the app.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
