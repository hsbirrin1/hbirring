import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, UTC
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats

# ---- Sector Tagging (used for valuation logic) ----
SECTOR = {
    "COST": "Retail",
    "XOM": "Energy",
    "UNH": "Healthcare",
    "GOOGL": "Tech",
    "BAC": "Banks"
}

TICKERS = {
    "COST": "0000909832",   # Costco Wholesale Corp
    "XOM":  "0000034088",   # Exxon Mobil Corp
    "UNH":  "0000731766",   # UnitedHealth Group Inc
    "GOOGL":"0001652044",   # Alphabet Inc (Class A)
    "BAC":  "0000070858",   # Bank of America Corp
}

TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "Revenues", "Revenue",
        "TotalRevenueNetOfInterestExpense",
        "InterestAndNoninterestIncome",
    ],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity"
    ],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PurchaseOfPropertyAndEquipment",
        "CapitalExpenditures"
    ],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "ebit": ["OperatingIncomeLoss"],
    "pretax": ["IncomeBeforeIncomeTaxes"],
    "tax_exp": ["IncomeTaxExpenseBenefit"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "debt_st": ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent", "CommercialPaper"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "div": ["CommonStockDividendsPerShareDeclared"]
}

FORMS_10K = {"10-K"}

OUTDIR = Path(__file__).resolve().parent
print(f"\n[INFO] Outputs will be saved to: {OUTDIR}\n")

UA = {"User-Agent": "your-email@domain.com (for academic use)"}

# --- SEC Data Fetch Functions ---
def get_company_concept(cik: str, tag: str, unit: str = "USD"):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{str(cik).zfill(10)}/us-gaap/{tag}.json"
    r = requests.get(url, headers=UA, timeout=30)
    if r.status_code != 200:
        return []
    units = r.json().get("units", {})
    return units.get(unit, [])

def get_first_available_series(cik: str, tag_list, unit="USD") -> pd.Series:
    for tag in tag_list:
        rows = get_company_concept(cik, tag, unit=unit)
        if rows:
            rows = [x for x in rows if x.get("form") in FORMS_10K and x.get("end")]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")
            return pd.Series(pd.to_numeric(df["val"], errors="coerce").values,
                             index=df["end"].values)
    return pd.Series(dtype=float)

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    out = a.divide(b)
    return out.where(b != 0)

def compute_yearly_metrics(cik: str) -> pd.DataFrame:
    s_revenue = get_first_available_series(cik, TAGS["revenue"])
    s_netinc  = get_first_available_series(cik, TAGS["net_income"])
    s_assets  = get_first_available_series(cik, TAGS["assets"])
    s_liab    = get_first_available_series(cik, TAGS["liabilities"])
    s_equity  = get_first_available_series(cik, TAGS["equity"])
    s_cash    = get_first_available_series(cik, TAGS["cash"])
    s_ocf     = get_first_available_series(cik, TAGS["ocf"])
    s_capex   = get_first_available_series(cik, TAGS["capex"])

    s_eps = get_first_available_series(cik, TAGS["eps"], unit="USD/shares")
    if s_eps.empty:
        s_eps = get_first_available_series(cik, TAGS["eps"], unit="USD/share")
    if s_eps.empty:
        s_eps = get_first_available_series(cik, TAGS["eps"])

    s_ebit    = get_first_available_series(cik, TAGS["ebit"])
    s_pretax  = get_first_available_series(cik, TAGS["pretax"])
    s_taxexp  = get_first_available_series(cik, TAGS["tax_exp"])

    s_debt_lt = get_first_available_series(cik, TAGS["debt_lt"])
    s_debt_st = get_first_available_series(cik, TAGS["debt_st"])
    s_da      = get_first_available_series(cik, TAGS["da"])  # optional

    idx = sorted(set().union(
        s_revenue.index, s_netinc.index, s_assets.index, s_liab.index,
        s_equity.index, s_cash.index, s_ocf.index, s_capex.index,
        s_eps.index, s_ebit.index, s_pretax.index, s_taxexp.index,
        s_debt_lt.index, s_debt_st.index, s_da.index
    ))
    if not idx:
        return pd.DataFrame()

    df = pd.DataFrame(index=idx)
    df["Revenue"]    = s_revenue.reindex(idx)
    df["NetIncome"]  = s_netinc.reindex(idx)
    df["Assets"]     = s_assets.reindex(idx)
    df["Liabilities"]= s_liab.reindex(idx)
    df["Equity"]     = s_equity.reindex(idx)
    df["Cash"]       = s_cash.reindex(idx)
    df["OCF"]        = s_ocf.reindex(idx)
    df["CapEx"]      = s_capex.reindex(idx)
    df["EPS"]        = s_eps.reindex(idx)
    df["EBIT"]       = s_ebit.reindex(idx)
    df["PreTax"]     = s_pretax.reindex(idx)
    df["TaxExp"]     = s_taxexp.reindex(idx)
    df["DA"]         = s_da.reindex(idx)

    total_debt = pd.Series(0.0, index=df.index)
    if not s_debt_lt.empty:
        total_debt = total_debt.add(s_debt_lt.reindex(df.index).fillna(0.0), fill_value=0.0)
    if not s_debt_st.empty:
        total_debt = total_debt.add(s_debt_st.reindex(df.index).fillna(0.0), fill_value=0.0)
    if (total_debt.abs() < 1e-9).all() and "Liabilities" in df:
        total_debt = df["Liabilities"]
    df["TotalDebt"] = total_debt

    df["NetMargin"] = safe_div(df["NetIncome"], df["Revenue"])
    df["ROE"] = safe_div(df["NetIncome"], df["Equity"])
    df["ROA"] = safe_div(df["NetIncome"], df["Assets"])
    df["DE"]  = safe_div(df["TotalDebt"], df["Equity"])

    tax_rate = safe_div(df["TaxExp"], df["PreTax"]).clip(lower=0.0, upper=0.50).fillna(0.21)
    nopat = df["EBIT"].fillna(df["PreTax"]) * (1.0 - tax_rate)
    invested_capital = (df["TotalDebt"].fillna(0.0) + df["Equity"].fillna(0.0)) - df["Cash"].fillna(0.0)
    invested_capital = invested_capital.replace(0.0, np.nan)
    df["ROIC"] = safe_div(nopat, invested_capital)

    df["FCF"] = df["OCF"] - df["CapEx"]
    df["FCFMargin"] = safe_div(df["FCF"], df["Revenue"])

    df["EBITDA"] = df["EBIT"].fillna(0) + df["DA"].fillna(0)
    df.loc[df["EBITDA"] == 0, "EBITDA"] = df.loc[df["EBITDA"] == 0, "EBIT"]

    return df

def eps_cagr_from_sec(cik: str, years: int = 5) -> float:
    s = get_first_available_series(cik, TAGS["eps"], unit="USD/shares")
    if s.empty:
        s = get_first_available_series(cik, TAGS["eps"], unit="USD/share")
    if s.empty:
        s = get_first_available_series(cik, TAGS["eps"])
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    end = s.iloc[-1]
    start_idx = max(0, len(s) - (years + 1))
    start = s.iloc[start_idx]
    n = (len(s) - 1 - start_idx)
    if start <= 0 or end <= 0 or n <= 0:
        if len(s) >= 2 and s.iloc[-2] != 0:
            return (end / s.iloc[-2]) - 1.0
        return np.nan
    return (end / start) ** (1.0 / n) - 1.0

def yahoo_prices(ticker: str, interval="1d", rng="120mo") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": rng}
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["chart"]["result"][0]
    ts = data["timestamp"]
    quote = data["indicators"]["quote"]
    df = pd.DataFrame({
        "Date": [datetime.fromtimestamp(t, UTC) for t in ts],
        "Close": quote["close"],
        "Volume": quote.get("volume"),
    }).dropna()
    return df.set_index("Date")

# ---- Ratio Computation and Output ----
ratio_records = {}
for tkr, cik in TICKERS.items():
    print(f"Pulling SEC facts for {tkr}...")
    df = compute_yearly_metrics(cik).sort_index()
    if df.empty:
        print(f"  Warning: not enough data for {tkr}")
        continue
    core = ["ROE","ROA","DE","NetMargin","ROIC","FCFMargin"]
    usable = df[core].notna().any(axis=1)
    if not usable.any():
        print(f"  Warning: core ratios missing for {tkr}")
        continue
    last_dt = df.index[usable][-1]
    rec = {
        "AsOf": pd.to_datetime(last_dt).date(),
        "ROE": float(df.at[last_dt, "ROE"]) if pd.notna(df.at[last_dt, "ROE"]) else np.nan,
        "ROA": float(df.at[last_dt, "ROA"]) if pd.notna(df.at[last_dt, "ROA"]) else np.nan,
        "DE":  float(df.at[last_dt, "DE"]) if pd.notna(df.at[last_dt, "DE"]) else np.nan,
        "NetMargin": float(df.at[last_dt, "NetMargin"]) if pd.notna(df.at[last_dt, "NetMargin"]) else np.nan,
        "ROIC": float(df.at[last_dt, "ROIC"]) if pd.notna(df.at[last_dt, "ROIC"]) else np.nan,
        "FCFMargin": float(df.at[last_dt, "FCFMargin"]) if pd.notna(df.at[last_dt, "FCFMargin"]) else np.nan,
        "EPS_Growth_5y": float(eps_cagr_from_sec(cik)),
    }
    ratio_records[tkr] = rec

latest_df = pd.DataFrame(ratio_records).T
raw_csv = OUTDIR / "latest_ratios_raw.csv"
latest_df.reset_index().rename(columns={"index":"Ticker"}).to_csv(raw_csv, index=False)
print(f"[SAVED] Raw ratios CSV → {raw_csv}")

POS = ["ROE","ROA","NetMargin","ROIC","FCFMargin","EPS_Growth_5y"]

scaled_df = latest_df.copy()
de_series = latest_df.get("DE").copy()
if de_series is not None:
    de_series = de_series.mask(de_series == 0)
scaled_df["DE_inv"] = np.where(de_series.notna(), 1.0 / (1.0 + de_series.clip(lower=0)), np.nan)
scaled_cols = []
for col in POS + ["DE_inv"]:
    s = scaled_df[col]
    col_s = col + "_S"
    if s.notna().sum() >= 2:
        vmin, vmax = s.min(skipna=True), s.max(skipna=True)
        denom = (vmax - vmin) if (vmax - vmin) != 0 else np.nan
        scaled_df[col_s] = (s - vmin) / denom
    elif s.notna().sum() == 1:
        scaled_df[col_s] = np.where(s.notna(), 1.0, np.nan)
    else:
        scaled_df[col_s] = np.nan
    scaled_cols.append(col_s)
scaled_df["HealthScore"] = scaled_df[scaled_cols].mean(axis=1, skipna=True)
def bucket(x):
    if pd.isna(x):
        return "NA"
    if x >= 2/3:
        return "High"
    if x >= 1/3:
        return "Medium"
    return "Low"
scaled_df["ScoreBucket"] = scaled_df["HealthScore"].apply(bucket)
scaled_csv = OUTDIR / "scaled_metrics_healthscore.csv"
scaled_df.reset_index().rename(columns={"index":"Ticker"}).to_csv(scaled_csv, index=False)
print(f"[SAVED] Scaled metrics CSV → {scaled_csv}")

# --- Apriori Rule Mining ---
flags = pd.DataFrame(index=scaled_df.index)
flags["High_ROE"]     = (scaled_df["ROE_S"] >= 2/3).fillna(False).astype(bool)
flags["High_ROA"]     = (scaled_df["ROA_S"] >= 2/3).fillna(False).astype(bool)
flags["High_Margin"]  = (scaled_df["NetMargin_S"] >= 2/3).fillna(False).astype(bool)
flags["High_ROIC"]    = (scaled_df["ROIC_S"] >= 2/3).fillna(False).astype(bool)
flags["Low_DE"]       = (scaled_df["DE_inv_S"] >= 2/3).fillna(False).astype(bool)
flags["Positive_EPS_Growth"] = (scaled_df["EPS_Growth_5y_S"] > 0.5).fillna(False).astype(bool)
flags["Positive_FCF"] = (scaled_df["FCFMargin_S"] > 0.5).fillna(False).astype(bool)
for b in ["High","Medium","Low"]:
    flags[f"Score_{b}"] = (scaled_df["ScoreBucket"] == b).fillna(False).astype(bool)
flags_csv = OUTDIR / "apriori_flags.csv"
flags.reset_index().rename(columns={"index":"Ticker"}).to_csv(flags_csv, index=False)
print(f"[SAVED] Apriori flags CSV → {flags_csv}")

itemsets = apriori(flags, min_support=0.4, use_colnames=True)
rules = association_rules(itemsets, metric="confidence", min_threshold=0.7)
if not rules.empty:
    rules = rules.sort_values(["confidence","lift"], ascending=False)
    print("\nAssociation Rules:\n", rules[["antecedents","consequents","support","confidence","lift"]].head(10))
    rules_csv = OUTDIR / "apriori_rules.csv"
    rules.to_csv(rules_csv, index=False)
    print(f"[SAVED] Apriori rules CSV → {rules_csv}")
else:
    print("No strong association rules found at current thresholds.")

# --- Portfolio $100k Score Allocation ---
TOTAL_INVEST = 100000
scores = scaled_df["HealthScore"].fillna(0)
N = len(scores)
if scores.sum() == 0 or N == 0:
    weights = pd.Series(1.0 / max(N, 1), index=scores.index)
else:
    equal_w = pd.Series(1.0 / N, index=scores.index)
    score_w = scores / scores.sum()
    alpha = 0.5
    weights = alpha * equal_w + (1 - alpha) * score_w
    weights = weights.clip(lower=0.10, upper=0.35)
    weights = weights / weights.sum()
alloc_series = (weights * TOTAL_INVEST).round()
residual = TOTAL_INVEST - alloc_series.sum()
if residual != 0:
    top_name = weights.idxmax()
    alloc_series[top_name] = alloc_series[top_name] + residual
alloc_df = alloc_series.to_frame(name="USD")
alloc_df.loc["Total"] = alloc_df["USD"].sum()
alloc_plot = alloc_df.drop(index=["Total"]) if "Total" in alloc_df.index else alloc_df.copy()
plt.figure(figsize=(7,7))
plt.pie(alloc_plot["USD"], labels=alloc_plot.index, autopct='%1.1f%%', startangle=90)
plt.title("Portfolio Allocation — $100k (Score-Tilted)")
plt.tight_layout()
pie_path = OUTDIR / "allocation_pie.png"
plt.savefig(pie_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"[SAVED] Allocation pie chart → {pie_path}")
alloc_out = alloc_df.drop(index=["Total"]) if "Total" in alloc_df.index else alloc_df.copy()
alloc_csv = OUTDIR / "portfolio_allocation.csv"
alloc_out.reset_index().rename(columns={"index":"Ticker"}).to_csv(alloc_csv, index=False)
print(f"[SAVED] Portfolio allocation CSV → {alloc_csv}")

# --- COVID Event Study (GOOGL) ---
def event_ttests_google_covid_multi(event_date="2020-03-11"):
    try:
        g = yahoo_prices("GOOGL")
        m = yahoo_prices("^GSPC")
    except Exception as e:
        print(f"Yahoo chart API error: {e}. Skipping event study.")
        return
    data = pd.DataFrame({
        "GOOGL_Close": g["Close"],
        "SPX_Close": m["Close"],
    }).dropna()
    horizons = [5, 20, 30]
    for w in horizons:
        data[f"Return_{w}d"] = (data["GOOGL_Close"].shift(-w) / data["GOOGL_Close"]) - 1.0
        data[f"Market_return_{w}d"] = (data["SPX_Close"].shift(-w) / data["SPX_Close"]) - 1.0
    event_dt = pd.to_datetime(event_date, utc=True)
    if event_dt < data.index.min() or event_dt > data.index.max():
        print("Event date outside price history; skipping event study.")
        return
    d0 = data.index[data.index.searchsorted(event_dt)]
    for w in horizons:
        dfw = data.dropna(subset=[f"Return_{w}d", f"Market_return_{w}d"])
        before = dfw.loc[dfw.index < event_dt, f"Return_{w}d"].tail(w)
        after  = dfw.loc[dfw.index > event_dt, f"Return_{w}d"].head(w)
        if len(before) < 2 or len(after) < 2:
            print(f"\nT-Test Results ({w}-day window): insufficient data.")
            continue
        t_stat, p_val = stats.ttest_ind(before, after, equal_var=False, nan_policy='omit')
        print(f"\nT-Test Results ({w}-day window, COVID event):")
        print(f"Event day used: {event_dt.date()}")
        print(f"Before mean return: {before.mean():.6f}")
        print(f"After mean return:  {after.mean():.6f}")
        print(f"T-statistic:        {t_stat:.4f}")
        print(f"P-value:            {p_val:.4f}")
        if p_val < 0.05:
            print("Statistically significant difference.")
        else:
            print("No statistically significant difference.")
    try:
        plt.figure(figsize=(11,5))
        plt.plot(data.index, data["GOOGL_Close"], label="GOOGL Close")
        plt.axvline(d0, linestyle="--", label="COVID event (2020-03-11)")
        plt.title("GOOGL price around WHO COVID declaration")
        plt.xlabel("Date"); plt.ylabel("Price (USD)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        price_path = OUTDIR / "googl_event_price.png"
        plt.savefig(price_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"[SAVED] Event-study price chart → {price_path}")
    except Exception:
        pass
event_ttests_google_covid_multi("2020-03-11")

# --- Valuation: DCF (with sensitivity grid) + Multiples (by sector) + DDM ---
def latest_price(ticker: str) -> float:
    try:
        px = yahoo_prices(ticker, rng="3mo")
        return float(px["Close"].iloc[-1])
    except Exception:
        return np.nan

def estimate_shares_outstanding(df: pd.DataFrame) -> float:
    try:
        ni = df["NetIncome"].dropna().iloc[-1]
        eps = df["EPS"].dropna().iloc[-1]
        if pd.notna(ni) and pd.notna(eps) and eps != 0:
            return float(ni / eps)
    except Exception:
        pass
    return np.nan

def dcf_valuation_from_sec(cik: str, growth_rate=0.05, discount_rate=0.10, years=5, terminal_growth=0.02) -> dict:
    df = compute_yearly_metrics(cik).sort_index()
    if df.empty or "FCF" not in df or df["FCF"].dropna().empty:
        return {"intrinsic_per_share": np.nan, "note": "No FCF data"}
    latest_fcf = float(df["FCF"].dropna().iloc[-1])
    shares = estimate_shares_outstanding(df)
    fcf_proj = [latest_fcf * ((1 + growth_rate) ** t) for t in range(1, years + 1)]
    disc = [(fcf / ((1 + discount_rate) ** t)) for t, fcf in enumerate(fcf_proj, 1)]
    if discount_rate <= terminal_growth:
        terminal_val_disc = np.nan
    else:
        tv = (fcf_proj[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        terminal_val_disc = tv / ((1 + discount_rate) ** years)
    intrinsic_firm_value = (sum(disc) + (terminal_val_disc if pd.notna(terminal_val_disc) else 0.0))
    if pd.notna(shares) and shares > 0:
        intrinsic_per_share = intrinsic_firm_value / shares
    else:
        intrinsic_per_share = np.nan
    return {
        "latest_fcf": latest_fcf,
        "shares_est": shares,
        "intrinsic_firm_value": intrinsic_firm_value,
        "intrinsic_per_share": intrinsic_per_share,
        "years": years,
        "g": growth_rate,
        "r": discount_rate,
        "g_terminal": terminal_growth,
    }

def dcf_sensitivity(cik, growth_rates=[0.03,0.05,0.07], discount_rates=[0.09,0.10,0.11], terminal_growth=0.02, years=5):
    df = compute_yearly_metrics(cik).sort_index()
    if df.empty or "FCF" not in df or df["FCF"].dropna().empty:
        return pd.DataFrame({"note": ["No FCF data"]})
    latest_fcf = float(df["FCF"].dropna().iloc[-1])
    shares = estimate_shares_outstanding(df)
    grid = []
    for g in growth_rates:
        for r in discount_rates:
            fcf_proj = [latest_fcf * ((1 + g) ** t) for t in range(1, years + 1)]
            disc = [(fcf / ((1 + r) ** t)) for t, fcf in enumerate(fcf_proj, 1)]
            if r <= terminal_growth:
                terminal_val_disc = np.nan
            else:
                tv = (fcf_proj[-1] * (1 + terminal_growth)) / (r - terminal_growth)
                terminal_val_disc = tv / ((1 + r) ** years)
            intrinsic_firm_val = (sum(disc) + (terminal_val_disc if pd.notna(terminal_val_disc) else 0.0))
            intrinsic_per_share = (intrinsic_firm_val / shares) if shares and shares > 0 else np.nan
            grid.append({'growth':g,'disc_rate':r,'terminal_g':terminal_growth,'years':years,'value_per_share':intrinsic_per_share})
    return pd.DataFrame(grid)

def sector_multiples_from_sec_and_price(cik, ticker, sector):
    df = compute_yearly_metrics(cik).sort_index()
    if df.empty:
        return {"note": "No SEC data"}
    px = latest_price(ticker)
    eps = df["EPS"].dropna().iloc[-1] if "EPS" in df and not df["EPS"].dropna().empty else np.nan
    equity = df["Equity"].dropna().iloc[-1] if "Equity" in df and not df["Equity"].dropna().empty else np.nan
    revenue = df["Revenue"].dropna().iloc[-1] if "Revenue" in df and not df["Revenue"].dropna().empty else np.nan
    ebitda = df["EBITDA"].dropna().iloc[-1] if "EBITDA" in df and not df["EBITDA"].dropna().empty else np.nan
    shares = estimate_shares_outstanding(df)
    mktcap = shares * px if pd.notna(shares) and pd.notna(px) else np.nan
    debt = df["TotalDebt"].dropna().iloc[-1] if "TotalDebt" in df and not df["TotalDebt"].dropna().empty else 0.0
    cash = df["Cash"].dropna().iloc[-1] if "Cash" in df and not df["Cash"].dropna().empty else 0.0
    ev = (mktcap + debt - cash) if pd.notna(mktcap) else np.nan
    results = {"P_E": np.nan, "P_B": np.nan, "EV/EBITDA": np.nan, "EV/Sales": np.nan}
    if sector == "Tech":
        results["EV/EBITDA"] = (ev / ebitda) if pd.notna(ev) and pd.notna(ebitda) and ebitda != 0 else np.nan
        results["EV/Sales"] = (ev / revenue) if pd.notna(ev) and pd.notna(revenue) and revenue != 0 else np.nan
        results["P_E"] = (px / eps) if pd.notna(px) and pd.notna(eps) and eps != 0 else np.nan
    elif sector == "Banks":
        results["P_E"] = (px / eps) if pd.notna(px) and pd.notna(eps) and eps != 0 else np.nan
        bvps = (equity / shares) if pd.notna(equity) and pd.notna(shares) and shares > 0 else np.nan
        results["P_B"] = (px / bvps) if pd.notna(px) and pd.notna(bvps) and bvps != 0 else np.nan
    elif sector == "Retail":
        results["P_E"] = (px / eps) if pd.notna(px) and pd.notna(eps) and eps != 0 else np.nan
        results["EV/EBITDA"] = (ev / ebitda) if pd.notna(ev) and pd.notna(ebitda) and ebitda != 0 else np.nan
    else:
        results["P_E"] = (px / eps) if pd.notna(px) and pd.notna(eps) and eps != 0 else np.nan
    return results

def ddm_valuation(cik, growth_rate=0.03, discount_rate=0.08):
    df = compute_yearly_metrics(cik).sort_index()
    dividends = get_first_available_series(cik, ["CommonStockDividendsPerShareDeclared"], unit="USD/shares")
    if dividends.empty:
        return {"note": "No dividend data"}
    latest_div = dividends.dropna().iloc[-1]
    if pd.isna(latest_div) or latest_div == 0:
        return {"note": "No recent dividend"}
    if discount_rate <= growth_rate:
        return {"note": "Invalid discount/growth"}
    fair_value = latest_div * (1 + growth_rate) / (discount_rate - growth_rate)
    return {
        "dividend_ps_latest": latest_div,
        "growth_rate": growth_rate,
        "discount_rate": discount_rate,
        "ddm_value_ps": fair_value
    }

def multiples_from_sec_and_price(cik: str, ticker: str) -> dict:
    df = compute_yearly_metrics(cik).sort_index()
    if df.empty:
        return {"note": "No SEC data"}
    px = latest_price(ticker)
    eps = df["EPS"].dropna().iloc[-1] if "EPS" in df and not df["EPS"].dropna().empty else np.nan
    equity = df["Equity"].dropna().iloc[-1] if "Equity" in df and not df["Equity"].dropna().empty else np.nan
    cash = df["Cash"].dropna().iloc[-1] if "Cash" in df and not df["Cash"].dropna().empty else 0.0
    debt = df["TotalDebt"].dropna().iloc[-1] if "TotalDebt" in df and not df["TotalDebt"].dropna().empty else 0.0
    ebitda = df["EBITDA"].dropna().iloc[-1] if "EBITDA" in df and not df["EBITDA"].dropna().empty else np.nan
    shares = estimate_shares_outstanding(df)
    mktcap = shares * px if pd.notna(shares) and pd.notna(px) else np.nan
    pe = (px / eps) if (pd.notna(px) and pd.notna(eps) and eps != 0) else np.nan
    bvps = (equity / shares) if (pd.notna(equity) and pd.notna(shares) and shares > 0) else np.nan
    pb = (px / bvps) if (pd.notna(px) and pd.notna(bvps) and bvps != 0) else np.nan
    ev = (mktcap + debt - cash) if pd.notna(mktcap) else np.nan
    ev_ebitda = (ev / ebitda) if (pd.notna(ev) and pd.notna(ebitda) and ebitda != 0) else np.nan
    return {
        "latest_price": px,
        "shares_est": shares,
        "market_cap": mktcap,
        "P_E": pe,
        "P_B": pb,
        "EV": ev,
        "EBITDA": ebitda,
        "EV_EBITDA": ev_ebitda,
    }

# ---- Main Valuation Loop: DCF, Multiples, DDM ----

val_rows = []
sensitivity_grids = {}
for tkr, cik in TICKERS.items():
    sector = SECTOR.get(tkr, "Other")
    print(f"\n[VALUATION] {tkr} ({sector})")
    # -- DCF and Sensitivity
    dcf = dcf_valuation_from_sec(cik, growth_rate=0.05, discount_rate=0.10, years=5, terminal_growth=0.02)
    sens_df = dcf_sensitivity(cik)
    sensitivity_grids[tkr] = sens_df
    sens_path = OUTDIR / f"dcf_sensitivity_{tkr}.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"  [SAVED] DCF sensitivity grid for {tkr} → {sens_path}")
    # -- Relative Multiples by Sector
    multiples = sector_multiples_from_sec_and_price(cik, tkr, sector)
    # -- DDM (Dividend stocks)
    ddm = ddm_valuation(cik)
    ddm_value = ddm.get("ddm_value_ps", np.nan) if "ddm_value_ps" in ddm else np.nan
    div_ps = ddm.get("dividend_ps_latest", np.nan) if "dividend_ps_latest" in ddm else np.nan
    ddm_note = ddm.get("note", "")
    mul = multiples_from_sec_and_price(cik, tkr)
    price = mul.get("latest_price", np.nan)
    shares = mul.get("shares_est", np.nan)
    mktcap = mul.get("market_cap", np.nan)
    row = {
        "Ticker": tkr,
        "Sector": sector,
        "Price_Latest": price,
        "MarketCap": mktcap,
        "DCF_Intrinsic_Per_Share": dcf.get("intrinsic_per_share", np.nan),
        "DCF_Undervalued?": (dcf.get("intrinsic_per_share",np.nan) > price) if (pd.notna(dcf.get("intrinsic_per_share",np.nan)) and pd.notna(price)) else np.nan,
        "DCF_Assumptions": f"g={dcf.get('g', np.nan):.2%}, r={dcf.get('r', np.nan):.2%}, T={dcf.get('years', np.nan)}, gT={dcf.get('g_terminal', np.nan):.2%}",
        "DDM_Value_Per_Share": ddm_value,
        "Dividend_PS_Latest": div_ps,
        "DDM_Note": ddm_note,
        "P_E": multiples.get("P_E", np.nan),
        "P_B": multiples.get("P_B", np.nan),
        "EV/EBITDA": multiples.get("EV/EBITDA", np.nan),
        "EV/Sales": multiples.get("EV/Sales", np.nan),
    }
    val_rows.append(row)

valuation_df = pd.DataFrame(val_rows)
valuation_csv = OUTDIR / "valuation_summary_full.csv"
valuation_df.to_csv(valuation_csv, index=False)
print(f"[SAVED] Valuation summary CSV (all approaches) → {valuation_csv}")

# --- DCF Sensitivity output: all as one file ---
sensitivity_all = []
for tkr, sens in sensitivity_grids.items():
    df = sens.copy()
    df["Ticker"] = tkr
    sensitivity_all.append(df)
all_sens_df = pd.concat(sensitivity_all)
sens_all_csv = OUTDIR / "dcf_sensitivity_all.csv"
all_sens_df.to_csv(sens_all_csv, index=False)
print(f"[SAVED] Full DCF sensitivity grid for all tickers → {sens_all_csv}")

# --- DCF Sensitivity Heatmaps ---
try:
    import seaborn as sns
    for tkr, sensdf in sensitivity_grids.items():
        if sensdf.isnull().all().all():
            continue
        piv = sensdf.pivot("growth", "disc_rate", "value_per_share")
        plt.figure(figsize=(6,4))
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f"DCF Value Sensitivity ({tkr})\nIntrinsic Value/Share")
        plt.ylabel("FCF Growth Rate")
        plt.xlabel("Discount Rate")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"dcf_sens_heatmap_{tkr}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] DCF sensitivity heatmap for {tkr} → {OUTDIR / f'dcf_sens_heatmap_{tkr}.png'}")
except ImportError:
    print("Seaborn not available: skipping heatmap plots.")

print("\nDone. Key outputs saved in:")
print(" - latest_ratios_raw.csv")
print(" - scaled_metrics_healthscore.csv")
print(" - apriori_flags.csv, apriori_rules.csv (if any)")
print(" - portfolio_allocation.csv, allocation_pie.png")
print(" - googl_event_price.png")
print(" - valuation_summary_full.csv")
print(" - dcf_sensitivity_all.csv, dcf_sensitivity_*.csv, dcf_sens_heatmap_*.png (if seaborn available)")
