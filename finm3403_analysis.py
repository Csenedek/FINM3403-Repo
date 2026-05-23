"""
FINM3403 Group Assignment: China Exposure from an Australian Investor Perspective
=================================================================================
Parts C through G: Data construction, regressions, correlations, risk-return,
and portfolio optimisation.

Run this script after placing your NBS data CSV files in the same folder.
All outputs are saved to an Excel workbook and PNG chart files.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────

START = "2011-03-01"   # one month before April 2011 so returns start April 2011
END   = "2025-12-31"
RF    = 0.0            # risk-free rate = 0 as specified in task sheet

TICKERS_AUS = [
    "FMG.AX", "RIO.AX",   # Iron ore / mining
    "WDS.AX", "STO.AX",   # LNG / energy
    "TWE.AX", "A2M.AX",   # Agriculture / food
    "IEL.AX", "FLT.AX",   # Education / tourism
    "COH.AX", "RMD.AX",   # Healthcare
    "GMG.AX", "QUB.AX",   # Logistics / property
]

INDUSTRIES = {
    "Iron Ore/Mining":       ["FMG.AX", "RIO.AX"],
    "LNG/Energy":            ["WDS.AX", "STO.AX"],
    "Agriculture/Food":      ["TWE.AX", "A2M.AX"],
    "Education/Tourism":     ["IEL.AX", "FLT.AX"],
    "Healthcare":            ["COH.AX", "RMD.AX"],
    "Logistics/Property":    ["GMG.AX", "QUB.AX"],
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("FINM3403 China Exposure Analysis")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. DOWNLOAD PRICE DATA
# ─────────────────────────────────────────────
print("\n[1] Downloading price data from Yahoo Finance...")

all_tickers = ["IOZ.AX", "MCHI", "AUDUSD=X"] + TICKERS_AUS

raw = yf.download(
    all_tickers,
    start=START,
    end=END,
    interval="1mo",
    auto_adjust=True,
    progress=False,
)["Close"]

# Resample to month-end and forward fill at most 1 period for minor gaps
prices = raw.resample("ME").last().ffill(limit=1)

# Trim to April 2011 – December 2025
prices = prices.loc["2011-04-01":"2025-12-31"]

print(f"    Price data shape: {prices.shape}")
print(f"    Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# ─────────────────────────────────────────────
# 2. CONSTRUCT MONTHLY RETURN SERIES IN AUD
# ─────────────────────────────────────────────
print("\n[2] Constructing monthly return series...")

# All ASX tickers are already priced in AUD
returns_raw = prices.pct_change().dropna(how="all")

# IOZ.AX: Australian market
r_aus = returns_raw["IOZ.AX"].rename("R_AUS")

# MCHI: convert USD returns to AUD
# AUD return = (1 + USD return) * (AUD/USD_t / AUD/USD_{t-1}) - 1
# AUDUSD=X gives AUD per USD, so higher = stronger AUD
audusd      = prices["AUDUSD=X"]
audusd_ret  = audusd.pct_change()
mchi_usd    = returns_raw["MCHI"]
r_mchi      = ((1 + mchi_usd) * (1 + audusd_ret) - 1).rename("R_MCHI_AUD")

# Individual ASX firm returns
firm_rets = returns_raw[TICKERS_AUS].copy()

# ─────────────────────────────────────────────
# 3. RAW INDIRECT PORTFOLIO (dynamic reweighting)
# ─────────────────────────────────────────────
print("\n[3] Building raw indirect portfolio with dynamic reweighting...")

# For each month, use only firms with available data.
# Within each industry: equal-weight available firms (1 or 2).
# Across industries: equal-weight available industries (up to 6).

industry_rets = pd.DataFrame(index=firm_rets.index)

for ind_name, tickers in INDUSTRIES.items():
    available = firm_rets[tickers]
    # Equal weight within industry using available firms each month
    ind_ret = available.mean(axis=1)  # mean ignores NaN by default
    industry_rets[ind_name] = ind_ret

# Equal-weight across industries each month
r_indirect = industry_rets.mean(axis=1).rename("R_Indirect")

print(f"    Industry portfolios built: {list(industry_rets.columns)}")

# ─────────────────────────────────────────────
# 4. MARKET-ADJUSTMENT REGRESSION
# ─────────────────────────────────────────────
print("\n[4] Running market-adjustment regression...")

# Align series
df_reg = pd.concat([r_indirect, r_aus], axis=1).dropna()
Y = df_reg["R_Indirect"]
X = sm.add_constant(df_reg["R_AUS"])

mkt_model  = sm.OLS(Y, X).fit(cov_type='HC3')
alpha_mkt  = mkt_model.params["const"]
beta_mkt   = mkt_model.params["R_AUS"]

print(f"    Market-adjustment regression: alpha = {alpha_mkt:.4f}, beta = {beta_mkt:.4f}")
print(f"    R-squared: {mkt_model.rsquared:.4f}")
print(f"    Beta t-stat: {mkt_model.tvalues['R_AUS']:.3f}, p-value: {mkt_model.pvalues['R_AUS']:.4f}")

# Market-adjusted indirect return
r_market_adj = (r_indirect - beta_mkt * r_aus).rename("R_Market_Adj")

# ─────────────────────────────────────────────
# 5. COMBINE ALL PORTFOLIO RETURNS
# ─────────────────────────────────────────────

# Align all four series
portfolios = pd.concat([r_aus, r_mchi, r_indirect, r_market_adj], axis=1).dropna(how="all")
portfolios.columns = ["R_AUS", "R_MCHI_AUD", "R_Indirect", "R_Market_Adj"]

print(f"\n[5] Final portfolio dataset: {portfolios.shape[0]} months")

# ─────────────────────────────────────────────
# 6. LOAD NBS CHINA ACTIVITY DATA
# ─────────────────────────────────────────────
print("\n[6] Loading NBS China activity data...")

# ── INSTRUCTIONS FOR NBS DATA ──────────────────────────────────────────
# Go to: https://data.stats.gov.cn/dg/website/page.html#/pc/national/en/monthData
# Download:
#   1. "Value-added of Industries Above Designated Size (YoY%)" → save as nbs_ip.csv
#   2. "Total Retail Sales of Consumer Goods (YoY%)"           → save as nbs_retail.csv
# Both should have columns: [Date, Value] with Date as YYYY-MM
# Place both files in the same folder as this script.
# ───────────────────────────────────────────────────────────────────────

nbs_loaded = False
try:
    ip_df     = pd.read_csv("nbs_ip.csv",     parse_dates=["Date"], index_col="Date")
    retail_df = pd.read_csv("nbs_retail.csv", parse_dates=["Date"], index_col="Date")

    ip_df.index     = ip_df.index.to_period("M").to_timestamp("M")
    retail_df.index = retail_df.index.to_period("M").to_timestamp("M")

    r_ip     = ip_df["Value"].rename("Delta_IP")     / 100
    r_retail = retail_df["Value"].rename("Delta_Retail") / 100

    china_data = pd.concat([r_ip, r_retail], axis=1)
    china_data = china_data.loc["2011-04-01":"2025-12-31"]
    nbs_loaded = True
    print(f"    NBS data loaded: {china_data.shape[0]} observations")
except FileNotFoundError:
    print("    WARNING: nbs_ip.csv or nbs_retail.csv not found.")
    print("    China validation regressions will be SKIPPED.")
    print("    Download data from NBS portal and rerun.")

# ─────────────────────────────────────────────
# 7. CHINA VALIDATION REGRESSIONS (Part C)
# ─────────────────────────────────────────────
print("\n[7] China validation regressions...")

validation_results = {}

if nbs_loaded:
    reg_series = {
        "R_Indirect":   portfolios["R_Indirect"],
        "R_Market_Adj": portfolios["R_Market_Adj"],
        "R_MCHI_AUD":   portfolios["R_MCHI_AUD"],
        "R_AUS":        portfolios["R_AUS"],
    }

    for name, series in reg_series.items():
        df_v = pd.concat([series, china_data], axis=1).dropna()
        if len(df_v) < 12:
            print(f"    Skipping {name}: insufficient overlap with NBS data")
            continue
        Y_v = df_v[series.name]
        X_v = sm.add_constant(df_v[["Delta_IP", "Delta_Retail"]])
        res = sm.OLS(Y_v, X_v).fit(cov_type='HC3')
        validation_results[name] = res
        print(f"\n    {name}:")
        print(f"      alpha = {res.params['const']:.4f} (t={res.tvalues['const']:.2f})")
        print(f"      beta_IP = {res.params['Delta_IP']:.4f} (t={res.tvalues['Delta_IP']:.2f})")
        print(f"      beta_Retail = {res.params['Delta_Retail']:.4f} (t={res.tvalues['Delta_Retail']:.2f})")
        print(f"      R2 = {res.rsquared:.4f}, N = {int(res.nobs)}")
else:
    print("    Skipped (NBS data not available)")

# ─────────────────────────────────────────────
# 8. ROLLING 36-MONTH CORRELATIONS (Part D)
# ─────────────────────────────────────────────
print("\n[8] Computing 36-month rolling correlations...")

WINDOW = 36

pairs = [
    ("R_AUS",        "R_MCHI_AUD",   "AUS vs Direct China ETF"),
    ("R_AUS",        "R_Indirect",   "AUS vs Raw Indirect"),
    ("R_AUS",        "R_Market_Adj", "AUS vs Market-Adjusted Indirect"),
    ("R_MCHI_AUD",   "R_Indirect",   "Direct ETF vs Raw Indirect"),
    ("R_MCHI_AUD",   "R_Market_Adj", "Direct ETF vs Market-Adjusted Indirect"),
]

roll_corr = pd.DataFrame(index=portfolios.index)
for a, b, label in pairs:
    s = portfolios[[a, b]].dropna()
    rc = s[a].rolling(WINDOW).corr(s[b])
    roll_corr[label] = rc

# Plot rolling correlations
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, (_, _, label) in enumerate(pairs):
    ax = axes[i]
    ax.plot(roll_corr.index, roll_corr[label], color=colors[i], linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(roll_corr[label].mean(), color=colors[i],
               linewidth=1, linestyle=":", label=f"Mean = {roll_corr[label].mean():.2f}")
    # Shade GFC and COVID periods
    ax.axvspan(pd.Timestamp("2011-08-01"), pd.Timestamp("2012-06-30"),
               alpha=0.1, color="red", label="EU Debt Crisis")
    ax.axvspan(pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-28"),
               alpha=0.1, color="orange", label="China Equity Crash")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"),
               alpha=0.1, color="purple", label="COVID-19")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Rolling Correlation")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

axes[5].set_visible(False)
plt.suptitle("36-Month Rolling Correlations Between Portfolio Pairs\n(Shaded regions: EU Debt Crisis, China Equity Crash, COVID-19)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "rolling_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Rolling correlation chart saved.")

# ─────────────────────────────────────────────
# 9. RISK-RETURN STATISTICS (Part E)
# ─────────────────────────────────────────────
print("\n[9] Computing risk-return statistics...")

def risk_return_stats(series, rf=0.0, label=None):
    s = series.dropna()
    mean_m  = s.mean()
    std_m   = s.std()
    mean_a  = mean_m * 12
    std_a   = std_m * np.sqrt(12)
    sharpe  = (mean_a - rf) / std_a if std_a > 0 else np.nan
    return {
        "Portfolio":       label or series.name,
        "Mean (Monthly)":  round(mean_m * 100, 4),
        "Std (Monthly)":   round(std_m * 100, 4),
        "Mean (Annual %)": round(mean_a * 100, 2),
        "Std (Annual %)":  round(std_a * 100, 2),
        "Sharpe Ratio":    round(sharpe, 4),
        "N (months)":      len(s),
    }

stats_list = []
labels = {
    "R_AUS":        "Australian Market (IOZ.AX)",
    "R_MCHI_AUD":   "Direct China ETF (MCHI, AUD)",
    "R_Indirect":   "Raw Indirect Portfolio",
    "R_Market_Adj": "Market-Adjusted Indirect Portfolio",
}
for col, label in labels.items():
    stats_list.append(risk_return_stats(portfolios[col], rf=RF, label=label))

stats_df = pd.DataFrame(stats_list)
print("\n    Risk-Return Summary:")
print(stats_df.to_string(index=False))

# Correlation matrix
corr_matrix = portfolios.rename(columns=labels).corr().round(4)
print("\n    Correlation Matrix:")
print(corr_matrix.to_string())

# ─────────────────────────────────────────────
# 10. PORTFOLIO OPTIMISATION (Part F)
# ─────────────────────────────────────────────
print("\n[10] Running portfolio optimisation (w = 0% to 50%)...")

weights  = np.arange(0, 0.55, 0.05)   # 0%, 5%, ..., 50%

strategies = {
    "Pure Australian Market":               None,
    "AUS + Direct China ETF":               "R_MCHI_AUD",
    "AUS + Raw Indirect":                   "R_Indirect",
    "AUS + Market-Adjusted Indirect":       "R_Market_Adj",
}

opt_results = {}
all_sweep   = {}

for strat_name, overlay_col in strategies.items():
    sweep = []
    for w in weights:
        if overlay_col is None:
            # Pure Australian market — w has no meaning, just report once
            r_port = portfolios["R_AUS"].dropna()
        else:
            combined = pd.concat([portfolios["R_AUS"], portfolios[overlay_col]], axis=1).dropna()
            r_port   = (1 - w) * combined["R_AUS"] + w * combined[overlay_col]

        mean_a = r_port.mean() * 12
        std_a  = r_port.std()  * np.sqrt(12)
        sharpe = (mean_a - RF) / std_a if std_a > 0 else np.nan

        sweep.append({
            "Strategy":       strat_name,
            "Weight (w)":     round(w, 2),
            "Ann. Return (%)": round(mean_a * 100, 2),
            "Ann. Std (%)":   round(std_a  * 100, 2),
            "Sharpe Ratio":   round(sharpe, 4),
        })

        if overlay_col is None:
            break   # Pure AUS is just one result

    sweep_df = pd.DataFrame(sweep)
    all_sweep[strat_name] = sweep_df

    if overlay_col is None:
        best_row = sweep_df.iloc[0]
    else:
        best_row = sweep_df.loc[sweep_df["Sharpe Ratio"].idxmax()]

    opt_results[strat_name] = best_row
    print(f"\n    {strat_name}:")
    print(f"      Optimal w = {best_row['Weight (w)']:.0%}")
    print(f"      Ann. Return = {best_row['Ann. Return (%)']:.2f}%")
    print(f"      Ann. Std    = {best_row['Ann. Std (%)']:.2f}%")
    print(f"      Sharpe      = {best_row['Sharpe Ratio']:.4f}")

# Plot efficient frontier sweeps
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors_strat = {
    "Pure Australian Market":           "#1f77b4",
    "AUS + Direct China ETF":           "#d62728",
    "AUS + Raw Indirect":               "#2ca02c",
    "AUS + Market-Adjusted Indirect":   "#9467bd",
}

for strat_name, sweep_df in all_sweep.items():
    c = colors_strat[strat_name]
    if strat_name == "Pure Australian Market":
        axes[0].scatter(sweep_df["Ann. Std (%)"], sweep_df["Ann. Return (%)"],
                        color=c, s=100, zorder=5, label=strat_name, marker="*")
        axes[1].scatter([0], sweep_df["Sharpe Ratio"],
                        color=c, s=100, zorder=5, label=strat_name, marker="*")
    else:
        axes[0].plot(sweep_df["Ann. Std (%)"], sweep_df["Ann. Return (%)"],
                     color=c, marker="o", markersize=5, label=strat_name)
        axes[1].plot(sweep_df["Weight (w)"] * 100, sweep_df["Sharpe Ratio"],
                     color=c, marker="o", markersize=5, label=strat_name)
        # Mark optimal
        best = sweep_df.loc[sweep_df["Sharpe Ratio"].idxmax()]
        axes[1].scatter(best["Weight (w)"] * 100, best["Sharpe Ratio"],
                        color=c, s=120, zorder=5, marker="*")

axes[0].set_xlabel("Annualised Std Dev (%)")
axes[0].set_ylabel("Annualised Return (%)")
axes[0].set_title("Risk-Return Trade-off by Strategy\n(each point = 5% increment in w)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Allocation to China Strategy (w %)")
axes[1].set_ylabel("Sharpe Ratio")
axes[1].set_title("Sharpe Ratio vs China Allocation\n(★ = optimal weight)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "portfolio_optimisation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n    Portfolio optimisation chart saved.")

# ─────────────────────────────────────────────
# 11. EXPORT TO EXCEL
# ─────────────────────────────────────────────
print("\n[11] Exporting all results to Excel...")

with pd.ExcelWriter(OUTPUT_DIR + "FINM3403_Results.xlsx", engine="openpyxl") as writer:

    # Sheet 1: Raw return series
    out_rets = pd.concat([
        r_aus, r_mchi, r_indirect, r_market_adj,
        firm_rets, industry_rets
    ], axis=1)
    out_rets.index = out_rets.index.strftime("%Y-%m")
    out_rets.to_excel(writer, sheet_name="Return Series")

    # Sheet 2: Market adjustment regression
    mkt_summary = pd.DataFrame({
        "Parameter": ["Alpha", "Beta (IOZ.AX)", "R-squared", "N (months)"],
        "Value": [
            round(alpha_mkt, 6),
            round(beta_mkt, 6),
            round(mkt_model.rsquared, 4),
            int(mkt_model.nobs),
        ],
        "t-statistic": [
            round(mkt_model.tvalues["const"], 3),
            round(mkt_model.tvalues["R_AUS"], 3),
            "", ""
        ],
        "p-value": [
            round(mkt_model.pvalues["const"], 4),
            round(mkt_model.pvalues["R_AUS"], 4),
            "", ""
        ],
    })
    mkt_summary.to_excel(writer, sheet_name="Market Adj Regression", index=False)

    # Sheet 3: China validation regressions
    if validation_results:
        val_rows = []
        for name, res in validation_results.items():
            val_rows.append({
                "Portfolio":        name,
                "Alpha":            round(res.params["const"], 6),
                "Alpha t-stat":     round(res.tvalues["const"], 3),
                "Beta_IP":          round(res.params["Delta_IP"], 6),
                "Beta_IP t-stat":   round(res.tvalues["Delta_IP"], 3),
                "Beta_Retail":      round(res.params["Delta_Retail"], 6),
                "Beta_Retail t-stat": round(res.tvalues["Delta_Retail"], 3),
                "R-squared":        round(res.rsquared, 4),
                "N":                int(res.nobs),
            })
        pd.DataFrame(val_rows).to_excel(writer, sheet_name="China Validation", index=False)
    else:
        pd.DataFrame({"Note": ["NBS data not loaded — rerun after adding nbs_ip.csv and nbs_retail.csv"]}) \
            .to_excel(writer, sheet_name="China Validation", index=False)

    # Sheet 4: Rolling correlations
    roll_out = roll_corr.copy()
    roll_out.index = roll_out.index.strftime("%Y-%m")
    roll_out.to_excel(writer, sheet_name="Rolling Correlations")

    # Sheet 5: Risk-return statistics
    stats_df.to_excel(writer, sheet_name="Risk-Return Stats", index=False)
    # Add correlation matrix below
    startrow = len(stats_df) + 3
    corr_matrix.to_excel(writer, sheet_name="Risk-Return Stats", startrow=startrow)

    # Sheet 6: Portfolio optimisation sweep
    all_sweep_df = pd.concat(list(all_sweep.values()), ignore_index=True)
    all_sweep_df.to_excel(writer, sheet_name="Portfolio Optimisation", index=False)

    # Sheet 7: Optimal weights summary
    opt_df = pd.DataFrame(opt_results).T.reset_index(drop=True)
    opt_df.to_excel(writer, sheet_name="Optimal Weights Summary", index=False)

print("    Excel file saved: FINM3403_Results.xlsx")

# ─────────────────────────────────────────────
# 12. SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("  - FINM3403_Results.xlsx  (all tables)")
print("  - rolling_correlations.png")
print("  - portfolio_optimisation.png")
if not nbs_loaded:
    print("\n  !! ACTION REQUIRED: Download NBS data and rerun for China")
    print("     validation regressions (Part C).")
    print("     URL: https://data.stats.gov.cn/dg/website/page.html#/pc/national/en/monthData")
    print("     Save as: nbs_ip.csv and nbs_retail.csv")
    print("     Columns required: Date (YYYY-MM), Value (YoY % as a number)")
print("\nDone.")
