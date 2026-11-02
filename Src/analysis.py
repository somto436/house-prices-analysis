"""Reproduce the house-price EDA: python Src/analysis.py"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display

cwd = Path.cwd()
candidates = [cwd / "Data/raw/train.csv", cwd.parent / "Data/raw/train.csv", cwd / "train.csv", cwd / "train(2).csv"]
data_path = next((p for p in candidates if p.is_file()), None)
if data_path is None:
    raise FileNotFoundError("Put train.csv or train(2).csv beside the notebook, or use Data/raw/train.csv.")
project_root = data_path.parents[2] if data_path.parent.name == "raw" else cwd
results_dir = project_root / "Results"
results_dir.mkdir(exist_ok=True)

# Preserve literal NA and None codes before making analysis decisions.
raw = pd.read_csv(data_path, keep_default_na=False)
houses = raw.copy()
for column in houses.columns:
    houses[column] = houses[column].mask(houses[column].eq("NA"))
# These measurement columns contain numeric values and NA codes in the CSV.
for column in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
    houses[column] = pd.to_numeric(houses[column], errors="coerce")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
money = FuncFormatter(lambda value, position: f"${value/1000:,.0f}k")

def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(results_dir / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

houses.head()

houses.info()
print(f"\nRows: {houses.shape[0]:,}; columns: {houses.shape[1]}")
print(f"Duplicate rows: {houses.duplicated().sum()}; duplicate IDs: {houses['Id'].duplicated().sum()}")
print(f"Missing SalePrice: {houses['SalePrice'].isna().sum()}")
print(f"Non-positive SalePrice: {(houses['SalePrice'] <= 0).sum()}")
print("Numerically stored columns:", len(houses.select_dtypes(include="number").columns))
print("Other columns:", len(houses.select_dtypes(exclude="number").columns))
display(houses[["SalePrice", "GrLivArea", "LotArea", "TotalBsmtSF", "OverallQual", "YearBuilt"]].describe().round(2))
houses.describe(include="number").to_csv(results_dir / "numeric_summary.csv")

missing = pd.DataFrame({
    "NA codes": raw.eq("NA").sum(),
    "Blank strings": raw.eq("").sum(),
    "Unavailable in analysis": houses.isna().sum()
})
missing["Unavailable (%)"] = 100 * missing["Unavailable in analysis"] / len(houses)
missing = missing.loc[missing["Unavailable in analysis"] > 0].sort_values("Unavailable in analysis", ascending=False)
display(missing.round(2))
missing.to_csv(results_dir / "missing_values.csv")
fig, ax = plt.subplots(figsize=(8, 5))
missing["Unavailable (%)"].sort_values().plot.barh(ax=ax, color="#526778")
ax.set(xlabel="Records with NA codes (%)", title="Unavailable values / feature codes by column", xlim=(0, 105))
save_plot(fig, "01_missing_values")

price = houses["SalePrice"]
log_price = np.log(price)
price_summary = price.describe().to_frame("SalePrice")
display(price_summary.round(2))
print(f"Price skewness: {price.skew():.3f}")
print(f"Log-price skewness: {log_price.skew():.3f}")
price_summary.to_csv(results_dir / "sale_price_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(price, bins=30, color="#526778", edgecolor="white")
axes[0].axvline(price.median(), color="black", linestyle="--", label=f"Median: ${price.median():,.0f}")
axes[0].xaxis.set_major_formatter(money)
axes[0].set(title="SalePrice: original scale", xlabel="Sale price", ylabel="Homes")
axes[0].legend(frameon=False, fontsize=8)
axes[1].hist(log_price, bins=30, color="#8D9DAB", edgecolor="white")
axes[1].set(title="SalePrice: natural-log scale", xlabel="ln(SalePrice)", ylabel="Homes")
save_plot(fig, "02_price_distribution")

distribution_columns = ["GrLivArea", "LotArea", "TotalBsmtSF", "YearBuilt"]
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for column, ax in zip(distribution_columns, axes.flat):
    ax.hist(houses[column].dropna(), bins=30, color="#526778", edgecolor="white")
    ax.set(title=column, xlabel=column, ylabel="Homes")
save_plot(fig, "03_predictor_distributions")
display(houses[distribution_columns].skew().rename("Skewness").to_frame().round(3))

numeric = houses.select_dtypes(include="number")
feature_columns = numeric.columns.drop(["Id", "MSSubClass", "MoSold", "YrSold", "SalePrice"])
pearson = houses[feature_columns].corrwith(price, method="pearson")
spearman = houses[feature_columns].corrwith(price, method="spearman")
pair_counts = houses[feature_columns].notna().mul(price.notna(), axis=0).sum()
correlations = pd.DataFrame({"Pearson r": pearson, "Spearman rho": spearman, "Pairs n": pair_counts})
correlations = correlations.loc[correlations["Pearson r"].abs().sort_values(ascending=False).index]
display(correlations.head(12).round(3))
correlations.to_csv(results_dir / "price_correlations.csv")

heat_columns = ["SalePrice"] + correlations.head(7).index.tolist()
heat = houses[heat_columns].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(heat, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(heat_columns)), heat_columns, rotation=45, ha="right")
ax.set_yticks(range(len(heat_columns)), heat_columns)
for i in range(len(heat_columns)):
    for j in range(len(heat_columns)):
        ax.text(j, i, f"{heat.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(heat.iloc[i,j]) > 0.65 else "black")
fig.colorbar(im, ax=ax, label="Pearson r", fraction=0.046)
ax.set_title("Price and selected high-correlation features")
save_plot(fig, "04_correlation_heatmap")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for column, ax in zip(["GrLivArea", "GarageArea", "YearBuilt"], axes):
    ax.scatter(houses[column], price, s=13, alpha=0.4, color="#526778", edgecolors="none")
    ax.yaxis.set_major_formatter(money)
    ax.set(xlabel=column, ylabel="Sale price", title=f"{column} vs SalePrice")
save_plot(fig, "05_price_scatterplots")

quality_summary = houses.groupby("OverallQual")["SalePrice"].agg(["count", "median", "mean"])
display(quality_summary.round(2))
quality_summary.to_csv(results_dir / "quality_summary.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
quality_levels = quality_summary.index.tolist()
ax.boxplot([houses.loc[houses["OverallQual"] == q, "SalePrice"] for q in quality_levels],
           tick_labels=[str(q) for q in quality_levels], patch_artist=True,
           boxprops={"facecolor": "#A8B5BF"}, medianprops={"color": "black"},
           flierprops={"marker": ".", "markersize": 3})
ax.set(xlabel="OverallQual (1 = lowest, 10 = highest)", ylabel="Sale price", title="Sale price by overall quality rating")
ax.yaxis.set_major_formatter(money)
save_plot(fig, "06_quality_boxplots")

neighborhood_summary = houses.groupby("Neighborhood")["SalePrice"].agg(["count", "median", "mean"]).sort_values("median")
display(neighborhood_summary.round(2))
neighborhood_summary.to_csv(results_dir / "neighborhood_summary.csv")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(neighborhood_summary.index, neighborhood_summary["median"], color="#526778")
for i, (_, row) in enumerate(neighborhood_summary.iterrows()):
    ax.text(row["median"] + 3500, i, f"n={int(row['count'])}", va="center", fontsize=8)
ax.set(xlabel="Median sale price", title="Neighbourhood medians (all groups; counts shown)", xlim=(0, 370000))
ax.xaxis.set_major_formatter(money)
save_plot(fig, "07_neighborhood_medians")

q1, q3 = price.quantile([0.25, 0.75])
iqr = q3 - q1
lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
price_extremes = houses.loc[(price < lower_fence) | (price > upper_fence)]
print(f"IQR: ${iqr:,.0f}; fences: ${lower_fence:,.2f} to ${upper_fence:,.2f}")
print(f"Flagged sale prices: {len(price_extremes)} of {len(houses)} ({len(price_extremes)/len(houses):.2%})")
print("Flagged records are retained in the main analysis.")
display(houses.nlargest(5, "GrLivArea")[["Id", "GrLivArea", "OverallQual", "SalePrice"]])
full_r = houses["GrLivArea"].corr(price)
restricted = houses.loc[houses["GrLivArea"] <= 4000]
restricted_r = restricted["GrLivArea"].corr(restricted["SalePrice"])
sensitivity = pd.DataFrame({"Sample": ["All homes", "GrLivArea <= 4000 (sensitivity only)"],
                            "n": [len(houses), len(restricted)], "Pearson r": [full_r, restricted_r]})
display(sensitivity.round(4))
sensitivity.to_csv(results_dir / "area_sensitivity.csv", index=False)
price_extremes[["Id", "SalePrice", "GrLivArea", "OverallQual"]].to_csv(results_dir / "price_extremes.csv", index=False)

processed_dir = project_root / "Data/processed"
processed_dir.mkdir(parents=True, exist_ok=True)
houses.to_csv(processed_dir / "houses_analysis.csv", index=False)
assert houses.shape == (1460, 81)
assert houses["Id"].is_unique and price.notna().all() and (price > 0).all()
assert len(price_extremes) == 61
assert houses["SalePrice"].equals(raw["SalePrice"])
print("Checks passed: shape, unique IDs, complete positive prices, outlier count, and unchanged outcome.")
