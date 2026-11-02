# Data provenance and preparation

The authoritative source is the supplied `train(2).csv` (1,460 rows, 81 columns). `raw/train.csv` preserves its exact bytes. Id is a unique record identifier; SalePrice is the outcome. No external data is merged.

The notebook and `Src/analysis.py` load with `keep_default_na=False` to retain literal codes. In the analysis copy, exact `NA` codes become unavailable values; numeric measurement columns LotFrontage, MasVnrArea, and GarageYrBlt are converted to numbers. The literal `None` category in MasVnrType remains a category. No values are imputed and no rows are removed. The resulting analysis copy is written to `processed/houses_analysis.csv`. Missing feature codes are not automatically treated as zero or interpreted as unknown: a full data dictionary is needed to distinguish absence from missing measurements.

All 1,460 records remain in the main EDA. Numeric correlations use available pairs and report n. Id, MSSubClass, MoSold, and YrSold are excluded from the exploratory numerical feature ranking. OverallQual is ordinal and is also examined with Spearman correlation and grouped boxplots.

The sensitivity analysis uses GrLivArea <= 4,000 only for comparison; it does not change the main dataset. Price extremes are flags, not confirmed errors. The NA-code table, processed CSV, result tables, and plots are produced by the notebook/script.
