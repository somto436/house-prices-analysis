# House prices: exploratory data analysis

Somtochukwu Kingsley Ekeh | Assignment 2 - Part 3

## Purpose
Describe sale prices, study distributions and correlations, and compare overall-quality and neighbourhood groups in the supplied House Prices training data. This is EDA, not a prediction model.

## Layout
- `Src/Assignment_2_Part_3_Completed.ipynb`: completed notebook with saved outputs.
- `Src/analysis.py`: runnable export of the notebook's analysis.
- `Data/raw/train.csv`: original supplied CSV, unchanged.
- `Data/processed/houses_analysis.csv`: analysis copy with NA codes represented as unavailable.
- `Data/README.md`: provenance and transformation notes.
- `Docs/House_Prices_EDA_Report.pdf`: one-page findings report.
- `Results/`: seven charts, summary tables, correlations, and sensitivity checks.
- `requirements.txt`: dependencies.

## Run
Use Python 3.10 or later. From the project root:

```bash
python -m pip install -r requirements.txt
python Src/analysis.py
jupyter lab Src/Assignment_2_Part_3_Completed.ipynb
```

Run notebook cells in order. A standalone notebook can load `train.csv` or `train(2).csv` from its own folder. No web download is needed. The script regenerates charts, result tables, and the processed CSV; the PDF is included as a static report.

## Findings
The 1,460 prices have mean $180,921.20 and median $163,000. Raw-price skewness is 1.883, versus 0.121 after a natural-log transformation. OverallQual and GrLivArea have the largest observed Pearson price correlations (0.791 and 0.709); quality is ordinal, so Spearman is also reported. Neighbourhoods differ in median price. Extreme observations are retained, and an explicitly labelled area sensitivity check is included.

## Limits
Associations are unadjusted and do not imply causation or future predictive accuracy. Numeric category codes are excluded from the feature ranking. NA can reflect absent features or unknown values; no blanket zero or mean imputation is used, and the raw input is preserved. Group sample sizes and available-pair counts are reported.

## GitHub
https://github.com/somto436/house-prices-analysis

## Sources
Supplied `train(2).csv`, saved unchanged as `Data/raw/train.csv`. Assignment background: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/code
Project structure follows the assignment and supplied project-organization reading.
