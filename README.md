# AI-Optimized DC Parametric Test Time Reduction

A Streamlit prototype for semiconductor DC parametric test analysis, predictive measurement, anomaly screening, settling-time analysis and risk-aware test optimization.

## Dataset
This project uses the synthetic demonstration dataset supplied with the assignment:
- `data/dc_test_results.csv`
- `data/settling_sweep.csv`
- `data/test_catalog.csv`

The dataset is synthetic, not real production semiconductor data.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this project folder to a public GitHub repository (or a repository accessible to Streamlit Cloud).
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create an app, select the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy.

## Safety and limitations
- This is an offline prototype using synthetic data.
- It does not connect to automatic test equipment (ATE).
- It does not authorize removing or skipping production tests.
- Test-reduction candidates require held-out-lot validation, escape-risk confidence bounds, measurement capability studies (Gage R&R/Cpk), monitoring, rollback and quality/customer approval.
- The predictive model holds out entire lots for evaluation.
- `is_latent_defect_truth` is used only to score anomaly detection, never as a model feature.
