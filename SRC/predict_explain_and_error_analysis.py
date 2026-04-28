"""
predict_explain_and_error_analysis.py
───────────────────────────────────────
Runs test cases, SHAP explanations, and false positive/negative analysis
for all three phishing-detection models in the Pimers project.

Requirements:  pip install shap pandas scikit-learn joblib
Run from project root:  python predict_explain_and_error_analysis.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────
TEST_SIZE        = 10
TOP_FEATURES     = 5
BACKGROUND_ROWS  = 100
LINE             = "=" * 70
THIN             = "-" * 70


# ─────────────────────────────────────────────────────────────────────────────
class PhishingModelTester:
    """Loads a trained model and runs predictions, explanations, error analysis."""

    def __init__(self, name, model_path, x_path, y_path, pos_label):
        self.name       = name
        self.pos_label  = pos_label
        self.model      = joblib.load(model_path)
        self.x_test, self.y_test = self._get_test_data(x_path, y_path)
        self.features   = list(self.x_test.columns)
        self.predictions = self.model.predict(self.x_test)
        self.explainer  = shap.TreeExplainer(
            self.model,
            data=self.x_test.iloc[:BACKGROUND_ROWS],
            feature_perturbation="interventional"
        )

    # ── data ─────────────────────────────────────────────────────────────────
    def _get_test_data(self, x_path, y_path):
        x = pd.read_csv(x_path)
        y = pd.read_csv(y_path).squeeze()
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return (
            x_test.iloc[:TEST_SIZE].reset_index(drop=True),
            y_test.iloc[:TEST_SIZE].reset_index(drop=True),
        )

    # ── prediction type ───────────────────────────────────────────────────────
    def _case_type(self, true, pred):
        is_pos_true = (true == self.pos_label)
        is_pos_pred = (pred == self.pos_label)
        if is_pos_true  and is_pos_pred:     return "TP"
        if not is_pos_true and not is_pos_pred: return "TN"
        if not is_pos_true and is_pos_pred:  return "FP"
        return "FN"

    # ── shap values for one row ───────────────────────────────────────────────
    def _shap_for_row(self, row_index):
        row  = self.x_test.iloc[row_index].values.reshape(1, -1)
        vals = self.explainer.shap_values(row)
        if isinstance(vals, list):
            vals = vals[1]          # positive-class SHAP values
        return vals.flatten()

    # ── print: test case table ────────────────────────────────────────────────
    def print_predictions(self):
        print(f"\n{THIN}")
        print(f"  {'#':<4} {'TRUE':>14}  {'PREDICTED':>14}  {'OK?':<6}  TYPE")
        print(THIN)

        for i in range(len(self.x_test)):
            true = self.y_test.iloc[i]
            pred = self.predictions[i]
            ok   = "✔" if true == pred else "✘"
            tag  = self._case_type(true, pred)
            print(f"  {i+1:<4} {str(true):>14}  {str(pred):>14}  {ok:<6}  {tag}")

    # ── print: shap explanations ──────────────────────────────────────────────
    def print_shap_explanations(self):
        print(f"\n{THIN}")
        print("  SHAP EXPLANATIONS  (top features per prediction)")
        print(THIN)

        for i in range(len(self.x_test)):
            true  = self.y_test.iloc[i]
            pred  = self.predictions[i]
            ok    = "✔ CORRECT" if true == pred else "✘ WRONG"
            tag   = self._case_type(true, pred)
            shap_vals = self._shap_for_row(i)

            ranked = sorted(
                zip(self.features, shap_vals),
                key=lambda p: abs(p[1]),
                reverse=True
            )

            print(f"\n  Test #{i+1}  |  True={true}  →  Pred={pred}  |  {ok}  [{tag}]")
            for feat, sv in ranked[:TOP_FEATURES]:
                arrow = "↑ pushes PHISHING" if sv > 0 else "↓ pushes SAFE"
                print(f"    {feat:<35} SHAP={sv:+.4f}  {arrow}")

    # ── print: false positive / negative analysis ─────────────────────────────
    def print_error_analysis(self):
        print(f"\n{THIN}")
        print("  FALSE POSITIVE / FALSE NEGATIVE ANALYSIS")
        print(THIN)

        # confusion matrix
        labels = sorted(set(self.y_test))
        try:
            cm = confusion_matrix(self.y_test, self.predictions, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index   =[f"True {l}" for l in labels],
                columns =[f"Pred {l}" for l in labels],
            )
            print("\n  Confusion Matrix:")
            print(cm_df.to_string())
        except Exception:
            pass

        # collect FP and FN rows
        fp_rows, fn_rows, error_list = [], [], []
        for i in range(len(self.x_test)):
            true = self.y_test.iloc[i]
            pred = self.predictions[i]
            tag  = self._case_type(true, pred)
            if tag == "FP":
                fp_rows.append(i)
                error_list.append((i + 1, "FP", true, pred,
                    "Safe sample wrongly flagged as phishing"))
            elif tag == "FN":
                fn_rows.append(i)
                error_list.append((i + 1, "FN", true, pred,
                    "Phishing sample missed — predicted safe"))

        # individual error list
        if error_list:
            print(f"\n  Errors found in {TEST_SIZE} test cases:")
            for num, etype, true, pred, reason in error_list:
                print(f"    • Test #{num}: [{etype}]  True={true}  Pred={pred}  → {reason}")
        else:
            print(f"\n  No errors in these {TEST_SIZE} test cases.")

        # feature-level breakdown per error group
        for group_label, indices in [
            ("FALSE POSITIVES — features that caused safe samples to look phishing", fp_rows),
            ("FALSE NEGATIVES — features that caused phishing to look safe",         fn_rows),
        ]:
            print(f"\n  ▶ {group_label}")
            if not indices:
                print("    None in this test set.")
                continue
            print(f"    Count: {len(indices)}")
            shap_matrix = np.array([np.abs(self._shap_for_row(i)) for i in indices])
            mean_abs    = shap_matrix.mean(axis=0)
            top_idx     = np.argsort(mean_abs)[::-1][:TOP_FEATURES]
            print(f"    Top {TOP_FEATURES} responsible features (avg |SHAP|):")
            for idx in top_idx:
                print(f"      {self.features[idx]:<35} avg |SHAP|={mean_abs[idx]:.4f}")

        print(f"\n  Summary → FP: {len(fp_rows)}   FN: {len(fn_rows)}"
              f"   (out of {TEST_SIZE} test cases)")

    # ── run everything ────────────────────────────────────────────────────────
    def run(self):
        print(f"\n{LINE}")
        print(f"  MODEL: {self.name}")
        print(LINE)
        self.print_predictions()
        self.print_shap_explanations()
        self.print_error_analysis()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    model_configs = [
        {
            "name":       "Phishing URL",
            "model_path": os.path.join("SRC", "Models", "Phishing_url_model.pkl"),
            "x_path":     "Data/Processed/x_url.csv",
            "y_path":     "Data/Processed/y_url.csv",
            "pos_label":  1,
        },
        {
            "name":       "Phishing Website",
            "model_path": os.path.join("SRC", "Models", "Phishing_website_model.pkl"),
            "x_path":     "Data/Processed/x_website.csv",
            "y_path":     "Data/Processed/y_website.csv",
            "pos_label":  "phishing",
        },
        {
            "name":       "Social Engineering",
            "model_path": os.path.join("SRC", "Models", "Social_model.pkl"),
            "x_path":     "Data/Processed/x_social.csv",
            "y_path":     "Data/Processed/y_social.csv",
            "pos_label":  "Attack",
        },
    ]

    print(f"\n{LINE}")
    print("  PHISHING DETECTION — PREDICTIONS, EXPLANATIONS & ERROR ANALYSIS")
    print(LINE)

    for cfg in model_configs:
        tester = PhishingModelTester(**cfg)
        tester.run()

    print(f"\n{LINE}")
    print("  DONE")
    print(LINE)


if __name__ == "__main__":
    main()
