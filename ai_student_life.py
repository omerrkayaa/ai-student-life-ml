"""Can students' AI tool usage predict the reported impact on their grades?

Runs the whole analysis end to end: EDA tables/plots, mutual information
scores, and a comparison of six classifiers with three evaluation setups
(5-fold CV, hold-out, train/val/test). Everything is written to results/.

    python ai_student_life.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    permutation_test_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "AI_Student_Life_india.csv"
OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"

SEED = 42
TARGET = "Impact_on_Grades"
NUMERIC = ["Age", "Daily_Usage_Hours"]
CATEGORICAL = ["Gender", "Education_Level", "City", "AI_Tool_Used", "Purpose", "Satisfaction_Level"]

# Random guessing that keeps the class proportions. Every model is compared against this.
BASELINE = "Baseline (stratified random)"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    BASELINE: DummyClassifier(strategy="stratified", random_state=SEED),
}


def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model):
    return Pipeline([("prep", make_preprocessor()), ("model", clone(model))])


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=200)
    plt.close()


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    # Yates' correction would pull 2x2 tables (e.g. Gender vs Gender) below 1.
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    if min(r, k) == 1:
        return 0.0
    return float(np.sqrt(chi2 / n / (min(r, k) - 1)))


def macro_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def describe_data(df):
    # Rows 301-400 come from a second (Pakistan) sample and reuse IDs 1-100,
    # so Student_ID is not unique. It is dropped before modelling anyway.
    summary = {
        "shape": list(df.shape),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "repeated_student_ids": int(df["Student_ID"].duplicated().sum()),
        "target_distribution": {k: int(v) for k, v in df[TARGET].value_counts().items()},
        "numeric_summary": df[NUMERIC].describe().round(3).to_dict(),
    }
    with open(OUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"{df.shape[0]} rows, {df.shape[1]} columns")
    print("missing values:", int(df.isna().sum().sum()), "| repeated IDs:", summary["repeated_student_ids"])
    print("target distribution:", summary["target_distribution"])


def stacked_impact_plot(df, column, title, filename):
    table = pd.crosstab(df[column], df[TARGET], normalize="index").round(3)
    table.to_csv(OUT_DIR / f"impact_by_{filename}_normalized.csv")

    ax = table.plot(kind="bar", stacked=True, figsize=(9, 5), rot=0)
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Proportion")
    ax.legend(title="Impact on Grades", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(f"impact_by_{filename}_stacked.png")


def run_eda(df):
    for column, title, filename in [
        (TARGET, "Distribution of Impact on Grades", "target_distribution.png"),
        ("AI_Tool_Used", "Distribution of AI Tools Used", "ai_tools_distribution.png"),
    ]:
        ax = df[column].value_counts().plot(kind="bar", figsize=(7, 5), rot=0)
        ax.set_title(title)
        ax.set_ylabel("Number of Students")
        save_fig(filename)

    stacked_impact_plot(df, "AI_Tool_Used", "Impact on Grades by AI Tool Used", "ai_tool")
    stacked_impact_plot(df, "Purpose", "Impact on Grades by Purpose of AI Usage", "purpose")

    usage = df.groupby(TARGET)["Daily_Usage_Hours"].agg(["count", "mean", "std", "min", "max"]).round(3)
    usage.to_csv(OUT_DIR / "usage_hours_by_impact.csv")

    classes = sorted(df[TARGET].unique())
    plt.figure(figsize=(8, 5))
    plt.boxplot([df.loc[df[TARGET] == c, "Daily_Usage_Hours"] for c in classes], tick_labels=classes)
    plt.title("Daily Usage Hours by Academic Impact")
    plt.ylabel("Daily Usage Hours")
    save_fig("usage_hours_by_impact_boxplot.png")

    cols = CATEGORICAL + [TARGET]
    assoc = pd.DataFrame([[cramers_v(df[a], df[b]) for b in cols] for a in cols], index=cols, columns=cols)
    assoc.round(4).to_csv(OUT_DIR / "categorical_association_cramers_v.csv")

    plt.figure(figsize=(9, 7))
    plt.imshow(assoc.values, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = assoc.iat[i, j]
            plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="black" if v > 0.5 else "white")
    plt.title("Categorical Association (Cramér's V)")
    save_fig("categorical_association_matrix_cramers_v.png")


def mutual_information(X, y):
    # Only used to look at which features carry information about the target.
    # The models below are trained on all features.
    prep = make_preprocessor()
    X_enc = prep.fit_transform(X)
    y_enc = LabelEncoder().fit_transform(y)
    scores = mutual_info_classif(X_enc, y_enc, random_state=SEED)

    mi = pd.DataFrame({"Feature": prep.get_feature_names_out(), "Mutual_Information": scores})
    mi = mi.sort_values("Mutual_Information", ascending=False)
    mi.round(5).to_csv(OUT_DIR / "feature_selection_mutual_information.csv", index=False)

    top = mi.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["Feature"], top["Mutual_Information"])
    plt.title("Top 15 Features by Mutual Information")
    plt.xlabel("Mutual Information Score")
    save_fig("feature_selection_mutual_information_top15.png")


def evaluate_cv(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    rows = []
    for name, model in MODELS.items():
        s = cross_validate(make_pipeline(model), X, y, cv=cv, scoring=scoring)
        rows.append({
            "Model": name,
            "CV Accuracy Mean": s["test_accuracy"].mean(),
            "CV Accuracy Std": s["test_accuracy"].std(),
            "CV Precision Macro": s["test_precision_macro"].mean(),
            "CV Recall Macro": s["test_recall_macro"].mean(),
            "CV F1 Macro": s["test_f1_macro"].mean(),
            "CV F1 Std": s["test_f1_macro"].std(),
        })

    res = pd.DataFrame(rows).sort_values("CV F1 Macro", ascending=False)
    res.round(4).to_csv(OUT_DIR / "model_comparison_cross_validation.csv", index=False)

    plot_df = res.sort_values("CV F1 Macro")
    colors = ["tab:gray" if m == BASELINE else "tab:blue" for m in plot_df["Model"]]
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["Model"], plot_df["CV F1 Macro"], xerr=plot_df["CV F1 Std"], color=colors)
    plt.title("5-Fold CV Macro F1 (mean and std over folds)")
    plt.xlabel("Macro F1")
    save_fig("model_comparison_cv_f1.png")
    return res


def evaluate_holdout(X_train, X_test, y_train, y_test):
    rows = []
    for name, model in MODELS.items():
        pred = make_pipeline(model).fit(X_train, y_train).predict(X_test)
        s = macro_scores(y_test, pred)
        rows.append({
            "Model": name,
            "Holdout Accuracy": s["accuracy"],
            "Holdout Precision Macro": s["precision"],
            "Holdout Recall Macro": s["recall"],
            "Holdout F1 Macro": s["f1"],
        })

    res = pd.DataFrame(rows).sort_values("Holdout F1 Macro", ascending=False)
    res.round(4).to_csv(OUT_DIR / "model_comparison_holdout.csv", index=False)
    return res


def evaluate_three_way(X, y):
    # 60/20/20 split. Models are ranked on the validation set; the test set is
    # only looked at for the final numbers.
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED
    )

    rows = []
    for name, model in MODELS.items():
        pipe = make_pipeline(model).fit(X_train, y_train)
        val_pred, test_pred = pipe.predict(X_val), pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Validation Accuracy": accuracy_score(y_val, val_pred),
            "Validation F1 Macro": f1_score(y_val, val_pred, average="macro", zero_division=0),
            "Test Accuracy": accuracy_score(y_test, test_pred),
            "Test F1 Macro": f1_score(y_test, test_pred, average="macro", zero_division=0),
        })

    res = pd.DataFrame(rows).sort_values("Validation F1 Macro", ascending=False)
    res.round(4).to_csv(OUT_DIR / "model_comparison_three_way_split.csv", index=False)
    return res


def analyse_best_model(name, X, y, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(MODELS[name]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    labels = list(pipe.classes_)

    cm = confusion_matrix(y_test, pred, labels=labels)
    pd.DataFrame(
        cm, index=[f"Actual {c}" for c in labels], columns=[f"Predicted {c}" for c in labels]
    ).to_csv(OUT_DIR / "confusion_matrix_best_cv_model_holdout.csv")

    report = pd.DataFrame(classification_report(y_test, pred, zero_division=0, output_dict=True)).T
    report.round(4).to_csv(OUT_DIR / "classification_report_best_cv_model_holdout.csv")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.title(f"Confusion Matrix - {name} (hold-out test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_fig("confusion_matrix_best_cv_model.png")

    # Is the CV score better than what we'd get with shuffled labels?
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    score, perm_scores, p_value = permutation_test_score(
        make_pipeline(MODELS[name]), X, y, cv=cv, scoring="f1_macro",
        n_permutations=200, random_state=SEED, n_jobs=-1,
    )
    with open(OUT_DIR / "permutation_test_best_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": name,
            "cv_f1_macro": round(float(score), 4),
            "shuffled_f1_mean": round(float(perm_scores.mean()), 4),
            "shuffled_f1_std": round(float(perm_scores.std()), 4),
            "n_permutations": len(perm_scores),
            "p_value": round(float(p_value), 4),
        }, f, indent=2)

    print(f"\nPermutation test ({name}): F1={score:.3f}, "
          f"shuffled labels F1={perm_scores.mean():.3f} (std {perm_scores.std():.3f}), p={p_value:.3f}")
    print(report.round(3))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    describe_data(df)
    run_eda(df)

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    mutual_information(X, y)

    cv_res = evaluate_cv(X, y)
    print("\n5-fold CV\n", cv_res.round(4).to_string(index=False))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    print("\nHold-out\n", evaluate_holdout(X_train, X_test, y_train, y_test).round(4).to_string(index=False))
    print("\nTrain/val/test\n", evaluate_three_way(X, y).round(4).to_string(index=False))

    best = cv_res.loc[cv_res["Model"] != BASELINE, "Model"].iloc[0]
    print("\nBest model by CV F1:", best)
    analyse_best_model(best, X, y, X_train, X_test, y_train, y_test)

    print("\nResults written to", OUT_DIR)


if __name__ == "__main__":
    main()
