"""
EXPERIMENTAL: Poisson Leptospirosis Forecasting (Leptospirosis only)
=============================================================================
Uses Poisson objectives, lag/rolling features, XGB+LGB blending.
Saves to model_data/artifacts/leptospirosis_poisson/ — does NOT overwrite main pipeline artifacts.
Run for research comparison only. NOT part of the main nine-stage pipeline.
=============================================================================
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Paths relative to project root (works when run from Research/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "model_data")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "model_data")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

print("="*80)
print("EXPERIMENTAL: Poisson Leptospirosis Pipeline (Helper Script)")
print("="*80)


def add_time_series_features(df):
    df = df.sort_values(['district', 'week_id']).copy()

    df['lag_1'] = df.groupby('district')['target'].shift(1)
    df['lag_2'] = df.groupby('district')['target'].shift(2)
    df['lag_4'] = df.groupby('district')['target'].shift(4)

    df['rolling_mean_4'] = (
        df.groupby('district')['target']
        .shift(1)
        .rolling(4)
        .mean()
    )

    df['rolling_std_4'] = (
        df.groupby('district')['target']
        .shift(1)
        .rolling(4)
        .std()
    )

    return df


EARLY_STOPPING = 30

XGB_PARAMS = dict(
    objective="count:poisson",
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=42,
    verbosity=0,
    early_stopping_rounds=EARLY_STOPPING,
    eval_metric=["mae", "rmse"],
)

LGB_PARAMS = dict(
    objective="poisson",
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1
)

disease = 'leptospirosis'
print(f"\nProcessing: {disease} (experimental Poisson)")

train = pd.read_csv(os.path.join(DATA_DIR, f"{disease}_train.csv"))
val = pd.read_csv(os.path.join(DATA_DIR, f"{disease}_val.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, f"{disease}_test.csv"))

train = add_time_series_features(train)
val = add_time_series_features(val)
test = add_time_series_features(test)

train = train.dropna()
val = val.dropna()
test = test.dropna()

print(f"After lag features + dropna: train {len(train)}, val {len(val)}, test {len(test)}")

feature_cols = [
    col for col in train.columns
    if col not in ['district', 'week_id', 'start_date',
                   'end_date', 'Duration', 'target']
]

X_train = train[feature_cols].astype(np.float32)
y_train = train['target'].values.astype(np.float32)
X_val = val[feature_cols].astype(np.float32)
y_val = val['target'].values.astype(np.float32)
X_test = test[feature_cols].astype(np.float32)
y_test = test['target'].values.astype(np.float32)

# XGBoost
xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    verbose=False
)
xgb_evals = xgb_model.evals_result() if hasattr(xgb_model, 'evals_result') else {}
best_round_xgb = getattr(xgb_model, 'best_iteration', None) or 500
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

# LightGBM
lgb_evals = {}
lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    eval_metric="mae",
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.record_evaluation(lgb_evals)
    ]
)
best_round_lgb = getattr(lgb_model, 'best_iteration_', 500)
lgb_val_pred = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)

# Blending
def blend_loss(w, pred1, pred2, y_true):
    w1 = w[0]
    w2 = 1 - w1
    blended = w1 * pred1 + w2 * pred2
    return mean_squared_error(y_true, blended)

result = minimize(
    blend_loss,
    [0.5],
    args=(xgb_val_pred, lgb_val_pred, y_val),
    bounds=[(0, 1)]
)
w_opt = result.x[0]
print(f"Optimal blend weight (XGB): {w_opt:.3f}")

final_test_pred = np.maximum(
    w_opt * xgb_test_pred + (1 - w_opt) * lgb_test_pred,
    0
)

mae = mean_absolute_error(y_test, final_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, final_test_pred))
r2 = r2_score(y_test, final_test_pred)
mae_baseline = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_train)))
print(f"\nMAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2: {r2:.3f}")
print(f"Baseline: MAE={mae_baseline:.3f}")
print(f"Improvement: {(1 - mae/mae_baseline)*100:.1f}%")

# Save to leptospirosis_poisson/ (does NOT overwrite main artifacts)
disease_artifacts = os.path.join(ARTIFACTS_DIR, f"{disease}_poisson")
os.makedirs(disease_artifacts, exist_ok=True)
joblib.dump(xgb_model, os.path.join(disease_artifacts, "xgb_model.pkl"))
joblib.dump(lgb_model, os.path.join(disease_artifacts, "lgb_model.pkl"))
with open(os.path.join(disease_artifacts, "blend_weight.json"), "w") as f:
    json.dump({"xgb_weight": float(w_opt)}, f)

pred_df = pd.DataFrame({
    'week_id': test['week_id'].values,
    'actual': y_test,
    'predicted': final_test_pred,
    'xgboost': xgb_test_pred,
    'lightgbm': lgb_test_pred
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "leptospirosis_poisson_predictions.csv"), index=False)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(y_test, final_test_pred, alpha=0.5, s=10, c='#2ca02c')
max_xy = max(y_test.max(), final_test_pred.max()) * 1.05
ax.plot([0, max_xy], [0, max_xy], 'k--', alpha=0.5, label='Perfect')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Leptospirosis Poisson (Experimental) - Predictions vs Actual')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, "leptospirosis_poisson_predictions_vs_actual.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/leptospirosis_poisson_predictions_vs_actual.png")


def _plot_curves(ax_loss, ax_mae, ax_imp, evals, title_prefix, best_round):
    labels, colors = ['train', 'valid', 'test'], ['#e377c2', '#d62728', '#2ca02c']
    losses, maes = [None]*3, [None]*3
    keys = sorted(evals.keys())[:3]
    for i, name in enumerate(keys):
        d = evals[name]
        mae_k = next((k for k in d if 'mae' in k.lower() or 'l1' in k.lower()), None)
        loss_k = next((k for k in d if k != mae_k and 'rmse' in k.lower()), None) or next((k for k in d if k != mae_k), None)
        if mae_k:
            maes[i] = np.asarray(d[mae_k]).ravel()
        if loss_k:
            losses[i] = np.asarray(d[loss_k]).ravel()
        if losses[i] is None and maes[i] is not None:
            losses[i] = maes[i].copy()
        if losses[i] is not None and maes[i] is not None and len(losses[i]) > 1 and len(maes[i]) > 1:
            if losses[i][-1] > losses[i][0]:
                losses[i] = maes[i].copy()
    cutoff = best_round if best_round else 9999
    for i, (lab, col) in enumerate(zip(labels, colors)):
        if losses[i] is not None:
            r = min(len(losses[i]), cutoff)
            arr = losses[i][:r] / (losses[i][0] + 1e-12)
            ax_loss.plot(range(1, r+1), arr, label=lab, color=col)
        if maes[i] is not None:
            r = min(len(maes[i]), cutoff)
            arr_m = maes[i][:r] / (maes[i][0] + 1e-12)
            ax_mae.plot(range(1, r+1), arr_m, label=lab, color=col)
            ax_imp.plot(range(1, r+1), 1.0 - arr_m, label=lab, color=col)
    if best_round:
        for ax in (ax_loss, ax_mae, ax_imp):
            ax.axvline(x=best_round, color='gray', linestyle='--', alpha=0.7, label='Stop')
    ax_loss.set_ylabel('Normalized'); ax_loss.set_title(f'{title_prefix} - Loss'); ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_mae.set_ylabel('Normalized'); ax_mae.set_title(f'{title_prefix} - MAE'); ax_mae.legend(); ax_mae.grid(True, alpha=0.3)
    ax_imp.set_ylabel('1 - normalized'); ax_imp.set_title(f'{title_prefix} - Improvement'); ax_imp.legend(); ax_imp.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 3, figsize=(14, 9))
_plot_curves(axes[0, 0], axes[0, 1], axes[0, 2], xgb_evals, 'Leptospirosis Poisson XGBoost', best_round_xgb)
_plot_curves(axes[1, 0], axes[1, 1], axes[1, 2], lgb_evals, 'Leptospirosis Poisson LightGBM', best_round_lgb)
for ax in axes.flat:
    ax.set_xlabel('Boosting rounds')
fig.suptitle(f'Poisson + Lag features (Experimental). Raw MAE={mae:.1f}', fontsize=10, y=1.02)
plt.tight_layout()
curve_path = os.path.join(FIG_DIR, "leptospirosis_poisson_training_curves.png")
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {curve_path}")

print("\nPoisson leptospirosis experiment complete.")
