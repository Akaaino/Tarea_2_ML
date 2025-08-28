#Tarea 2 - ML
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TAREA_DIR = "C:/Users/Administrador/Desktop/Machine L" #cambiar a la ruta a directorio de su equipo
FIGURAS_DIR = os.path.join(TAREA_DIR, "Figuras")
SALIDA_DIR = os.path.join(TAREA_DIR, "Archivos_Generados")
os.makedirs(FIGURAS_DIR, exist_ok=True)
os.makedirs(SALIDA_DIR, exist_ok=True)

DATA_PATH = "C:/Users/Administrador/Desktop/Machine L/week3.csv" #cambiar a la ruta a directorio de su equipo

# Leer primera línea 
with open(DATA_PATH, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()

# Cargar datos
df = pd.read_csv(DATA_PATH, comment='#', header=None)
df.columns = ["X1", "X2", "y"]

# Scatter 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["X1"].values, df["X2"].values, df["y"].values)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
ax.set_title("Dispersión 3D de los datos (X1, X2, y)")
plt.savefig(os.path.join(FIGURAS_DIR, "scatter_3d_datos.png"), bbox_inches="tight", dpi=150)
plt.close(fig)

# Preparación de features
feature_names = ["X1", "X2"]
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(df[feature_names].values)
poly_feature_names = poly.get_feature_names_out(feature_names)

X = df[feature_names].values
y = df["y"].values

def alpha_from_C(C):
    return 1.0 / (2.0 * C)

def make_lasso(C):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha_from_C(C), max_iter=200000, tol=1e-6, random_state=42))
    ])

def make_ridge(C):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha_from_C(C), random_state=42))
    ])

#valores de C
C_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100, 300,500, 1000]

# Lasso coeficientes
rows_lasso = []
for C in C_values:
    lasso = make_lasso(C)
    lasso.fit(X, y)
    model = lasso.named_steps["model"]
    coefs = model.coef_
    intercept = model.intercept_
    for name, coef in zip(poly_feature_names, coefs):
        rows_lasso.append({"Modelo":"Lasso","C":C,"alpha":alpha_from_C(C),"Parámetro":name,"Coeficiente":coef})
    rows_lasso.append({"Modelo":"Lasso","C":C,"alpha":alpha_from_C(C),"Parámetro":"intercepto","Coeficiente":intercept})
pd.DataFrame(rows_lasso).to_csv(os.path.join(SALIDA_DIR,"coeficientes_lasso.csv"), index=False)

# Ridge coeficientes
rows_ridge = []
for C in C_values:
    ridge = make_ridge(C)
    ridge.fit(X, y)
    model = ridge.named_steps["model"]
    coefs = model.coef_
    intercept = model.intercept_
    for name, coef in zip(poly_feature_names, coefs):
        rows_ridge.append({"Modelo":"Ridge","C":C,"alpha":alpha_from_C(C),"Parámetro":name,"Coeficiente":coef})
    rows_ridge.append({"Modelo":"Ridge","C":C,"alpha":alpha_from_C(C),"Parámetro":"intercepto","Coeficiente":intercept})
pd.DataFrame(rows_ridge).to_csv(os.path.join(SALIDA_DIR,"coeficientes_ridge.csv"), index=False)

# Superficie: construir malla extendida
x1_min, x1_max = df["X1"].min(), df["X1"].max()
x2_min, x2_max = df["X2"].min(), df["X2"].max()
r1, r2 = (x1_max - x1_min), (x2_max - x2_min)
pad1, pad2 = max(1.0, 0.25*r1), max(1.0, 0.25*r2)
grid_x1 = np.linspace(x1_min - pad1, x1_max + pad1, 60)
grid_x2 = np.linspace(x2_min - pad2, x2_max + pad2, 60)
GX1, GX2 = np.meshgrid(grid_x1, grid_x2)
grid_points = np.c_[GX1.ravel(), GX2.ravel()]

C_show = sorted(set([min(C_values), 1e-2, 1, max(C_values)]))

# Lasso: superficies
for C in C_show:
    lasso = make_lasso(C)
    lasso.fit(X, y)
    y_pred_grid = lasso.predict(grid_points).reshape(GX1.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(GX1, GX2, y_pred_grid, linewidth=0, antialiased=True, alpha=0.7)
    ax.scatter(df["X1"].values, df["X2"].values, df["y"].values)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Predicción y")
    ax.set_title(f"Lasso - Superficie de predicción (C={C})")
    plt.savefig(os.path.join(FIGURAS_DIR, f"lasso_surface_C{C}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

# Ridge: superficies
for C in C_show:
    ridge = make_ridge(C)
    ridge.fit(X, y)
    y_pred_grid = ridge.predict(grid_points).reshape(GX1.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(GX1, GX2, y_pred_grid, linewidth=0, antialiased=True, alpha=0.7)
    ax.scatter(df["X1"].values, df["X2"].values, df["y"].values)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Predicción y")
    ax.set_title(f"Ridge - Superficie de predicción (C={C})")
    plt.savefig(os.path.join(FIGURAS_DIR, f"ridge_surface_C{C}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

# --- VALIDACIÓN CRUZADA CON VARIAS MÉTRICAS ---
def cv_metrics_for_Cs(model_maker, Cs, X, y, cv_splits=5):
    mae_means, mae_stds = [], []
    mse_means, mse_stds = [], []
    rmse_means, rmse_stds = [], []
    r2_means, r2_stds = [], []

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for C in Cs:
        maes, mses, rmses, r2s = [], [], [], []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = model_maker(C)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            maes.append(mean_absolute_error(y_test, y_pred))
            mses.append(mean_squared_error(y_test, y_pred))
            rmses.append(np.sqrt(mses[-1]))
            r2s.append(r2_score(y_test, y_pred))

        mae_means.append(np.mean(maes))
        mae_stds.append(np.std(maes, ddof=1))
        mse_means.append(np.mean(mses))
        mse_stds.append(np.std(mses, ddof=1))
        rmse_means.append(np.mean(rmses))
        rmse_stds.append(np.std(rmses, ddof=1))
        r2_means.append(np.mean(r2s))
        r2_stds.append(np.std(r2s, ddof=1))

    return {
        "MAE_mean": np.array(mae_means),
        "MAE_std": np.array(mae_stds),
        "MSE_mean": np.array(mse_means),
        "MSE_std": np.array(mse_stds),
        "RMSE_mean": np.array(rmse_means),
        "RMSE_std": np.array(rmse_stds),
        "R2_mean": np.array(r2_means),
        "R2_std": np.array(r2_stds)
    }

C_grid = np.logspace(-4, 3, 20)

# Lasso
lasso_metrics = cv_metrics_for_Cs(make_lasso, C_grid, X, y, cv_splits=5)

# Ridge
ridge_metrics = cv_metrics_for_Cs(make_ridge, C_grid, X, y, cv_splits=5)

# Guardar resultados en CSV
cv_df = pd.DataFrame({
    "C": C_grid,
    # Lasso
    "Lasso_MAE_mean": lasso_metrics["MAE_mean"],
    "Lasso_MAE_std": lasso_metrics["MAE_std"],
    "Lasso_MSE_mean": lasso_metrics["MSE_mean"],
    "Lasso_MSE_std": lasso_metrics["MSE_std"],
    "Lasso_RMSE_mean": lasso_metrics["RMSE_mean"],
    "Lasso_RMSE_std": lasso_metrics["RMSE_std"],
    "Lasso_R2_mean": lasso_metrics["R2_mean"],
    "Lasso_R2_std": lasso_metrics["R2_std"],
    # Ridge
    "Ridge_MAE_mean": ridge_metrics["MAE_mean"],
    "Ridge_MAE_std": ridge_metrics["MAE_std"],
    "Ridge_MSE_mean": ridge_metrics["MSE_mean"],
    "Ridge_MSE_std": ridge_metrics["MSE_std"],
    "Ridge_RMSE_mean": ridge_metrics["RMSE_mean"],
    "Ridge_RMSE_std": ridge_metrics["RMSE_std"],
    "Ridge_R2_mean": ridge_metrics["R2_mean"],
    "Ridge_R2_std": ridge_metrics["R2_std"],
})

cv_df.to_csv(os.path.join(SALIDA_DIR, "cv_results_all_metricas.csv"), index=False)

# Baseline RMSE (para referencia visual)
baseline_rmses = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    y_train, y_test = y[train_idx], y[test_idx]
    y_hat = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    baseline_rmses.append(np.sqrt(mean_squared_error(y_test, y_hat)))
baseline_mean = float(np.mean(baseline_rmses))

# --- Gráficas CV para todas las métricas ---
metrics = ["MAE", "MSE", "RMSE", "R2"]
for metric in metrics:
    plt.figure()
    # Lasso
    plt.errorbar(C_grid, lasso_metrics[f"{metric}_mean"], yerr=lasso_metrics[f"{metric}_std"],
                 fmt='o-', capsize=3, label=f"Lasso (5-fold)")
    # Ridge
    plt.errorbar(C_grid, ridge_metrics[f"{metric}_mean"], yerr=ridge_metrics[f"{metric}_std"],
                 fmt='s-', capsize=3, label=f"Ridge (5-fold)")
    
    # Línea de baseline solo para métricas de error
    if metric in ["MAE", "MSE", "RMSE"]:
        plt.axhline(baseline_mean, linestyle="--", color="gray",
                    label=f"Baseline media (RMSE≈{baseline_mean:.3f})")
    
    plt.xscale("log")
    plt.xlabel("C (escala log)")
    plt.ylabel(metric)
    plt.title(f"Validación cruzada 5-fold: {metric} vs C")
    plt.legend()
    plt.savefig(os.path.join(FIGURAS_DIR, f"cv_{metric.lower()}_vs_C.png"), bbox_inches="tight", dpi=150)
    plt.close()

print("Terminado, Salidas en:", TAREA_DIR)
