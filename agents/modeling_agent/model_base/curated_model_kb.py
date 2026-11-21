"""
Curated Model Knowledge Base for the Modeling Agent.
~30 most commonly used models for geospatial research, organized by category.
The LLM reads this directly and selects the best-fit model for each step.
"""

CURATED_MODEL_KB = ({
    "statistical_regression": [
        {
            "name": "OLS (Ordinary Least Squares)",
            "import_path": "spreg.OLS",
            "key_params": "y, x, w (optional), name_y, name_x, name_ds",
            "use_when": "Standard linear regression; baseline model before spatial models",
            "example": "from spreg import OLS\nmodel = OLS(y, x, w=w, name_y='depvar', name_x=x_names, name_ds='data')\nprint(model.summary)"
        },
        {
            "name": "Logistic Regression",
            "import_path": "sklearn.linear_model.LogisticRegression",
            "key_params": "penalty, C, solver, max_iter",
            "use_when": "Binary or multiclass classification of categorical outcomes",
            "example": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)"
        },
        {
            "name": "Poisson Regression",
            "import_path": "statsmodels.api.GLM with family=Poisson",
            "key_params": "endog, exog, family=sm.families.Poisson()",
            "use_when": "Count data (e.g., number of visits, incidents); non-negative integers",
            "example": "import statsmodels.api as sm\nmodel = sm.GLM(y, X, family=sm.families.Poisson()).fit()\nprint(model.summary())"
        },
        {
            "name": "Negative Binomial Regression",
            "import_path": "statsmodels.api.GLM with family=NegativeBinomial",
            "key_params": "endog, exog, family=sm.families.NegativeBinomial(alpha=1.0)",
            "use_when": "Overdispersed count data where Poisson variance assumption fails",
            "example": "import statsmodels.api as sm\nmodel = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()\nprint(model.summary())"
        },
    ],
    "spatial_regression": [
        {
            "name": "Spatial Lag Model (GM_Lag)",
            "import_path": "spreg.GM_Lag",
            "key_params": "y, x, w, name_y, name_x, name_w, name_ds",
            "use_when": "Spatial dependence in the dependent variable; neighboring values influence outcomes",
            "example": """
            from spreg import GM_Lag
import pandas as pd

# Fit spatial lag model (GM_Lag automatically includes a constant)
model = GM_Lag(
    y,
    x,
    w=w,
    name_y='depvar',
    name_x=x_names,
    name_w='queen',
    name_ds='data'
)

# Print full model summary (recommended)
print(model.summary)

# Extract coefficients including spatial lag parameter (rho)
results_df = pd.DataFrame({
    'Variable': model.name_x + ['Spatial_Lag_Rho'],
    'Coefficient': model.betas.flatten(),
    'Std_Error': model.std_err.flatten(),
    'Z_stat': model.z_stat[:, 0],
    'P_value': model.z_stat[:, 1]
})

print(results_df)"""
        },
        {
            "name": "Spatial Error Model (GM_Error)",
            "import_path": "spreg.GM_Error",
            "key_params": "y, x, w, name_y, name_x, name_w, name_ds",
            "use_when": "Spatial autocorrelation in error terms; omitted spatially-correlated variables",
            "example": "from spreg import GM_Error\nmodel = GM_Error(y, x, w=w, name_y='depvar', name_x=x_names, name_w='queen', name_ds='data')\nprint(model.summary)"
        },
{
            "name": "GWR (Geographically Weighted Regression)",
            "import_path": "mgwr.gwr.GWR",
            "key_params": "coords, y, X, bw, kernel='bisquare', fixed=False",
            "use_when": "Spatially varying relationships; coefficients change across geography",
            "example": "from mgwr.gwr import GWR\nfrom mgwr.sel_bw import Sel_BW\nbw = Sel_BW(coords, y, X).search()\nmodel = GWR(coords, y, X, bw).fit()\nprint(model.summary())"
        },
        {
            "name": "MGWR (Multiscale GWR)",
            "import_path": "mgwr.gwr.MGWR",
            "key_params": "coords, y, X, selector (from Sel_BW), kernel='bisquare', fixed=False",
            "use_when": "Different variables operate at different spatial scales",
            "example": "from mgwr.gwr import MGWR\nfrom mgwr.sel_bw import Sel_BW\nselector = Sel_BW(coords, y, X, multi=True)\nbws = selector.search()\nmodel = MGWR(coords, y, X, selector).fit()\nprint(model.summary())"
        },
    ],
    "exploratory_spatial_analysis": [
        {
            "name": "Moran's I (Global)",
            "import_path": "esda.Moran",
            "key_params": "y, w, permutations=999",
            "use_when": "Test for global spatial autocorrelation; is the variable spatially clustered?",
            "example": "from esda import Moran\nimport libpysal\nw = libpysal.weights.Queen.from_dataframe(gdf)\nmi = Moran(gdf['variable'].values, w)\nprint(f'I={mi.I:.4f}, p={mi.p_sim:.4f}')"
        },
        {
            "name": "Local Moran's I (LISA)",
            "import_path": "esda.Moran_Local",
            "key_params": "y, w, permutations=999",
            "use_when": "Identify local clusters and outliers (HH, LL, HL, LH hotspots)",
            "example": "from esda import Moran_Local\nlisa = Moran_Local(gdf['variable'].values, w)\ngdf['lisa_q'] = lisa.q  # quadrant labels\ngdf['lisa_p'] = lisa.p_sim"
        },
        {
            "name": "Getis-Ord G* (Local)",
            "import_path": "esda.getisord.G_Local",
            "key_params": "y, w, transform='B', permutations=999, star=True",
            "use_when": "Identify statistically significant hot spots and cold spots",
            "example": "from esda.getisord import G_Local\ng_local = G_Local(gdf['variable'].values, w, transform='B', star=True)\ngdf['Gi_star'] = g_local.Zs\ngdf['Gi_p'] = g_local.p_sim"
        },
        {
            "name": "Getis-Ord G (Global)",
            "import_path": "esda.getisord.G",
            "key_params": "y, w, permutations=999",
            "use_when": "Test for global high/low clustering tendency",
            "example": "from esda.getisord import G\ng = G(gdf['variable'].values, w)\nprint(f'G={g.G:.4f}, p={g.p_sim:.4f}')"
        }

    ]})
#     ,)
#         {
#             "name": "2SLS (Two-Stage Least Squares)",
#             "import_path": "spreg.TSLS",
#             "key_params": "y, x, yend, q, w (optional), name_y, name_x",
#             "use_when": "Endogenous variables present; instrumental variable regression",
#             "example": "from spreg import TSLS\nmodel = TSLS(y, x, yend=yend, q=q, w=w, name_y='depvar', name_x=x_names)"
#         },
#         {
#             "name": "Logistic Regression",
#             "import_path": "sklearn.linear_model.LogisticRegression",
#             "key_params": "penalty, C, solver, max_iter",
#             "use_when": "Binary or multiclass classification of categorical outcomes",
#             "example": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)"
#         },
#         {
#             "name": "Poisson Regression",
#             "import_path": "statsmodels.api.GLM with family=Poisson",
#             "key_params": "endog, exog, family=sm.families.Poisson()",
#             "use_when": "Count data (e.g., number of visits, incidents); non-negative integers",
#             "example": "import statsmodels.api as sm\nmodel = sm.GLM(y, X, family=sm.families.Poisson()).fit()\nprint(model.summary())"
#         },
#         {
#             "name": "Negative Binomial Regression",
#             "import_path": "statsmodels.api.GLM with family=NegativeBinomial",
#             "key_params": "endog, exog, family=sm.families.NegativeBinomial(alpha=1.0)",
#             "use_when": "Overdispersed count data where Poisson variance assumption fails",
#             "example": "import statsmodels.api as sm\nmodel = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()\nprint(model.summary())"
#         },
#         {
#             "name": "Weighted Least Squares",
#             "import_path": "statsmodels.api.WLS",
#             "key_params": "endog, exog, weights",
#             "use_when": "Heteroscedastic data; observations have different variances",
#             "example": "import statsmodels.api as sm\nmodel = sm.WLS(y, X, weights=w).fit()\nprint(model.summary())"
#         },
#         {
#             "name": "Lasso Regression",
#             "import_path": "sklearn.linear_model.Lasso",
#             "key_params": "alpha, max_iter",
#             "use_when": "Feature selection via L1 regularization; sparse models",
#             "example": "from sklearn.linear_model import Lasso\nmodel = Lasso(alpha=0.1)\nmodel.fit(X_train, y_train)"
#         },
#         {
#             "name": "Ridge Regression",
#             "import_path": "sklearn.linear_model.Ridge",
#             "key_params": "alpha",
#             "use_when": "Multicollinearity; L2 regularization to stabilize coefficients",
#             "example": "from sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)\nmodel.fit(X_train, y_train)"
#         },
#     ],
#     "spatial_regression": [
#         {
#             "name": "Spatial Lag Model (GM_Lag)",
#             "import_path": "spreg.GM_Lag",
#             "key_params": "y, x, w, name_y, name_x, name_w, name_ds",
#             "use_when": "Spatial dependence in the dependent variable; neighboring values influence outcomes",
#             "example": "from spreg import GM_Lag\nmodel = GM_Lag(y, x, w=w, name_y='depvar', name_x=x_names, name_w='queen', name_ds='data')\nprint(model.summary)"
#         },
#         {
#             "name": "Spatial Error Model (GM_Error)",
#             "import_path": "spreg.GM_Error",
#             "key_params": "y, x, w, name_y, name_x, name_w, name_ds",
#             "use_when": "Spatial autocorrelation in error terms; omitted spatially-correlated variables",
#             "example": "from spreg import GM_Error\nmodel = GM_Error(y, x, w=w, name_y='depvar', name_x=x_names, name_w='queen', name_ds='data')\nprint(model.summary)"
#         },
#         {
#             "name": "Spatial Lag+Error Combo (GM_Combo)",
#             "import_path": "spreg.GM_Combo",
#             "key_params": "y, x, w, name_y, name_x, name_w, name_ds",
#             "use_when": "Both spatial lag and spatial error dependence present",
#             "example": "from spreg import GM_Combo\nmodel = GM_Combo(y, x, w=w, name_y='depvar', name_x=x_names)\nprint(model.summary)"
#         },
#         {
#             "name": "GWR (Geographically Weighted Regression)",
#             "import_path": "mgwr.gwr.GWR",
#             "key_params": "coords, y, X, bw, kernel='bisquare', fixed=False",
#             "use_when": "Spatially varying relationships; coefficients change across geography",
#             "example": "from mgwr.gwr import GWR\nfrom mgwr.sel_bw import Sel_BW\nbw = Sel_BW(coords, y, X).search()\nmodel = GWR(coords, y, X, bw).fit()\nprint(model.summary())"
#         },
#         {
#             "name": "MGWR (Multiscale GWR)",
#             "import_path": "mgwr.gwr.MGWR",
#             "key_params": "coords, y, X, selector (from Sel_BW), kernel='bisquare', fixed=False",
#             "use_when": "Different variables operate at different spatial scales",
#             "example": "from mgwr.gwr import MGWR\nfrom mgwr.sel_bw import Sel_BW\nselector = Sel_BW(coords, y, X, multi=True)\nbws = selector.search()\nmodel = MGWR(coords, y, X, selector).fit()\nprint(model.summary())"
#         },
#     ],
#     "exploratory_spatial_analysis": [
#         {
#             "name": "Moran's I (Global)",
#             "import_path": "esda.Moran",
#             "key_params": "y, w, permutations=999",
#             "use_when": "Test for global spatial autocorrelation; is the variable spatially clustered?",
#             "example": "from esda import Moran\nimport libpysal\nw = libpysal.weights.Queen.from_dataframe(gdf)\nmi = Moran(gdf['variable'].values, w)\nprint(f'I={mi.I:.4f}, p={mi.p_sim:.4f}')"
#         },
#         {
#             "name": "Local Moran's I (LISA)",
#             "import_path": "esda.Moran_Local",
#             "key_params": "y, w, permutations=999",
#             "use_when": "Identify local clusters and outliers (HH, LL, HL, LH hotspots)",
#             "example": "from esda import Moran_Local\nlisa = Moran_Local(gdf['variable'].values, w)\ngdf['lisa_q'] = lisa.q  # quadrant labels\ngdf['lisa_p'] = lisa.p_sim"
#         },
#         {
#             "name": "Getis-Ord G* (Local)",
#             "import_path": "esda.getisord.G_Local",
#             "key_params": "y, w, transform='B', permutations=999, star=True",
#             "use_when": "Identify statistically significant hot spots and cold spots",
#             "example": "from esda.getisord import G_Local\ng_local = G_Local(gdf['variable'].values, w, transform='B', star=True)\ngdf['Gi_star'] = g_local.Zs\ngdf['Gi_p'] = g_local.p_sim"
#         },
#         {
#             "name": "Getis-Ord G (Global)",
#             "import_path": "esda.getisord.G",
#             "key_params": "y, w, permutations=999",
#             "use_when": "Test for global high/low clustering tendency",
#             "example": "from esda.getisord import G\ng = G(gdf['variable'].values, w)\nprint(f'G={g.G:.4f}, p={g.p_sim:.4f}')"
#         },
#         {
#             "name": "Spatial Weights (Queen)",
#             "import_path": "libpysal.weights.Queen",
#             "key_params": "from_dataframe(gdf)",
#             "use_when": "Polygon contiguity; neighbors share edges or vertices",
#             "example": "import libpysal\nw = libpysal.weights.Queen.from_dataframe(gdf)\nw.transform = 'r'  # row-standardize"
#         },
#         {
#             "name": "Spatial Weights (KNN)",
#             "import_path": "libpysal.weights.KNN",
#             "key_params": "from_dataframe(gdf, k=5)",
#             "use_when": "K nearest neighbors; works for points or polygons; fixed number of neighbors",
#             "example": "import libpysal\nw = libpysal.weights.KNN.from_dataframe(gdf, k=5)\nw.transform = 'r'"
#         },
#         {
#             "name": "Spatial Weights (Distance Band)",
#             "import_path": "libpysal.weights.DistanceBand",
#             "key_params": "from_dataframe(gdf, threshold=d)",
#             "use_when": "Neighbors within a distance threshold; good for point patterns",
#             "example": "import libpysal\nw = libpysal.weights.DistanceBand.from_dataframe(gdf, threshold=5000)\nw.transform = 'r'"
#         },
#     ],
#     "machine_learning": [
#         {
#             "name": "Random Forest Regressor",
#             "import_path": "sklearn.ensemble.RandomForestRegressor",
#             "key_params": "n_estimators=100, max_depth, random_state, min_samples_split",
#             "use_when": "Non-linear regression; feature importance; robust to outliers",
#             "example": "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\nimportances = model.feature_importances_"
#         },
#         {
#             "name": "Random Forest Classifier",
#             "import_path": "sklearn.ensemble.RandomForestClassifier",
#             "key_params": "n_estimators=100, max_depth, random_state, class_weight",
#             "use_when": "Classification tasks; feature importance for categorical outcomes",
#             "example": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)"
#         },
#         {
#             "name": "Gradient Boosting Regressor",
#             "import_path": "sklearn.ensemble.GradientBoostingRegressor",
#             "key_params": "n_estimators=100, learning_rate=0.1, max_depth=3",
#             "use_when": "High-accuracy regression; sequential error correction",
#             "example": "from sklearn.ensemble import GradientBoostingRegressor\nmodel = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)\nmodel.fit(X_train, y_train)"
#         },
#         {
#             "name": "K-Means Clustering",
#             "import_path": "sklearn.cluster.KMeans",
#             "key_params": "n_clusters, random_state, n_init='auto'",
#             "use_when": "Partition data into k groups; identify spatial or demographic clusters",
#             "example": "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')\ngdf['cluster'] = kmeans.fit_predict(X)"
#         },
#         {
#             "name": "DBSCAN",
#             "import_path": "sklearn.cluster.DBSCAN",
#             "key_params": "eps, min_samples",
#             "use_when": "Density-based clustering; finds clusters of arbitrary shape; detects outliers",
#             "example": "from sklearn.cluster import DBSCAN\ndb = DBSCAN(eps=0.5, min_samples=5)\ngdf['cluster'] = db.fit_predict(X)"
#         },
#         {
#             "name": "PCA (Principal Component Analysis)",
#             "import_path": "sklearn.decomposition.PCA",
#             "key_params": "n_components",
#             "use_when": "Dimensionality reduction; multicollinearity reduction; exploratory analysis",
#             "example": "from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\nX_scaled = StandardScaler().fit_transform(X)\npca = PCA(n_components=5)\nX_pca = pca.fit_transform(X_scaled)"
#         },
#         {
#             "name": "SVM (Support Vector Machine)",
#             "import_path": "sklearn.svm.SVR or sklearn.svm.SVC",
#             "key_params": "kernel='rbf', C=1.0, gamma='scale'",
#             "use_when": "Classification or regression with complex decision boundaries",
#             "example": "from sklearn.svm import SVR\nmodel = SVR(kernel='rbf', C=1.0)\nmodel.fit(X_train, y_train)"
#         },
#     ],
# }


def lookup_model_for_repair(failed_code: str, step_description: str) -> str:
    """Search the curated KB for models mentioned in the failed code or step description.
    Returns a focused reference block for ONLY the matching model(s), or empty string if none found."""
    text = (failed_code + " " + step_description).lower()

    # Build search aliases: map keywords to KB entries
    matches = []
    for category, models in CURATED_MODEL_KB.items():
        for m in models:
            # Check model name, import path, and common aliases
            name_lower = m["name"].lower()
            import_lower = m["import_path"].lower()
            # Extract short identifiers (e.g., "GWR", "OLS", "GM_Lag", "Moran")
            short_names = [
                name_lower,
                import_lower.split(".")[-1].lower(),  # e.g., "gwr", "ols", "gm_lag"
            ]
            if any(sn in text for sn in short_names):
                matches.append((category, m))

    if not matches:
        return ""

    lines = [
        "## Model Reference (from Knowledge Base)",
        "The following model documentation may help fix the error:\n"
    ]
    for category, m in matches:
        lines.append(f"**{m['name']}** (category: {category})")
        lines.append(f"  Import: `{m['import_path']}`")
        lines.append(f"  Params: {m['key_params']}")
        lines.append(f"  Use when: {m['use_when']}")
        lines.append(f"  Example:\n  ```python\n  {m['example']}\n  ```")
        lines.append("")

    return "\n".join(lines)


def format_curated_kb_for_prompt() -> str:
    """Format the curated model KB into a readable text block for the LLM prompt."""
    lines = [
        "## Model Knowledge Base",
        "Below is a curated reference of commonly used models. "
        "If the step already names a specific model, find it here and use the documented import path and parameters. "
        "If no model is named, select the most appropriate one from this KB. "
        "If nothing here fits the task, use your own knowledge.\n"
    ]

    for category, models in CURATED_MODEL_KB.items():
        cat_label = category.replace("_", " ").title()
        lines.append(f"### {cat_label}")
        for m in models:
            lines.append(f"**{m['name']}**")
            lines.append(f"  Import: `{m['import_path']}`")
            lines.append(f"  Params: {m['key_params']}")
            lines.append(f"  Use when: {m['use_when']}")
            lines.append(f"  Example:\n  ```python\n  {m['example']}\n  ```")
            lines.append("")
    return "\n".join(lines)