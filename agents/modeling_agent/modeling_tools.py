"""
Tool base for ModelingAgent.
Each tool is a callable function that performs a specific modeling task.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
# scikit-learn is imported inside the methods that fit and score models,
# so starting the app doesn't load it (and SciPy) up front.
import importlib


class ModelingTools:
    """Collection of tools for the ModelingAgent."""

    def __init__(self):
        self.fitted_models = {}  # Store fitted models by name
        self.results_cache = {}  # Store results for reuse

    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """
        Return tool definitions for LLM function calling.
        This follows the OpenAI function calling schema.
        """
        return [
            {
                "name": "select_models",
                "description": "Select the most appropriate statistical or machine learning models based on task description. Returns top 3 recommended models with their full import paths.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the modeling task (e.g., 'conduct paired t-tests', 'perform regression')"
                        },
                        "model_type": {
                            "type": "string",
                            "enum": ["classifier", "regressor", "cluster", "other"],
                            "description": "Optional filter for model type"
                        }
                    },
                    "required": ["task_description"]
                }
            },
            {
                "name": "fit_model",
                "description": "Fit a selected model to the provided data. Automatically handles train-test split and model instantiation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_class": {
                            "type": "string",
                            "description": "Full class path of the model (e.g., 'sklearn.linear_model.LinearRegression')"
                        },
                        "data_path": {
                            "type": "string",
                            "description": "Path to the dataset file (CSV, shapefile, etc.)"
                        },
                        "target_variable": {
                            "type": "string",
                            "description": "Name of the target/dependent variable column"
                        },
                        "predictor_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of predictor/independent variable column names"
                        },
                        "test_size": {
                            "type": "number",
                            "description": "Proportion of data for testing (default: 0.2)"
                        },
                        "model_params": {
                            "type": "object",
                            "description": "Optional parameters to pass to model constructor"
                        }
                    },
                    "required": ["model_class", "data_path", "target_variable", "predictor_variables"]
                }
            },
            {
                "name": "evaluate_model",
                "description": "Evaluate a fitted model's performance using appropriate metrics. Returns R², RMSE for regression, or accuracy/F1 for classification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the fitted model to evaluate"
                        }
                    },
                    "required": ["model_name"]
                }
            },
            {
                "name": "compare_models",
                "description": "Compare performance of multiple fitted models side-by-side using relevant metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of model names to compare"
                        }
                    },
                    "required": ["model_names"]
                }
            },
            {
                "name": "perform_statistical_test",
                "description": "Perform statistical tests (t-test, ANOVA, chi-square, etc.) on the data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_type": {
                            "type": "string",
                            "enum": ["t-test", "paired-t-test", "anova", "chi-square", "correlation"],
                            "description": "Type of statistical test to perform"
                        },
                        "data_path": {
                            "type": "string",
                            "description": "Path to the dataset file"
                        },
                        "variable1": {
                            "type": "string",
                            "description": "First variable for the test"
                        },
                        "variable2": {
                            "type": "string",
                            "description": "Second variable for the test (if applicable)"
                        },
                        "groups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Group variable for ANOVA or group comparison"
                        }
                    },
                    "required": ["test_type", "data_path", "variable1"]
                }
            },
            {
                "name": "get_model_summary",
                "description": "Get detailed summary statistics and coefficients from a fitted model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the fitted model"
                        }
                    },
                    "required": ["model_name"]
                }
            }
        ]

    def select_models(self, task_description: str, model_type: Optional[str] = None,
                     model_selector=None) -> Dict[str, Any]:
        """
        Select appropriate models using the model selector.

        Args:
            task_description: Description of the task
            model_type: Optional model type filter
            model_selector: Function to call for model selection

        Returns:
            Dictionary with recommended models
        """
        if not model_selector:
            return {"error": "Model selector not available"}

        models = model_selector(task_description, model_type, top_k=3)
        return {
            "recommended_models": models,
            "count": len(models),
            "task_description": task_description
        }

    def fit_model(self, model_class: str, data_path: str, target_variable: str,
                  predictor_variables: List[str], test_size: float = 0.2,
                  model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fit a model to data.

        Args:
            model_class: Full class path (e.g., 'sklearn.linear_model.LinearRegression')
            data_path: Path to dataset
            target_variable: Target variable name
            predictor_variables: List of predictor variable names
            test_size: Test set proportion
            model_params: Optional model parameters

        Returns:
            Dictionary with fitting results
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score
        try:
            # Load data
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith(('.shp', '.gpkg')):
                import geopandas as gpd
                data = gpd.read_file(data_path)
            else:
                return {"error": f"Unsupported file format: {data_path}"}

            # Prepare features and target
            X = data[predictor_variables].copy()
            y = data[target_variable].copy()

            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Import and instantiate model
            module_path, class_name = model_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)

            model_params = model_params or {}
            model = ModelClass(**model_params)

            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            is_classifier = hasattr(model, 'predict_proba')

            if is_classifier:
                train_score = accuracy_score(y_train, y_pred_train)
                test_score = accuracy_score(y_test, y_pred_test)
                metric_name = "accuracy"
            else:
                train_score = r2_score(y_train, y_pred_train)
                test_score = r2_score(y_test, y_pred_test)
                metric_name = "r2_score"

            # Store fitted model
            model_name = class_name
            self.fitted_models[model_name] = {
                "model": model,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test,
                "predictor_variables": predictor_variables,
                "target_variable": target_variable,
                "is_classifier": is_classifier
            }

            return {
                "success": True,
                "model_name": model_name,
                "model_class": model_class,
                f"train_{metric_name}": float(train_score),
                f"test_{metric_name}": float(test_score),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": len(predictor_variables),
                "features": predictor_variables,
                "target": target_variable
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate a fitted model."""
        from sklearn.metrics import (
            mean_squared_error, r2_score, accuracy_score,
            classification_report, confusion_matrix,
        )
        if model_name not in self.fitted_models:
            return {"error": f"Model '{model_name}' not found. Fit the model first."}

        try:
            model_data = self.fitted_models[model_name]
            model = model_data["model"]
            y_test = model_data["y_test"]
            y_pred_test = model_data["y_pred_test"]
            is_classifier = model_data["is_classifier"]

            if is_classifier:
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred_test)),
                    "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
                    "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist()
                }
            else:
                mse = mean_squared_error(y_test, y_pred_test)
                metrics = {
                    "r2_score": float(r2_score(y_test, y_pred_test)),
                    "rmse": float(np.sqrt(mse)),
                    "mse": float(mse)
                }

            return {
                "success": True,
                "model_name": model_name,
                "metrics": metrics,
                "n_test": len(y_test)
            }

        except Exception as e:
            return {"error": str(e), "success": False}

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple fitted models."""
        comparisons = []

        for model_name in model_names:
            if model_name in self.fitted_models:
                eval_result = self.evaluate_model(model_name)
                if eval_result.get("success"):
                    comparisons.append({
                        "model_name": model_name,
                        "metrics": eval_result["metrics"]
                    })

        if not comparisons:
            return {"error": "No valid models to compare"}

        return {
            "success": True,
            "comparisons": comparisons,
            "n_models": len(comparisons)
        }

    def perform_statistical_test(self, test_type: str, data_path: str,
                                 variable1: str, variable2: Optional[str] = None,
                                 groups: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform statistical test."""
        try:
            # Load data
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith(('.shp', '.gpkg')):
                import geopandas as gpd
                data = gpd.read_file(data_path)
            else:
                return {"error": f"Unsupported file format: {data_path}"}

            from scipy import stats

            if test_type == "t-test":
                if not variable2:
                    return {"error": "variable2 required for t-test"}

                group1 = data[variable1].dropna()
                group2 = data[variable2].dropna()
                t_stat, p_value = stats.ttest_ind(group1, group2)

                return {
                    "success": True,
                    "test_type": "t-test",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n1": len(group1),
                    "n2": len(group2)
                }

            elif test_type == "paired-t-test":
                if not variable2:
                    return {"error": "variable2 required for paired t-test"}

                var1 = data[variable1].dropna()
                var2 = data[variable2].dropna()

                # Align indices
                common_idx = var1.index.intersection(var2.index)
                var1 = var1.loc[common_idx]
                var2 = var2.loc[common_idx]

                t_stat, p_value = stats.ttest_rel(var1, var2)

                return {
                    "success": True,
                    "test_type": "paired-t-test",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n_pairs": len(var1)
                }

            elif test_type == "anova":
                if not groups:
                    return {"error": "groups required for ANOVA"}

                group_data = [data[data[groups[0]] == g][variable1].dropna() for g in data[groups[0]].unique()]
                f_stat, p_value = stats.f_oneway(*group_data)

                return {
                    "success": True,
                    "test_type": "anova",
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n_groups": len(group_data)
                }

            elif test_type == "correlation":
                if not variable2:
                    return {"error": "variable2 required for correlation"}

                var1 = data[variable1].dropna()
                var2 = data[variable2].dropna()

                # Align indices
                common_idx = var1.index.intersection(var2.index)
                var1 = var1.loc[common_idx]
                var2 = var2.loc[common_idx]

                corr_coef, p_value = stats.pearsonr(var1, var2)

                return {
                    "success": True,
                    "test_type": "correlation",
                    "correlation_coefficient": float(corr_coef),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "n": len(var1)
                }

            else:
                return {"error": f"Unknown test type: {test_type}"}

        except Exception as e:
            return {"error": str(e), "success": False}

    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model summary."""
        if model_name not in self.fitted_models:
            return {"error": f"Model '{model_name}' not found"}

        try:
            model_data = self.fitted_models[model_name]
            model = model_data["model"]

            summary = {
                "model_name": model_name,
                "n_features": len(model_data["predictor_variables"]),
                "features": model_data["predictor_variables"],
                "target": model_data["target_variable"],
                "n_train": len(model_data["X_train"]),
                "n_test": len(model_data["X_test"])
            }

            # Add coefficients if available (linear models)
            if hasattr(model, 'coef_'):
                coefficients = dict(zip(
                    model_data["predictor_variables"],
                    model.coef_.tolist() if hasattr(model.coef_, 'tolist') else [float(model.coef_)]
                ))
                summary["coefficients"] = coefficients

            if hasattr(model, 'intercept_'):
                summary["intercept"] = float(model.intercept_)

            # Add feature importances if available (tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(
                    model_data["predictor_variables"],
                    model.feature_importances_.tolist()
                ))
                summary["feature_importances"] = importances

            return {
                "success": True,
                **summary
            }

        except Exception as e:
            return {"error": str(e), "success": False}