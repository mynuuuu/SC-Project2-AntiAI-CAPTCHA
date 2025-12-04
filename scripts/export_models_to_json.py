# Code to export models to JSON
# Author: Naman Taneja

import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.tree import _tree

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
OUTPUT_FILE = MODELS_DIR / "slider_classifier_portable.json"

def tree_to_dict(tree, feature_names=None):
    """
    Convert a single sklearn tree to a dictionary
    """
    tree_ = tree.tree_
    if feature_names is None:
        feature_names_ = [f"feature_{i}" for i in range(tree_.n_features)]
    else:
        feature_names_ = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
    
    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names_[node]
            threshold = tree_.threshold[node]
            return {
                "type": "split",
                "feature_index": int(tree_.feature[node]),
                "threshold": float(threshold),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            value = tree_.value[node][0]
            probs = (value / value.sum()).tolist()
            return {
                "type": "leaf",
                "value": probs
            }
    
    return recurse(0)

def export_random_forest(model):
    """Export Random Forest model to dict"""
    trees = []
    for estimator in model.estimators_:
        trees.append(tree_to_dict(estimator))
    
    return {
        "type": "random_forest",
        "n_estimators": len(trees),
        "trees": trees,
        "classes": model.classes_.tolist() if hasattr(model, "classes_") else [0, 1]
    }

def export_gradient_boosting(model):
    """Export Gradient Boosting model to dict"""
    trees = []
    for i, estimator_array in enumerate(model.estimators_):
        for estimator in estimator_array:
            tree_dict = tree_to_dict(estimator)
            
            def fix_leaves(node):
                if node["type"] == "leaf":
                    node["value"] = node["value"][0]
                else:
                    fix_leaves(node["left"])
                    fix_leaves(node["right"])
            
            fix_leaves(tree_dict)
            trees.append(tree_dict)
            
    return {
        "type": "gradient_boosting",
        "n_estimators": len(trees),
        "learning_rate": model.learning_rate,
        "init_score": float(model.init_.predict(np.array([[0]*model.n_features_in_]))[0]) if hasattr(model.init_, 'predict') else 0.0,
        "trees": trees
    }

def export_scaler(scaler):
    """Export StandardScaler to dict"""
    return {
        "type": "standard_scaler",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }

def main():
    print(f"Loading models from {MODELS_DIR}...")
    
    try:
        ensemble_path = MODELS_DIR / "slider_classifier_ensemble.pkl"
        ensemble = joblib.load(ensemble_path)
        
        scaler_path = MODELS_DIR / "slider_classifier_scaler.pkl"
        scaler = joblib.load(scaler_path)
        
        print("Models loaded. Converting to JSON...")
        
        portable_model = {
            "scaler": export_scaler(scaler),
            "random_forest": export_random_forest(ensemble['random_forest']),
            "gradient_boosting": export_gradient_boosting(ensemble['gradient_boosting'])
        }
        
        print(f"Saving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(portable_model, f)
            
        print(f"Done! File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error exporting models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
