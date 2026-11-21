import inspect, json
from sklearn.utils import all_estimators

def build_model_base():
    model_docs = []
    for name, cls in all_estimators():
        doc = inspect.getdoc(cls) or ""
        entry = {
            "model_name": name,
            "full_class": f"{cls.__module__}.{name}",
            "type": (
                "classifier" if "Classifier" in name.lower()
                else "regressor" if "regressor" in name.lower()
                else "cluster" if "cluster" in name.lower()
                else "other"
            ),
            "doc": doc[:2000],  # trim for embedding efficiency
        }
        model_docs.append(entry)

    with open("model_docs.json", "w") as f:
        json.dump(model_docs, f, indent=2)

    print(f"✅ Saved {len(model_docs)} model docs.")
    return model_docs

# if __name__ == "__main__":
build_model_base()