import pickle


class GestureClassifier:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.labels = data["label_names"]

    def predict(self, X, threshold=0.3):
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]

        idx = probs.argmax()
        confidence = probs[idx]

        if confidence < threshold:
            return "Unknown", confidence

        return self.labels[idx], confidence
