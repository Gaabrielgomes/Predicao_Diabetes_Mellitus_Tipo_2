
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Acuracia": accuracy_score(y_test, pred),
        "Precisao": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba) if proba is not None else None
    }
    return metrics, confusion_matrix(y_test, pred), pred, proba

def train_models(X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    metrics_lr, cm_lr, pred_lr, proba_lr = evaluate(lr, X_train_scaled, X_test_scaled, y_train, y_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    metrics_rf, cm_rf, pred_rf, proba_rf = evaluate(rf, X_train, X_test, y_train, y_test)

    return (lr, metrics_lr, cm_lr, pred_lr, proba_lr,
            rf, metrics_rf, cm_rf, pred_rf, proba_rf)
