from sklearn import linear_model
from sklearn import ensemble

MODELS = {
    "lr": linear_model.LogisticRegression(),
    "rf": ensemble.RandomForestClassifier(n_estimators=5, n_jobs=-1, verbose=2)
}