import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
import numpy as np

import argparse
import joblib

import config

import models_dispatcher

def predict(model):
    df = pd.read_csv(config.test_data_loc)
    test_idx = df["id"].values
    predictions = None

    for fold in range(5):
        df = pd.read_csv(config.test_data_loc)

        
        encoders = joblib.load(os.path.join(config.models_location, f"{model}_{fold}_label_encoder.bin"))
        cols = joblib.load(os.path.join(config.models_location, f"{model}_{fold}_columns.bin"))
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        
        clf = joblib.load(os.path.join(config.models_location, f"{model}_{fold}.bin"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if fold == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    submission = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"{model}.csv", index=False)
    return submission
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    predict(model=args.model)
