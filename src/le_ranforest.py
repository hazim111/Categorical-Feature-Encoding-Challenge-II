import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

import argparse
import config 
import models_dispatcher
import joblib

def run(fold,model):

    df = pd.read_csv(config.training_data_with_folds)
    df_test = pd.read_csv(config.test_data_loc)

    features = [feature for feature in df.columns if feature not in ("id","target","kfold")]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")
        df_test[col] = df_test[col].astype(str).fillna("NONE")


    
    label_encoders = {}
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col].values.tolist()+ df_test[col].values.tolist())
        df[col] = lbl.transform(df[col].values.tolist())
        label_encoders[col] = lbl

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # modelling

    clf = models_dispatcher.MODELS[model]

    clf.fit(x_train,df_train.target.values)

    valid_preds = clf.predict_proba(x_valid)[:,1]

    # scoring
    auc = metrics.roc_auc_score(df_valid.target.values,valid_preds)
    print(f"FOLD={fold}, Accuracy = {auc}")

    joblib.dump(clf, os.path.join(config.models_location,f"{model}_{fold}.bin"))
    joblib.dump(label_encoders, os.path.join(config.models_location,f"{model}_{fold}_label_encoder.bin"))
    joblib.dump(features, os.path.join(config.models_location, f"{model}_{fold}_columns.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",type=int)


    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    run(fold= args.fold, model=args.model)