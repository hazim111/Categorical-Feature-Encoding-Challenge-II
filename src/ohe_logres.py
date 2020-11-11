import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

import argparse
import config 
import models_dispatcher

def run(fold,model):

    df = pd.read_csv(config.training_data_with_folds)

    features = [feature for feature in df.columns if feature not in ("id","target","kfold")]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features

    full_data = pd.concat([df_train[features], df_valid[features]],axis=0)

    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # modelling

    clf = models_dispatcher.MODELS[model]

    clf.fit(x_train,df_train.target.values)

    valid_preds = clf.predict_proba(x_valid)[:,1]

    # scoring
    auc = metrics.roc_auc_score(df_valid.target.values,valid_preds)
    print(f"FOLD={fold}, Accuracy = {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold",type=int)


    parser.add_argument("--model",type=str)

    args = parser.parse_args()

    run(fold= args.fold, model=args.model)
