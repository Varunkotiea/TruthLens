import os
import argparse
import time
import warnings
import random
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Project imports
from utils.preprocess import preprocess_batch

warnings.filterwarnings("ignore")

# Global Configuration Paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(csv_path: str, label_real: str = "REAL", label_fake: str = "FAKE") -> pd.DataFrame:
    """Loads, validates, and standardizes the input dataset CSV."""
    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Found: {list(df.columns)}")

    if "title" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["content"] = df["text"].fillna("")

    df["label_raw"] = df["label"].astype(str).str.strip().str.upper()

    label_map = {
        label_real.upper(): 1,
        label_fake.upper(): 0,
        "1": 1, "0": 0,
        "TRUE": 1, "FALSE": 0,
    }
    df["label_bin"] = df["label_raw"].map(label_map)

    bad = df["label_bin"].isna().sum()
    if bad > 0:
        print(f"⚠️  Dropping {bad} rows with unrecognised labels.")
        df = df.dropna(subset=["label_bin"])

    df["label_bin"] = df["label_bin"].astype(int)
    df = df[df["content"].str.strip().ne("")]
    df = df.dropna(subset=["content"])

    print(f"   Clean shape: {df.shape}")
    print(f"   REAL: {(df['label_bin']==1).sum()} | FAKE: {(df['label_bin']==0).sum()}")

    return df[["content", "label_bin"]].reset_index(drop=True)


def train(csv_path: str, label_real: str = "REAL", label_fake: str = "FAKE"):
    """Executes the full training, evaluation, and artifact saving pipeline."""
    df = load_dataset(csv_path, label_real, label_fake)

    print("\n🔄 Preprocessing text...")
    t0 = time.time()
    X_raw = preprocess_batch(df["content"].tolist())
    y = df["label_bin"].values
    print(f"   Done in {time.time()-t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Split: {len(X_train)} train | {len(X_test)} test")

    print("\n🔢 Fitting TF-IDF Vectorizer…")
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")

    print("\n🤖 Training classifiers…")
    models = {
        "Logistic Regression": LogisticRegression(
            C=5.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
        "Passive Aggressive": PassiveAggressiveClassifier(
            C=0.5, max_iter=1000, random_state=42,
            class_weight="balanced",
        ),
    }

    best_model = None
    best_acc = 0.0
    best_name = ""

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"   {name}: accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_name = name

    print(f"\n🏆 Best model: {best_name} (accuracy={best_acc:.4f})")

    y_pred = best_model.predict(X_test_vec)
    y_prob = (
        best_model.predict_proba(X_test_vec)[:, 1]
        if hasattr(best_model, "predict_proba")
        else None
    )

    print("\n" + "="*50 + "\nCLASSIFICATION REPORT\n" + "="*50)
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    print("CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    if y_prob is not None:
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    print(f"\n💾 Saving artifacts to {MODEL_DIR}/")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "model_name": best_name,
        "accuracy": round(best_acc, 4),
        "vocab_size": len(vectorizer.vocabulary_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "label_map": {1: label_real.upper(), 0: label_fake.upper()},
        "ngram_range": "(1, 2)",
        "max_features": 100_000,
    }
    joblib.dump(metadata, METADATA_PATH)

    print("\n✅ Training complete! Run the app with: streamlit run app.py\n")
    return best_model, vectorizer


def create_demo_model():
    """Generates a fall-back baseline model using pre-defined local text sequences."""
    print("\n⚡ Creating baseline demo model…")

    real_samples = [
        "scientists at oxford university published peer reviewed study nature medicine confirming vaccine ninety five percent effective phase three clinical trials",
        "researchers from harvard medical school found regular physical exercise reduces risk type two diabetes forty percent seven year longitudinal study",
        "world health organization released updated guidelines pandemic prevention recommending booster doses high risk populations based clinical evidence",
        "new peer reviewed research journal lancet shows aspirin reduces cardiovascular disease risk among adults over fifty years old",
        "university hospital clinical trial results published new england journal medicine show immunotherapy treatment improves cancer survival rates significantly",
        "federal reserve raised interest rates twenty five basis points unanimous decision federal open market committee bringing benchmark rate five percent",
        "apple reported quarterly revenue eighty nine billion dollars beating analyst estimates driven strong iphone sales services segment growth",
        "bureau labor statistics released jobs report showing economy added two hundred thousand positions unemployment rate remains four percent",
        "senate passed bipartisan infrastructure bill sixty seven votes including funding roads bridges broadband internet clean energy projects",
        "supreme court ruled five four decision landmark case constitutional rights privacy digital communications warrant requirements",
        "google announced new artificial intelligence model outperforms previous benchmarks natural language processing image recognition tasks",
        "nasa successfully launched james webb space telescope replacement hubble providing unprecedented infrared imaging deep universe",
        "according official spokesperson department confirmed policy change following internal review recommendations compliance",
        "data released government agency shows significant improvement key economic indicators compared previous quarter year period",
        "central bank minutes released showing policymakers debated pace interest rate increases inflation economic growth outlook"
    ]

    fake_samples = [
        "doctors silenced big pharma discovered drinking bleach mixed lemon juice cures cancer three days government hiding miracle remedy decades",
        "shocking vaccine contains microsoft microchip bill gates track location control thoughts using five g tower signals death threats silence",
        "urgent warning covid vaccine depopulation agenda hidden dose thousands funeral directors finding strange metallic objects vaccinated corpses",
        "miracle cure suppressed big pharma natural remedy heals everything doctors dont want know share before deleted truth hidden decades",
        "george soros funding secret underground network antifa terrorists destroy america install one world government documents confirm millions",
        "deep state globalist plot destroying country patriots fight back constitution abolished democrat party complicit agenda share truth now",
        "breaking bombshell election stolen rigged dead voters dominion machines hacked proof suppressed mainstream media deep state coverup",
        "wake up sheeple mainstream media hiding truth share viral before deleted censored banned shadow truth revolution coming",
        "shocking truth exposed exclusive bombshell revelation liberal agenda lies propaganda machine dying alternative media rises patriots",
        "chemtrails poison spraying sky government secret program population reduction depopulation agenda geoengineering admit truth finally",
        "this will blow your mind share everyone knows truth finally out mainstream media will never show you this breaking exclusive",
        "they dont want you know this share before deleted banned censored suppressed truth coming out millions awakening happening now",
        "elite globalists terrified people learning truth thats why censoring banning silencing voices patriots alternative media rising",
        "breaking exclusive whistleblower leaks documents prove conspiracy cover up decades finally exposed share widely spread truth now",
        "hundreds thousands dead covered mainstream media social media censorship fact checkers paid silence truth alternative sources only"
    ]

    random.seed(42)
    aug_real, aug_fake = [], []

    for s in real_samples:
        aug_real.append(s)
        words = s.split()
        if len(words) > 6:
            random.shuffle(words[2:])
        aug_real.append(" ".join(words))

    for s in fake_samples:
        aug_fake.append(s)
        words = s.split()
        if len(words) > 6:
            random.shuffle(words[2:])
        aug_fake.append(" ".join(words))

    texts  = aug_real * 3 + aug_fake * 3
    labels = [1] * len(aug_real) * 3 + [0] * len(aug_fake) * 3

    print(f"   Training on {len(texts)} augmented sample variants.")
    X = preprocess_batch(texts)

    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_te_vec = vectorizer.transform(X_te)

    model = LogisticRegression(C=3.0, max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_tr_vec, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te_vec))
    print(f"   Demo accuracy on validation split: {acc*100:.1f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "model_name": "Logistic Regression (Demo Model)",
        "accuracy": round(acc, 4),
        "vocab_size": len(vectorizer.vocabulary_),
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "label_map": {1: "REAL", 0: "FAKE"},
        "ngram_range": "(1, 2)",
        "max_features": 20_000,
        "is_demo": True,
    }
    joblib.dump(metadata, METADATA_PATH)

    print("✅ Baseline fallback model saved successfully.\n")
    return model, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fake News Detection model")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--label-true", type=str, default="REAL", help="Value for REAL news in label column")
    parser.add_argument("--label-fake", type=str, default="FAKE", help="Value for FAKE news in label column")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        train(args.data, args.label_true, args.label_fake)
    else:
        if args.data:
            print(f"⚠️  Dataset not found at '{args.data}'. Generating baseline model alternative...")
        create_demo_model()