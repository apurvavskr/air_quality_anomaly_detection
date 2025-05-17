import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def clean_data(df):
    df = df.copy()
    print("Original:", len(df))

    # Drop unnamed or fully empty columns
    df.drop(columns=[col for col in df.columns if "Unnamed" in col or df[col].isna().all()], inplace=True)
    print("After dropping unnamed/empty columns:", len(df))

    df.columns = df.columns.str.strip()

    # Replace -200 with NaN
    df.replace(-200, pd.NA, inplace=True)
    print("After replacing -200 with NaN:", len(df))

    # Fix time format
    df['Time'] = df['Time'].astype(str).str.replace(".", ":", regex=False)

    # Convert all except 'Date' and 'Time' to numeric
    #for col in df.columns:
    #    if col not in ["Date", "Time"]:
    #        df[col] = pd.to_numeric(df[col], errors='coerce')
    #print("After converting numeric columns:", len(df))

    # Drop high-missing column
    df.drop(columns=["NMHC(GT)"], inplace=True, errors="ignore")
    print("After dropping NMHC(GT):", len(df))

    # Drop rows with more than 5 missing values
    max_missing_allowed = 5
    df.dropna(thresh=(df.shape[1] - max_missing_allowed), inplace=True)
    print("After dropping rows with too many NaNs:", len(df))

    # Impute the remaining missing values with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print("After imputing remaining NaNs:", len(df))

    return df

def extract_time_features(df):
    out = df.copy()
    if "Date" in out.columns and "Time" in out.columns:
        out["Time"] = out["Time"].astype(str).str.replace(".", ":", regex=False)
        out["Datetime"] = pd.to_datetime(out["Date"] + " " + out["Time"], dayfirst=True, errors='coerce')
        out["Hour"] = out["Datetime"].dt.hour
        out["Month"] = out["Datetime"].dt.month
    return out


def scale_features(df):
    numeric_cols = df.select_dtypes(include="number").columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)
    return scaled_df


def run_pca(scaled_data, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(scaled_data)
    return X_pca, pca.explained_variance_ratio_

def run_isolation_forest(scaled_data, contamination=0.05):
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(scaled_data)
    scores = iso.decision_function(scaled_data)  # anomaly scores
    return scores, labels, iso