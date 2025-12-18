import pandas as pd
import glob
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# センサ設定
# =========================
SENSOR_COLS = [f"Sensor{i}" for i in range(1, 17)]

REGIONS = {
    "forefoot": [1, 2, 3, 4, 9, 10, 11, 12],
    "midfoot":  [5, 6, 13, 14],
    "rearfoot": [7, 8, 15, 16],
}

# =========================
# 特徴量抽出（maxP）
# =========================
def extract_maxP(csv_path):
    df = pd.read_csv(csv_path)

    # ★ 255 = 最大足圧、0 = 最小足圧
    # → 反転処理は行わない

    features = {}
    for region, sensors in REGIONS.items():
        cols = [f"Sensor{i}" for i in sensors]
        # 各領域における最大足圧
        features[f"{region}_maxP"] = df[cols].max().max()

    return features

# =========================
# 学習用ラベル取得（ファイル名）
# =========================
def get_train_label(filename):
    if filename.endswith("_elderly.csv"):
        return 1  # 高齢者
    elif filename.endswith("_young.csv"):
        return 0  # 非高齢者
    else:
        raise ValueError(f"Unknown train filename: {filename}")

# =========================
# 学習データ構築
# =========================
def build_train_dataset(train_dir):
    X, y = [], []

    for file in glob.glob(os.path.join(train_dir, "*.csv")):
        X.append(extract_maxP(file))
        y.append(get_train_label(os.path.basename(file)))

    return pd.DataFrame(X), pd.Series(y)

# =========================
# テストデータ構築（判別専用）
# =========================
def build_test_dataset(test_dir):
    X, names = [], []

    for file in glob.glob(os.path.join(test_dir, "*.csv")):
        X.append(extract_maxP(file))
        names.append(os.path.basename(file))

    return pd.DataFrame(X), names

# =========================
# メイン処理
# =========================
if __name__ == "__main__":

    BASE_DIR = "foot_pressure_data"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    TEST_DIR  = os.path.join(BASE_DIR, "test")

    # ---------
    # 学習
    # ---------
    X_train, y_train = build_train_dataset(TRAIN_DIR)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    model.fit(X_train, y_train)

    # ---------
    # 判別（test）
    # ---------
    X_test, test_names = build_test_dataset(TEST_DIR)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- 判別結果 ---")
    for name, pred, prob in zip(test_names, y_pred, y_prob):
        label = "高齢者" if pred == 1 else "非高齢者"
        print(f"{name}: {label}（高齢者確率 = {prob:.2f}）")