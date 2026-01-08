import pandas as pd
import glob
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# センサー位置の定義（Y軸方向：つま先=0, かかと=3）
# =========================
Y_MAP = {
    0: [1, 2, 9, 10],   # つまさき
    1: [3, 4, 11, 12],  # 中足部前
    2: [5, 6, 13, 14],  # 中足部後
    3: [7, 8, 15, 16]   # かかと
}

def extract_advanced_features(csv_path):
    df = pd.read_csv(csv_path)
    # 安定した中間部分（20%〜80%）だけを抽出してノイズを除去
    df = df.iloc[int(len(df)*0.2) : int(len(df)*0.8)]
    
    sensor_cols = [f"Sensor{i}" for i in range(1, 17)]
    l_sensors = [f"Sensor{i}" for i in range(1, 9)]
    r_sensors = [f"Sensor{i}" for i in range(9, 17)]

    features_list = []
    
    for _, row in df.iterrows():
        total = row[sensor_cols].sum() + 1e-6
        if total < 100: continue # 荷重が弱すぎるフレームは無視

        # 1. 重心位置（COP_Y）の計算：つま先からかかとのどこに体重があるか
        weighted_y = 0
        for y_pos, sensors in Y_MAP.items():
            s_sum = row[[f"Sensor{s}" for s in sensors]].sum()
            weighted_y += s_sum * y_pos
        cop_y = weighted_y / total
        
        # 2. 左右バランス
        l_sum = row[l_sensors].sum()
        balance_l = l_sum / total
        
        features_list.append({
            "cop_y": cop_y,
            "balance_l": balance_l,
            "total_p": total / 4000 # 全体の重さ（正規化）
        })
    
    feat_df = pd.DataFrame(features_list)
    
    # ファイル全体の統計量を特徴量にする
    res = {
        "cop_y_mean": feat_df["cop_y"].mean(),
        "cop_y_std":  feat_df["cop_y"].std(),   # 重心の揺れ（重要！）
        "balance_mean": feat_df["balance_l"].mean(),
        "total_std": feat_df["total_p"].std()  # 踏ん張りの変動
    }
    return res

# --- 学習・実行部分 ---
if __name__ == "__main__":
    BASE_DIR = "foot_pressure_data"
    TRAIN_DIR = os.path.join(BASE_DIR, "train_defined")
    TEST_DIR  = os.path.join(BASE_DIR, "test_defined")

    # 学習データの構築
    X_train, y_train = [], []
    for file in glob.glob(os.path.join(TRAIN_DIR, "*.csv")):
        X_train.append(extract_advanced_features(file))
        y_train.append(1 if "_elderly.csv" in file else 0)
    
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train)

    # シンプルなモデルで過学習を防ぐ
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.5)) # 正則化を強めて安定させる
    ])
    model.fit(X_train, y_train)

    # 判別
    print("\n--- 改善版：重心分析による判別結果 ---")
    correct = 0
    test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))
    
    for file in test_files:
        name = os.path.basename(file)
        feat = pd.DataFrame([extract_advanced_features(file)])
        prob = model.predict_proba(feat)[0, 1]
        pred = 1 if prob > 0.5 else 0
        
        true_label = 1 if "_elderly.csv" in name else 0
        label_str = "高齢者" if pred == 1 else "非高齢者"
        result = "【正解】" if pred == true_label else "【不正解】"
        
        print(f"{name:<25}: {label_str} (高齢者確率: {prob:.2f}) {result}")
        if pred == true_label: correct += 1

    print(f"\n正解率: {(correct/len(test_files))*100:.1f}% ({correct}/{len(test_files)})")