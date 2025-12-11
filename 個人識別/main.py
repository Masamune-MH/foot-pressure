import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 設定
# ==========================================
DATA_FOLDER = "data_folder"
WINDOW_SIZE = 100

# --- センサ座標定義 (そのまま) ---
SENSOR_COORDS = {
    1: (-2, 3), 2: (-4, 3), 3: (-2, 2), 4: (-4, 2),
    5: (-2, 1), 6: (-4, 1), 7: (-2, 0), 8: (-4, 0),
    9:  (2, 3), 10: (4, 3), 11: (2, 2), 12: (4, 2),
    13: (2, 1), 14: (4, 1), 15: (2, 0), 16: (4, 0)
}

def calculate_cop_trajectory(sensor_data_values):
    """ 重心(COP)計算 """
    coords = np.array([SENSOR_COORDS[i+1] for i in range(16)])
    pressure_values = sensor_data_values.astype(float)
    total_pressure = np.sum(pressure_values, axis=1)
    total_pressure[total_pressure == 0] = 1e-6
    cop_x = np.dot(pressure_values, coords[:, 0]) / total_pressure
    cop_y = np.dot(pressure_values, coords[:, 1]) / total_pressure
    return pd.DataFrame({'X': cop_x, 'Y': cop_y})

def calculate_sway_features(cop_df):
    """ 特徴量抽出 (論文準拠) """
    x = cop_df['X'].values
    y = cop_df['Y'].values
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    ap_range = np.max(y) - np.min(y)
    ml_range = np.max(x) - np.min(x)
    mean_velocity = path_length / len(cop_df)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.array([path_length, ap_range, ml_range, mean_velocity, mean_x, mean_y])

def extract_features_from_file(filepath, window_size):
    """ ファイルから学習用データセットを作る """
    try:
        df = pd.read_csv(filepath, header=None)
        user_name = str(df.iloc[0, 0]) # 1列目の名前
        sensor_df = df.iloc[:, 1:17]   # データ部分
        
        if sensor_df.shape[1] < 16: return None, None
    except:
        return None, None

    features_list = []
    df_cop = calculate_cop_trajectory(sensor_df.values)
    
    # スライディングウィンドウ (学習データを増やすために少しずつずらすのもアリ)
    step = window_size // 2  # 50%ずつ重ねてデータを2倍に増やすテクニック
    
    for start_idx in range(0, len(df_cop) - window_size, step):
        end_idx = start_idx + window_size
        window_cop = df_cop.iloc[start_idx:end_idx, :]
        features_list.append(calculate_sway_features(window_cop))
        
    return np.array(features_list), user_name

def main():
    # 1. データセットの作成
    X = [] # 特徴量 (問題)
    y = [] # ラベル (正解: 名前)
    
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    print(f"{len(csv_files)} 個のファイルを読み込み中...")

    for filepath in csv_files:
        features, name = extract_features_from_file(filepath, WINDOW_SIZE)
        if features is not None and len(features) > 0:
            for f in features:
                X.append(f)
                y.append(name)
            print(f"  - 読込完了: {name} ({len(features)}サンプル)")

    X = np.array(X)
    y = np.array(y)
    
    print(f"\n全データ数: {len(X)}")
    
    # 2. 訓練データとテストデータに分割 (8:2)
    # random_state=42 は毎回同じ分け方にするおまじない
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"学習用データ: {len(X_train)}件")
    print(f"テスト用データ: {len(X_test)}件")

    # 3. AIモデルの構築と学習 (ランダムフォレスト)
    print("\nAIモデルを学習中...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train) # <--- ここで学習！

    # 4. 予測と評価
    print("テストデータで性能評価中...")
    y_pred = model.predict(X_test) # テストデータで予測させる

    # 正解率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== 結果 ===")
    print(f"正解率 (Accuracy): {accuracy * 100:.2f}%")
    
    # 誰を誰と間違えたかの詳細
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred))

    # 特徴量の重要度 (AIがどこを見て判断したか)
    print("\n特徴量の重要度 (どの指標が効いたか):")
    feature_names = ["軌跡長", "A-P幅", "M-L幅", "速度", "重心X", "重心Y"]
    importances = model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    main()