import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
# 日本語フォント設定（環境によっては文字化けする可能性があります。その場合は 'sans-serif' などに変更してください）
plt.rcParams['font.family'] = 'sans-serif' 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 設定
# ==========================================
# フォルダ構成に合わせてパスを変更
BASE_FOLDER = "../foot_pressure_data"  
TRAIN_FOLDER = os.path.join(BASE_FOLDER, "train")
TEST_FOLDER = os.path.join(BASE_FOLDER, "test")

WINDOW_SIZE = 10

# --- センサ座標定義 ---
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

def calculate_sway_features(cop_df, time_data=None):
    """ 特徴量抽出 """
    x = cop_df['X'].values
    y = cop_df['Y'].values
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    ap_range = np.max(y) - np.min(y)
    ml_range = np.max(x) - np.min(x)
    
    if time_data is not None and len(time_data) > 1:
        duration = time_data[-1] - time_data[0]
        if duration == 0: duration = 1
        mean_velocity = path_length / duration
    else:
        mean_velocity = path_length / len(cop_df)

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return np.array([path_length, ap_range, ml_range, mean_velocity, mean_x, mean_y])

def extract_features_from_file(filepath, window_size):
    """ ファイル単体から特徴量を抽出 """
    try:
        df = pd.read_csv(filepath) 
        if 'Label' not in df.columns or 'Sensor1' not in df.columns:
            return None, None

        user_name = str(df['Label'].iloc[0])
        sensor_df = df.iloc[:, 2:18]
        time_vals = df['Time'].values if 'Time' in df.columns else None

        if sensor_df.shape[1] < 16: return None, None

    except Exception as e:
        print(f"エラー ({filepath}): {e}")
        return None, None

    features_list = []
    df_cop = calculate_cop_trajectory(sensor_df.values)
    step = window_size // 2
    
    for start_idx in range(0, len(df_cop) - window_size, step):
        end_idx = start_idx + window_size
        window_cop = df_cop.iloc[start_idx:end_idx, :]
        window_time = time_vals[start_idx:end_idx] if time_vals is not None else None
        feat = calculate_sway_features(window_cop, window_time)
        features_list.append(feat)
        
    return np.array(features_list), user_name

def load_dataset_from_folder(folder_path):
    """ 指定フォルダ内の全CSVから学習用データ(X, y)を作成 """
    X_list = []
    y_list = []
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"フォルダ読込中: {folder_path} ({len(csv_files)}ファイル)")
    
    for filepath in csv_files:
        features, name = extract_features_from_file(filepath, WINDOW_SIZE)
        if features is not None and len(features) > 0:
            for f in features:
                X_list.append(f)
                y_list.append(name)
    
    return np.array(X_list), np.array(y_list)

def main():
    # 1. データセットの読み込み (Train / Test を別々に読み込む)
    print("--- 学習データ(Train)の読み込み ---")
    X_train, y_train = load_dataset_from_folder(TRAIN_FOLDER)
    
    print("\n--- 検証データ(Test)の読み込み ---")
    X_test, y_test = load_dataset_from_folder(TEST_FOLDER)

    # データがあるか確認
    if len(X_train) == 0 or len(X_test) == 0:
        print("エラー: データが正しく読み込めませんでした。フォルダ構成を確認してください。")
        print(f"Train数: {len(X_train)}, Test数: {len(X_test)}")
        return

    print(f"\n学習データ数: {len(X_train)}")
    print(f"検証データ数: {len(X_test)}")

    # 2. 学習
    print("\nAIモデルを学習中...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. 評価
    print("評価中...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== 結果 ===")
    print(f"正解率 (Accuracy): {accuracy * 100:.2f}%")
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred))

    # 4. 混同行列（Confusion Matrix）の可視化
    print("混同行列を表示します...")
    
    # クラス名（ユーザー名）のリストを取得
    labels = model.classes_
    
    # 行列計算
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # グラフィカルに表示
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # 色使いを調整 (Bluesなどが見やすい)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    
    plt.title("Confusion Matrix (Prediction Result)")
    plt.tight_layout()
    plt.show()

    # 特徴量の重要度
    print("\n特徴量の重要度:")
    feature_names = ["軌跡長", "A-P幅(前後)", "M-L幅(左右)", "平均速度", "重心X平均", "重心Y平均"]
    importances = model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    main()