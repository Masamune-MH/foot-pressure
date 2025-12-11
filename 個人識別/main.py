import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt  # グラフ描画用にインポート
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 設定
# ==========================================
DATA_FOLDER = "../foot_pressure_data" # 前回の保存先フォルダ名に合わせました
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
    
    # ゼロ除算回避
    total_pressure[total_pressure == 0] = 1e-6
    
    cop_x = np.dot(pressure_values, coords[:, 0]) / total_pressure
    cop_y = np.dot(pressure_values, coords[:, 1]) / total_pressure
    return pd.DataFrame({'X': cop_x, 'Y': cop_y})

def calculate_sway_features(cop_df, time_data=None):
    """ 特徴量抽出 """
    x = cop_df['X'].values
    y = cop_df['Y'].values
    
    # 1. 総軌跡長 (Total Path Length)
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    # 2. 動揺範囲 (Range)
    ap_range = np.max(y) - np.min(y) # 前後
    ml_range = np.max(x) - np.min(x) # 左右
    
    # 3. 平均速度 (Mean Velocity)
    # Timeデータがあれば正確に計算、なければデータ点数で概算
    if time_data is not None and len(time_data) > 1:
        duration = time_data[-1] - time_data[0]
        if duration == 0: duration = 1 # エラー回避
        mean_velocity = path_length / duration
    else:
        mean_velocity = path_length / len(cop_df) # 簡易版

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    return np.array([path_length, ap_range, ml_range, mean_velocity, mean_x, mean_y])

def extract_features_from_file(filepath, window_size):
    """ 
    新しいCSV形式に対応したデータ読み込み関数
    Header: [Label, Time, Sensor1, ..., Sensor16]
    """
    try:
        # ヘッダーありで読み込む
        df = pd.read_csv(filepath) 
        
        # 必要な列があるかチェック
        if 'Label' not in df.columns or 'Sensor1' not in df.columns:
            print(f"スキップ: {filepath} (形式が違います)")
            return None, None

        user_name = str(df['Label'].iloc[0]) # 名前
        
        # センサーデータは "Sensor1" ～ "Sensor16" の列を取得
        # ilocを使う場合: 0:Label, 1:Time, 2~17:Sensors なので [:, 2:18]
        sensor_df = df.iloc[:, 2:18]
        
        # 時間データがあれば取得
        time_vals = df['Time'].values if 'Time' in df.columns else None

        if sensor_df.shape[1] < 16: return None, None

    except Exception as e:
        print(f"エラー ({filepath}): {e}")
        return None, None

    features_list = []
    df_cop = calculate_cop_trajectory(sensor_df.values)
    
    # データを可視化用に保存しておく（最初の1ファイルだけ表示などするため）
    # ここでは学習データ作成が主目的なので計算だけ進める

    step = window_size // 2
    
    for start_idx in range(0, len(df_cop) - window_size, step):
        end_idx = start_idx + window_size
        
        window_cop = df_cop.iloc[start_idx:end_idx, :]
        
        # ウィンドウ内のTimeデータ
        window_time = time_vals[start_idx:end_idx] if time_vals is not None else None
        
        feat = calculate_sway_features(window_cop, window_time)
        features_list.append(feat)
        
    return np.array(features_list), user_name

def visualize_trajectory(filepath):
    """ 指定したファイルの重心軌跡を描画するおまけ関数 """
    try:
        df = pd.read_csv(filepath)
        sensor_df = df.iloc[:, 2:18]
        df_cop = calculate_cop_trajectory(sensor_df.values)
        
        plt.figure(figsize=(6, 6))
        plt.plot(df_cop['X'], df_cop['Y'], alpha=0.7, label='COP Trace')
        plt.scatter(df_cop['X'].mean(), df_cop['Y'].mean(), color='red', label='Mean')
        plt.title(f"COP Trajectory: {df['Label'].iloc[0]}")
        plt.xlabel("X (Left-Right)")
        plt.ylabel("Y (Back-Front)")
        plt.grid(True)
        plt.legend()
        plt.axis('equal') # 縦横比を合わせる
        plt.show()
    except:
        pass

def main():
    # 1. データセットの作成
    X = [] 
    y = [] 
    
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    print(f"{len(csv_files)} 個のファイルを読み込み中...")

    if len(csv_files) == 0:
        print("エラー: CSVファイルが見つかりません。フォルダ名を確認してください。")
        return

    for filepath in csv_files:
        features, name = extract_features_from_file(filepath, WINDOW_SIZE)
        
        # おまけ: 1つ目のファイルだけグラフ表示してみる
        # if len(X) == 0: visualize_trajectory(filepath)

        if features is not None and len(features) > 0:
            for f in features:
                X.append(f)
                y.append(name)
            print(f"  - 読込完了: {name} ({len(features)}サンプル)")

    if len(X) == 0:
        print("有効なデータがありませんでした。")
        return

    X = np.array(X)
    y = np.array(y)
    
    print(f"\n全データ数: {len(X)}")
    
    # 2. 分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. 学習
    print("AIモデルを学習中...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. 評価
    print("評価中...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== 結果 ===")
    print(f"正解率 (Accuracy): {accuracy * 100:.2f}%")
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred))

    print("\n特徴量の重要度:")
    feature_names = ["軌跡長", "A-P幅(前後)", "M-L幅(左右)", "平均速度", "重心X平均", "重心Y平均"]
    importances = model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")

if __name__ == "__main__":
    main()