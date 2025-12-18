import pandas as pd
import numpy as np
import glob
import os
import joblib  # 知能の保管用
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 設定エリア
# ==========================================
BASE_FOLDER = "../foot_pressure_data"
TRAIN_FOLDER = os.path.join(BASE_FOLDER, "train") # 学習用データ (ファイル名に young/elderly を含める)
TEST_FOLDER = os.path.join(BASE_FOLDER, "test")   # 判定したいデータ

DB_FILE = "foot_feature_database.csv"  # 特徴量データベース（CSV）
MODEL_FILE = "foot_model.joblib"       # AIモデル（知能）

MAX_SENSOR_VALUE = 255

# センサの場所定義
IDX_FOREFOOT = [0, 1, 2, 3, 8, 9, 10, 11]
IDX_MIDFOOT  = [4, 5, 12, 13]
IDX_REARFOOT = [6, 7, 14, 15]

def calculate_pressure_features(raw_values):
    """ センサ値を反転し、部位ごとの比率を計算（資料に基づき中足部と後足を重視） """
    pressure_values = np.maximum(MAX_SENSOR_VALUE - raw_values, 0)
    total = np.sum(pressure_values) + 1e-6
    
    if total < 50: return None
    
    # 部位別の比率を計算
    ratio_ff = np.sum(pressure_values[IDX_FOREFOOT]) / total
    ratio_mf = np.sum(pressure_values[IDX_MIDFOOT]) / total
    ratio_rf = np.sum(pressure_values[IDX_REARFOOT]) / total
    
    # AIに渡す特徴量リスト
    return [ratio_ff, ratio_mf, ratio_rf]

def extract_features_from_file(filepath):
    """ ファイルから全フレームの特徴量を抽出 """
    try:
        df = pd.read_csv(filepath)
        # ※csvの形式に合わせて列番号は調整してください（現在は2列目以降を使用）
        sensor_df = df.iloc[:, 2:18] 
        raw_data = sensor_df.values.astype(float)
        
        feats = []
        for row in raw_data:
            f = calculate_pressure_features(row)
            if f: feats.append(f)
        return feats
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# ==========================================
# 1. 学習とデータベース保管（ファイル名判定版）
# ==========================================
def learn_and_store():
    print("=== 学習モード: データベースを更新中 ===")
    
    if not os.path.exists(TRAIN_FOLDER):
        print(f"エラー: 学習用フォルダが見つかりません: {TRAIN_FOLDER}")
        return

    all_features = []
    csv_files = glob.glob(os.path.join(TRAIN_FOLDER, "*.csv"))
    
    if not csv_files:
        print("エラー: 学習用CSVファイルが見つかりません。")
        return

    for f in csv_files:
        filename = os.path.basename(f).lower() # 小文字にして判定しやすくする
        
        # ▼ ファイル名からラベルを決定 ▼
        if "young" in filename:
            label = "若者"
        elif "elderly" in filename:
            label = "高齢者"
        else:
            print(f"スキップ: ファイル名に 'young' も 'elderly' も含まれていません -> {filename}")
            continue
        
        # 特徴量抽出
        feats = extract_features_from_file(f)
        if feats:
            for ft in feats:
                all_features.append(ft + [label])
            print(f"読込完了: {filename} ({label}) - {len(feats)}フレーム")

    if not all_features:
        print("学習可能なデータがありませんでした。ファイル名を確認してください。")
        return

    # 特徴量をCSVデータベースとして保管
    new_df = pd.DataFrame(all_features, columns=["FF", "MF", "RF", "Label"])
    new_df.to_csv(DB_FILE, index=False, encoding="utf-8-sig")
    
    # AI（モデル）の学習と保管
    X = new_df.drop("Label", axis=1)
    y = new_df["Label"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    
    print("-" * 30)
    print(f"完了: {len(new_df)}件のデータを学習し、モデルを保存しました。")

# ==========================================
# 2. 判定（テストデータと比較）
# ==========================================
def predict_test_data():
    if not os.path.exists(MODEL_FILE):
        print("エラー: 学習済みモデルがありません。先に学習を実行してください。")
        return
    
    if not os.path.exists(TEST_FOLDER):
        print(f"エラー: テスト用フォルダが見つかりません: {TEST_FOLDER}")
        return

    print("=== 判定モード: データ判定中 ===")
    model = joblib.load(MODEL_FILE)
    test_files = glob.glob(os.path.join(TEST_FOLDER, "*.csv"))
    
    print(f"{'ファイル名':<30} | {'判定結果'}")
    print("-" * 50)
    
    for f in test_files:
        feats = extract_features_from_file(f)
        if feats:
            # 各フレームの予測を行い、多数決で決める
            preds = model.predict(feats)
            if len(preds) > 0:
                final_res = pd.Series(preds).mode()[0]
                print(f"{os.path.basename(f):<30} | {final_res}")
            else:
                print(f"{os.path.basename(f):<30} | 判定不可（有効データなし）")

if __name__ == "__main__":
    # 実行したい方のコメントアウトを外してください
    
    # 1. データを学習させる時
    learn_and_store()
    
    # 2. 新しいデータを判定する時
    predict_test_data()