import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics.pairwise import euclidean_distances

# ==========================================
# 設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # main.pyの絶対パス
DATA_FOLDER = os.path.join(BASE_DIR, "../foot_pressure_data")# CSVが入っているフォルダ名
WINDOW_SIZE = 100             # 1歩とみなす行数（サンプリングレートに合わせて調整）

def extract_features_from_file(filepath, window_size):
    # CSVを読み込む（ヘッダー行あり）
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"エラー: {filepath} を読み込めませんでした。 {e}")
        return None, None

    # ラベル列を除去して数値データだけ残す
    df_values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # 1列目以降を数値変換

    # NaN を含む行は除外（足りない行があれば削除）
    df_values = df_values.dropna()

    if len(df_values) == 0:
        print(f"警告: {filepath} には有効な数値データがありません。")
        return None, None

    features_list = []
    
    # ウィンドウ単位で切り出し
    for start_idx in range(0, len(df_values), window_size):
        end_idx = start_idx + window_size
        if end_idx > len(df_values):
            break
        
        window = df_values.iloc[start_idx:end_idx, :]

        means = window.mean(axis=0).values
        maxs = window.max(axis=0).values
        stds = window.std(axis=0).values

        feature_vector = np.concatenate([means, maxs, stds])
        features_list.append(feature_vector)

    return np.array(features_list), df_values.shape[0]

def main():
    # ==========================================
    # 1. データの読み込みと特徴量抽出
    # ==========================================
    all_features = []
    labels = []
    
    # 指定フォルダ内の .csv ファイルをすべて取得
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    
    if not csv_files:
        print(f"エラー: '{DATA_FOLDER}' フォルダにCSVファイルが見つかりません。")
        # テスト用にダミーデータを作るならここですが、今回はファイルを想定
        return

    print(f"{len(csv_files)} 個のファイルが見つかりました。処理を開始します...")

    for filepath in csv_files:
        # ファイル名からユーザー名を取得 (例: "data/user_A.csv" -> "user_A")
        filename = os.path.basename(filepath)
        user_name = os.path.splitext(filename)[0]
        
        # 特徴量抽出を実行
        features, raw_rows = extract_features_from_file(filepath, WINDOW_SIZE)
        
        if features is not None and len(features) > 0:
            print(f"  - {user_name}: {raw_rows}行 -> {len(features)}歩分のデータを抽出")
            
            # 抽出したデータの数だけ、ラベル(名前)もリストに追加
            for f in features:
                all_features.append(f)
                labels.append(user_name)

    # 全員分のデータを1つの行列にする
    X = np.array(all_features)
    y = np.array(labels)
    
    print(f"\nデータセット構築完了: 全 {len(X)} サンプル")

    # ==========================================
    # 2. 識別実験 (ユークリッド距離)
    # ==========================================
    
    # --- シナリオ設定 ---
    # 0番目のデータ（最初の人）を「登録済みの本人」と仮定し、
    # それ以外の全データと比較して認証できるか試す
    
    target_index = 0
    enrolled_vector = X[target_index]
    enrolled_name = y[target_index]
    
    print(f"\n--- 認証テスト開始 ---")
    print(f"登録ユーザー: [{enrolled_name}] (サンプルID: {target_index})")
    
    # 判定の厳しさ（閾値）。データのスケールによって調整が必要
    # 1回実行してみて、本人の距離と他人の距離を見てから決めるのが良いです
    THRESHOLD = 3.0 
    
    print(f"閾値: {THRESHOLD}")
    print("-" * 60)
    print(f"{'比較対象':<15} | {'正解ラベル':<10} | {'距離':<10} | {'判定結果'}")
    print("-" * 60)

    # 全データと比較
    for i in range(len(X)):
        # 自分自身との比較はスキップ（距離0になるため）
        if i == target_index:
            continue
            
        test_vector = X[i]
        test_name = y[i]
        
        # ユークリッド距離計算
        dist = np.linalg.norm(enrolled_vector - test_vector)
        
        # 判定
        is_verified = dist <= THRESHOLD
        result_str = "Match (本人)" if is_verified else "Mismatch (他人)"
        
        # 正解ラベルと比較して、判定が合っていたかチェック
        is_actually_same_person = (enrolled_name == test_name)
        status_mark = "OK" if is_verified == is_actually_same_person else "NG"
        
        # 結果表示（全部出すと多いので、最初と、他人のデータの一部だけ表示）
        # ※実際の実験では全てログに保存したほうが良いです
        if test_name != enrolled_name and (i < 5 or i % 5 == 0): 
            print(f"{test_name:<15} | {str(is_actually_same_person):<10} | {dist:.4f}     | {result_str} ({status_mark})")

if __name__ == "__main__":
    main()