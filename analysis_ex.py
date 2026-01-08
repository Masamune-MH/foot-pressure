import pandas as pd
import glob
import os
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# 設定エリア
# =========================
BASE_DIR = "./foot_pressure_data" # 実際のパスに合わせてください
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# センサ設定
REGIONS = {
    "forefoot": [1, 2, 3, 4, 9, 10, 11, 12],
    "midfoot":  [5, 6, 13, 14],
    "rearfoot": [7, 8, 15, 16],
}

# =========================
# 特徴量抽出関数
# =========================
def extract_maxP(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # カラム名チェック
        if 'Sensor1' in df.columns:
            # そのまま
            pass
        else:
            # 2列目以降をセンサデータとみなす場合の処理など必要に応じて追加
            pass

        features = {}
        for region, sensors in REGIONS.items():
            cols = [f"Sensor{i}" for i in sensors if f"Sensor{i}" in df.columns]
            if not cols:
                features[f"{region}_maxP"] = 0
                continue
            # 各領域における最大足圧
            features[f"{region}_maxP"] = df[cols].max().max()
        return features
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def get_train_label(filename):
    # ファイル名からラベルを判定 (ルールに合わせて変更してください)
    filename = filename.lower()
    if "elderly" in filename or "senior" in filename:
        return 1  # 高齢者
    elif "young" in filename or "normal" in filename:
        return 0  # 非高齢者
    return None

def build_dataset(folder_path, is_train=True):
    X = []
    y = [] # trainのみ
    names = [] # testのみ

    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        print(f"警告: {folder_path} にCSVファイルがありません。")
        return pd.DataFrame(), [], []

    for file in files:
        feats = extract_maxP(file)
        if feats is None: continue

        if is_train:
            label = get_train_label(os.path.basename(file))
            if label is not None:
                X.append(feats)
                y.append(label)
        else:
            X.append(feats)
            names.append(os.path.basename(file))
            
    if is_train:
        return pd.DataFrame(X), pd.Series(y), []
    else:
        return pd.DataFrame(X), [], names

# =========================
# メイン処理
# =========================
def main():
    # 1. 訓練データを全部読み込む
    print("--- データの読み込み ---")
    X_all, y_all, _ = build_dataset(TRAIN_DIR, is_train=True)
    
    if len(X_all) < 5:
        print("エラー: 訓練データが少なすぎます（最低でも5つ以上必要です）")
        return

    # 2. 訓練データ(Train)と検証データ(Val)に分割する
    # test_size=0.3 は「30%を検証用に回す」という意味
    # random_state=42 は「毎回同じ分け方をする」ための固定値
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    print(f"データ総数: {len(X_all)}件")
    print(f" -> 学習用(Train): {len(X_train)}件")
    print(f" -> 検証用(Val)  : {len(X_val)}件 (これを仮のテストデータとして使います)")

    # 3. データの正規化 (スケーリング)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) # 検証用も同じ基準で変換

    # 4. パラメータ調整ループ
    # 試したいパラメータの組み合わせを定義
    param_candidates = {
        'C': [0.01, 0.1, 1, 10, 100],            # 正則化の強さ
        'class_weight': [None, 'balanced'],      # データの偏り補正
        'solver': ['liblinear', 'lbfgs']         # 計算方法
    }

    best_score = 0
    best_params = {}
    best_model = None

    print("\n--- パラメータ調整開始 ---")
    # 全組み合わせを総当たり
    keys, values = zip(*param_candidates.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        
        # モデル作成・学習
        try:
            model = LogisticRegression(random_state=42, **params)
            model.fit(X_train_scaled, y_train)
            
            # 検証データでテスト
            val_pred = model.predict(X_val_scaled)
            score = accuracy_score(y_val, val_pred)
            
            # 結果表示（長くなるので良い結果だけ詳しく出すなどの調整も可）
            # print(f"Params: {params} -> Val Accuracy: {score:.2f}")

            # ベストスコア更新チェック
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model # とりあえず保持
        except Exception as e:
            continue

    print(f"\n★ 最適パラメータが見つかりました: {best_params}")
    print(f"★ 検証データでの正解率: {best_score*100:.1f}%")

    # 5. 最適パラメータを使って、訓練データ全体で再学習する
    # (検証用データも無駄にせず、最後は全てのデータで学習させて賢くする)
    print("\n--- 全データでの再学習 & 本番テスト ---")
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all) # 全データでスケーラー作り直し
    
    final_model = LogisticRegression(random_state=42, **best_params)
    final_model.fit(X_all_scaled, y_all)
    
    # 重要な特徴量（係数）の確認
    coefs = final_model.coef_[0]
    feature_names = X_all.columns
    print("【学習結果の重要度 (係数)】")
    for name, val in zip(feature_names, coefs):
        print(f"  {name}: {val:.4f}")
    print("  (プラス=高齢者寄り, マイナス=若者寄り)")

    # 6. 本当のテストデータ(Testフォルダ)の判定
    X_test, _, test_names = build_dataset(TEST_DIR, is_train=False)
    
    if len(X_test) > 0:
        X_test_scaled = final_scaler.transform(X_test)
        preds = final_model.predict(X_test_scaled)
        probs = final_model.predict_proba(X_test_scaled)[:, 1]

        print(f"\n【{TEST_DIR} の判定結果】")
        for name, pred, prob in zip(test_names, preds, probs):
            label_str = "高齢者" if pred == 1 else "若者"
            print(f"{name:<25} : {label_str} (確信度: {prob*100:.1f}%)")
    else:
        print(f"テストデータが見つかりませんでした: {TEST_DIR}")

if __name__ == "__main__":
    main()