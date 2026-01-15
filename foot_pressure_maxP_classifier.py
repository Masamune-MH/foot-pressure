import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import warnings

# ---------------------------------------------------------
# 文字化け対策: japanize_matplotlib があれば使い、なければ英語にする
# ---------------------------------------------------------
try:
    import japanize_matplotlib
    USE_JAPANESE = True
except ImportError:
    USE_JAPANESE = False
    print("※ japanize_matplotlib がインストールされていないため、英語表記になります。")

# 警告を非表示
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# グラフ描画と評価用
from sklearn.model_selection import learning_curve, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# フォント設定（日本語が使えるかどうかに応じて切り替え）
if USE_JAPANESE:
    plt.rcParams['font.family'] = 'IPAexGothic' # japanize_matplotlibの標準
else:
    plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 設定
# ==========================================
BASE_FOLDER = "foot_pressure_data" 
TRAIN_FOLDER = os.path.join(BASE_FOLDER, "train_defined") 
TEST_FOLDER = os.path.join(BASE_FOLDER, "test_defined")   

MAX_SENSOR_VALUE = 255

# ---------------------------------------------------------
# 1. センサ座標定義
# ---------------------------------------------------------
SENSOR_COORDS = {
    # --- 前足部 ---
    2: (-1.5, 1.0), 3: (-0.5, 1.0),
    10: (0.5, 1.0), 11: (1.5, 1.0),
    # --- 中足部 ---
    4: (-1.5, 0.0), 5: (-0.5, 0.0),
    12: (0.5, 0.0), 13: (1.5, 0.0),
    # --- 後足部 ---
    6: (-1.5, -1.0), 7: (-0.5, -1.0),
    14: (0.5, -1.0), 15: (1.5, -1.0)
}

def extract_features(filepath):
    """ 改良版特徴量 (計4つ) """
    try:
        df = pd.read_csv(filepath)
        
        if 'Sensor1' in df.columns:
            sensor_cols = [f'Sensor{i}' for i in range(1, 17)]
            sensor_df = df[sensor_cols]
        else:
            sensor_df = df.iloc[:, 2:18]

        raw_data = sensor_df.values.astype(float)
        pressure_data = MAX_SENSOR_VALUE - raw_data
        pressure_data = np.maximum(pressure_data, 0)
        
        cop_x_list = []
        cop_y_list = []
        valid_frames = 0
        
        for frame in pressure_data:
            total_p = 0
            w_x = 0
            w_y = 0
            for i, (x, y) in SENSOR_COORDS.items():
                p = frame[i]
                if p < 5: continue
                total_p += p
                w_x += p * x
                w_y += p * y
            
            if total_p >= 1e-3:
                cop_x_list.append(w_x / total_p)
                cop_y_list.append(w_y / total_p)
                valid_frames += 1
        
        if len(cop_x_list) < 2:
            return None

        cop_x_arr = np.array(cop_x_list)
        cop_y_arr = np.array(cop_y_list)

        dx = np.diff(cop_x_arr)
        dy = np.diff(cop_y_arr)
        total_dist = np.sum(np.sqrt(dx**2 + dy**2))
        std_x = np.std(cop_x_arr)
        std_y = np.std(cop_y_arr)
        duration = float(valid_frames)

        features = [total_dist, std_x, std_y, duration]
        return features

    except Exception as e:
        print(f"読込エラー {filepath}: {e}")
        return None

def load_train_data(folder_path):
    X = []
    y = []
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"学習データフォルダ: {folder_path}")
    
    for f in files:
        fname = os.path.basename(f).lower()
        label = None
        if '_elderly' in fname: label = 1
        elif '_young' in fname: label = 0
            
        if label is not None:
            feats = extract_features(f)
            if feats is not None:
                X.append(feats)
                y.append(label)
    return np.array(X), np.array(y)

def augment_data(X, y, num_copies=10, noise_level=0.05):
    """ データ拡張 """
    X_aug = []
    y_aug = []
    np.random.seed(42)
    for feat, label in zip(X, y):
        X_aug.append(feat)
        y_aug.append(label)
        for _ in range(num_copies):
            noise = np.random.normal(0, noise_level, len(feat))
            noise[0] *= 10.0 
            noise[3] *= 20.0 
            X_aug.append(feat + noise)
            y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)

def plot_learning_curve_graph(estimator, X, y, title="Learning Curve"):
    """ 
    学習曲線を描画（Y軸修正版）
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    
    # ラベル設定（文字化け回避のため条件分岐）
    if USE_JAPANESE:
        xlabel_txt = "学習データ数 (枚)"
        ylabel_txt = "正解率 (Accuracy)"
        label_train = "Training score (学習)"
        label_cv = "Cross-validation (検証)"
    else:
        xlabel_txt = "Training examples"
        ylabel_txt = "Score (Accuracy)"
        label_train = "Training score"
        label_cv = "Cross-validation score"

    plt.xlabel(xlabel_txt, fontsize=14)
    plt.ylabel(ylabel_txt, fontsize=14)
    
    # ★★★ ここを修正しました！ ★★★
    # 0.4 ではなく 0.0 から表示して、見切れを防ぐ
    plt.ylim(0.3, 1.1)
    
    # チャンスレベル(50%)の線
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Chance Level (50%)')
    
    plt.grid(alpha=0.3)

    # 訓練データ
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label=label_train, linewidth=2)

    # 検証データ
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label=label_cv, linewidth=2)

    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()

# =========================================================
# メイン処理
# =========================================================
def main():
    X_raw, y_raw = load_train_data(TRAIN_FOLDER)
    if len(X_raw) == 0:
        print("エラー: 学習データが見つかりません。")
        return

    scaler = StandardScaler()
    X_raw_scaled = scaler.fit_transform(X_raw)

    print("\n=======================================================")
    print(" Phase 1: 各モデルのパラメータチューニング (LOOCV)")
    print("=======================================================")

    # A. ロジスティック回帰
    print("\n[1] Logistic Regression...")
    param_grid_lr = {'C': [0.001, 0.01, 0.1, 1.0, 10.0], 'solver': ['liblinear']}
    grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=2000), param_grid_lr, cv=LeaveOneOut(), scoring='accuracy')
    grid_lr.fit(X_raw_scaled, y_raw)
    best_lr = grid_lr.best_estimator_
    print(f"  Best Params: {grid_lr.best_params_}")

    # B. SVM
    print("\n[2] SVM...")
    param_grid_svm = {'C': [0.001, 0.01, 0.1, 1.0, 10.0], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['rbf', 'linear']}
    grid_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=LeaveOneOut(), scoring='accuracy')
    grid_svm.fit(X_raw_scaled, y_raw)
    best_svm = grid_svm.best_estimator_
    print(f"  Best Params: {grid_svm.best_params_}")

    print("\n=======================================================")
    print(" Phase 2: アンサンブル学習 & データ拡張で最終モデル構築")
    print("=======================================================")

    ensemble_model = VotingClassifier(estimators=[('lr', best_lr), ('svm', best_svm)], voting='soft')
    X_aug, y_aug = augment_data(X_raw, y_raw, num_copies=10, noise_level=0.05)
    scaler_final = StandardScaler()
    X_aug_scaled = scaler_final.fit_transform(X_aug)

    # グラフ表示処理
    print("グラフを作成中... (ウィンドウを閉じると続きが実行されます)")
    
    # 1. 学習曲線
    plot_learning_curve_graph(ensemble_model, X_aug_scaled, y_aug, title="Ensemble Learning Curve (New Features)")
    
    # 2. 混同行列
    y_pred_cv = cross_val_predict(ensemble_model, X_aug_scaled, y_aug, cv=5)
    cm = confusion_matrix(y_aug, y_pred_cv)
    
    # 文字化け対策ラベル
    labels = ["若者(Young)", "高齢者(Elderly)"] if USE_JAPANESE else ["Young", "Elderly"]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    
    if USE_JAPANESE:
        plt.title("混同行列 (Cross Validation)", fontsize=16)
        plt.xlabel("AIの予測", fontsize=14)
        plt.ylabel("正解データ", fontsize=14)
    else:
        plt.title("Confusion Matrix (Cross Validation)", fontsize=16)
    
    for text in disp.text_.ravel():
        text.set_fontsize(18)

    plt.show() 

    # 最終学習
    ensemble_model.fit(X_aug_scaled, y_aug)
    print("学習完了！")

    # 未知データの分類
    print(f"\n=======================================================")
    print(f" 未知データ({TEST_FOLDER})の分類結果")
    print("=======================================================")
    
    test_files = glob.glob(os.path.join(TEST_FOLDER, "*.csv"))
    print(f"{'File Name':<25} | {'Probability(Elderly)':<20} | {'Prediction'}")
    print("-" * 65)

    for f_path in test_files:
        feats = extract_features(f_path)
        if feats is None: continue
        
        feats_scaled = scaler_final.transform([feats])
        prediction = ensemble_model.predict(feats_scaled)[0]
        proba = ensemble_model.predict_proba(feats_scaled)[0][1]
        
        label_str = "Elderly Pattern" if prediction == 1 else "Young Pattern"
        if USE_JAPANESE:
            label_str = "高齢者パターン" if prediction == 1 else "若者パターン"

        fname = os.path.basename(f_path)
        print(f"{fname:<25} | {proba*100:.1f}%                | {label_str}")

if __name__ == "__main__":
    main()