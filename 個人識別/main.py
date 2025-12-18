import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif' 

# ==========================================
# 設定エリア
# ==========================================
# データが入っているフォルダ
BASE_FOLDER = "../foot_pressure_data"
TEST_FOLDER = os.path.join(BASE_FOLDER, "test")

# ★高齢者判定の閾値（中足部比率）
# 「全体の圧力のうち、22%以上が中足部にかかっていたら高齢者パターン」とする
THRESHOLD_MF_RATIO = 0.22 

# ★センサの最大値（これを使って値を反転させます）
# Arduino側が0~255なら255、0~1023なら1023にしてください
MAX_SENSOR_VALUE = 255

# --- センサの場所定義 (指定された配置) ---
# 前足部 (Forefoot): 0~3, 8~11
IDX_FOREFOOT = [0, 1, 2, 3, 8, 9, 10, 11]
# 中足部 (Midfoot): 4~5, 12~13
IDX_MIDFOOT  = [4, 5, 12, 13]
# 後足部 (Rearfoot): 6~7, 14~15
IDX_REARFOOT = [6, 7, 14, 15]


def calculate_pressure_features(raw_values):
    """ 
    センサ値(小さいほど高圧)を、(大きいほど高圧)に変換して計算する関数
    """
    # ==========================================================
    # 1. 値の反転処理 (ここがポイント)
    # 圧力がかかると値が小さくなるセンサを、扱いやすいように逆転させます
    # ==========================================================
    pressure_values = MAX_SENSOR_VALUE - raw_values
    
    # 計算誤差でマイナスにならないように0で止める
    pressure_values = np.maximum(pressure_values, 0)
    
    # 2. 領域ごとの圧力合計計算（反転後の値を使います）
    p_ff = np.sum(pressure_values[IDX_FOREFOOT]) # 前足部
    p_mf = np.sum(pressure_values[IDX_MIDFOOT])  # 中足部
    p_rf = np.sum(pressure_values[IDX_REARFOOT]) # 後足部
    
    total_pressure = p_ff + p_mf + p_rf
    
    # 足が浮いている（圧力がほぼゼロ）場合は計算しない
    # ノイズ対策として、合計値が小さいときは無視します
    if total_pressure < 50: 
        return None, 0, 0, 0

    # 3. 比率計算 (中足部 / 全体)
    ratio_mf = p_mf / total_pressure
    
    return ratio_mf, p_ff, p_mf, p_rf

def analyze_file_pattern(filepath):
    """ 1つのCSVファイルを読み込んで判定する """
    try:
        # CSV読み込み
        df = pd.read_csv(filepath)
        
        # データ列の特定（Sensor1～Sensor16 または 2列目以降）
        if 'Sensor1' in df.columns:
            sensor_cols = [f'Sensor{i}' for i in range(1, 17)]
            sensor_df = df[sensor_cols]
        else:
            # カラム名がない場合、2列目から18列目(インデックス2~17)を取得
            sensor_df = df.iloc[:, 2:18]

        # 計算のために少数(float)型に変換
        raw_sensor_data = sensor_df.values.astype(float)
        
        if raw_sensor_data.shape[1] != 16:
            print(f"列数エラー: {filepath} は16列ではありません")
            return None

    except Exception as e:
        print(f"読込エラー ({filepath}): {e}")
        return None

    # --- フレームごとの判定 ---
    elderly_count = 0
    young_count = 0
    valid_ratios = []

    for i in range(len(raw_sensor_data)):
        row_raw = raw_sensor_data[i]
        
        # 特徴量計算（ここで内部的に値が反転されます）
        result = calculate_pressure_features(row_raw)
        
        # 足が乗っていないフレームはスキップ
        if result[0] is None:
            continue
            
        ratio_mf, p_ff, p_mf, p_rf = result
        valid_ratios.append(ratio_mf)

        # ★ 判定ロジック ★
        # 中足部の比率が閾値を超えていたら「高齢者パターン」
        if ratio_mf > THRESHOLD_MF_RATIO:
            elderly_count += 1
        else:
            young_count += 1

    # 結果集計
    valid_frames = elderly_count + young_count
    if valid_frames == 0:
        return {
            "file": os.path.basename(filepath),
            "result": "データなし(足離れ)",
            "avg_mf_ratio": 0
        }

    # 最終判定（多数決）
    final_judgement = "高齢者パターン" if elderly_count > young_count else "若者パターン"
    avg_mf_ratio = np.mean(valid_ratios)

    return {
        "file": os.path.basename(filepath),
        "result": final_judgement,
        "avg_mf_ratio": avg_mf_ratio
    }

def main():
    print(f"=== 足圧判定プログラム ===")
    print(f"設定: 値が大きいほど高圧力に変換して計算")
    print(f"閾値: 中足部が全体の {THRESHOLD_MF_RATIO*100:.1f}% 以上なら高齢者と判定")
    print("-" * 60)

    # フォルダ内のCSVリストを取得
    csv_files = glob.glob(os.path.join(TEST_FOLDER, "*.csv"))
    
    if not csv_files:
        print(f"CSVファイルが見つかりません: {TEST_FOLDER}")
        return

    print(f"{'ファイル名':<25} | {'中足部率':<10} | {'判定結果'}")
    print("-" * 60)

    for filepath in csv_files:
        res = analyze_file_pattern(filepath)
        if res:
            ratio_str = f"{res['avg_mf_ratio']*100:.1f}%"
            print(f"{res['file']:<25} | {ratio_str:<10} | {res['result']}")

if __name__ == "__main__":
    main()