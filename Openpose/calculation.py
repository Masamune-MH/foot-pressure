import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 設定エリア: 比較したい2つのCSVファイルを指定
# ==========================================
FILE_A = "gait_analysis_result.csv"      # Aさんのデータ (例: 元気な歩き方)
FILE_B = "result_edobashiri.csv"      # Bさんのデータ (比較用に同じファイルでも可)

# グラフの線の名前
LABEL_A = "Person A"
LABEL_B = "Person B"

# ノイズ除去（移動平均）の強度
# 数値を大きくすると線が滑らかになります (1=そのまま, 5=少し滑らか, 10=かなり滑らか)
SMOOTHING = 5 
# ==========================================

def load_and_smooth(filepath, window):
    """CSVを読み込んで、少し滑らかにする関数"""
    try:
        df = pd.read_csv(filepath)
        # 移動平均を使ってノイズ（ガタつき）を取る
        df['L_Knee_Angle'] = df['L_Knee_Angle'].rolling(window=window, min_periods=1).mean()
        df['R_Knee_Angle'] = df['R_Knee_Angle'].rolling(window=window, min_periods=1).mean()
        return df
    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
        return None

def main():
    # データの読み込み
    df_a = load_and_smooth(FILE_A, SMOOTHING)
    df_b = load_and_smooth(FILE_B, SMOOTHING)

    if df_a is None or df_b is None:
        return

    # グラフの作成 (2段構成: 上が右膝、下が左膝)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- 1. 右膝 (Right Knee) の比較 ---
    ax1.plot(df_a['time_sec'], df_a['R_Knee_Angle'], label=LABEL_A, color='blue', linewidth=2)
    ax1.plot(df_b['time_sec'], df_b['R_Knee_Angle'], label=LABEL_B, color='red', linestyle='--', linewidth=2)
    
    ax1.set_title("Right Knee Angle Comparison (右膝の角度)", fontsize=14)
    ax1.set_ylabel("Angle (degrees)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- 2. 左膝 (Left Knee) の比較 ---
    ax2.plot(df_a['time_sec'], df_a['L_Knee_Angle'], label=LABEL_A, color='blue', linewidth=2)
    ax2.plot(df_b['time_sec'], df_b['L_Knee_Angle'], label=LABEL_B, color='red', linestyle='--', linewidth=2)
    
    ax2.set_title("Left Knee Angle Comparison (左膝の角度)", fontsize=14)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Angle (degrees)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()