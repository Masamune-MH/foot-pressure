import serial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3Dグラフ用
import time
import sys

# --- 設定項目 ---
SERIAL_PORT = "COM4" # ★★★ Arduino IDEで確認したポート番号 ★★★
BAUD_RATE = 9600
SENSOR_COUNT = 16

# センサー1〜8 (左足) の座標
LEFT_FOOT_POS = [
    (0, 0), (1, 0), # つまさき (Sens 1, 2)
    (0, 1), (1, 1), # 土踏まず前 (Sens 3, 4)
    (0, 2), (1, 2), # 土踏まず後 (Sens 5, 6)
    (0, 3), (1, 3)  # かかと (Sens 7, 8)
]

# センサー9〜16 (右足) の座標
# 左足の隣 (X座標にスペースを空けて配置)
RIGHT_FOOT_POS = [
    (3, 0), (4, 0), # つまさき (Sens 9, 10)
    (3, 1), (4, 1), # 土踏まず前 (Sens 11, 12)
    (3, 2), (4, 2), # 土踏まず後 (Sens 13, 14)
    (3, 3), (4, 3)  # かかと (Sens 15, 16)
]

# 全センサーの座標リスト (index 0 が Sensor 1 に対応)
SENSOR_POSITIONS = LEFT_FOOT_POS + RIGHT_FOOT_POS


def connect_to_arduino(port, baud):
    """Arduinoに接続を試みる"""
    print(f"{port} への接続を試みています...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2) 
        ser.flushInput()
        print(f"{port} に接続しました。")
        return ser
    except serial.SerialException as e:
        print(f"エラー: {port} に接続できません。\n{e}")
        sys.exit(1)


def setup_3d_plot():
    """3Dグラフの初期設定"""
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 棒の配置座標 (x, y) と太さ (dx, dy)
    x_pos = [p[0] for p in SENSOR_POSITIONS]
    y_pos = [p[1] for p in SENSOR_POSITIONS]
    z_pos = np.zeros(SENSOR_COUNT) # 底面の高さは0

    dx = np.ones(SENSOR_COUNT) * 0.6 # 棒の太さ
    dy = np.ones(SENSOR_COUNT) * 0.6
    dz = np.zeros(SENSOR_COUNT)      # 初期の高さは0

    # 色の指定 (論文の図3に似せる: 青, 赤, 緑, 紫...)
    # 簡易的に左足と右足で色を変えるなどの設定も可能
    colors = ['blue'] * 8 + ['red'] * 8

    # 初期の3D棒グラフを描画
    # (bar3dは更新が難しいため、毎回クリアして書き直す方式を採用するか、
    #  あるいはコレクションを書き換える)
    # ここではシンプルに、ループ内で `ax.bar3d` を呼び直す形にします。
    
    ax.set_title("Real-time Foot Pressure (3D)")
    ax.set_xlabel("Left / Right")
    ax.set_ylabel("Toe / Heel")
    ax.set_zlabel("Pressure (0-255)")
    ax.set_zlim(0, 260)
    
    # 視点の調整 (見やすい角度に)
    ax.view_init(elev=30, azim=-60)

    return fig, ax, x_pos, y_pos, dx, dy, colors


def main():
    ser = connect_to_arduino(SERIAL_PORT, BAUD_RATE)
    fig, ax, x_pos, y_pos, dx, dy, colors = setup_3d_plot()

    print("リアルタイム3D表示を開始します。(Ctrl+Cで終了)")
    
    try:
        while True:
            # --- データ受信 ---
            raw_data = ser.read(SENSOR_COUNT)
            ser.read_until(b'\n') 

            if len(raw_data) < SENSOR_COUNT:
                continue
                
            pressure_data = np.frombuffer(raw_data, dtype=np.uint8)

            print(f"現在の圧力値: {pressure_data.tolist()}")
            
            # dz (高さ) を更新
            dz = 255-pressure_data

            # --- 3Dグラフ更新 ---
            ax.cla() # グラフをクリア (前の棒を消す)
            
            # 再設定 (クリアするとラベルなども消えるため)
            ax.set_title("Real-time Foot Pressure")
            ax.set_xlabel("Left <---> Right")
            ax.set_ylabel("Toe <---> Heel")
            ax.set_zlabel("Pressure")
            ax.set_zlim(0, 260)
            
            # X軸, Y軸の目盛りを消すか調整
            ax.set_xticks([0.5, 3.5])
            ax.set_xticklabels(['Left Foot', 'Right Foot'])
            ax.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax.set_yticklabels(['Toe', 'Arch1', 'Arch2', 'Heel'])

            # 棒を描画
            ax.bar3d(x_pos, y_pos, np.zeros(SENSOR_COUNT), dx, dy, dz, color=colors, alpha=0.8)
            
            plt.pause(0.01)

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("終了します")
    finally:
        ser.close()
        plt.close()

if __name__ == "__main__":
    main()