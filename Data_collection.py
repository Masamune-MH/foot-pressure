import serial
import time
import csv
import sys
import numpy as np

# --- 設定 ---
SERIAL_PORT = "COM5"  # Arduinoのポートに合わせて変更
BAUD_RATE = 9600
SENSOR_COUNT = 16
CSV_FILENAME = "foot_pressure_data.csv"
SAMPLES_PER_SESSION = 100 # 1回の計測で保存するデータ数（約5〜10秒分）

def collect_data():
    # シリアル通信のセットアップ
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # 接続待機
        ser.flushInput()
        print(f"{SERIAL_PORT} に接続しました。")
    except Exception as e:
        print(f"接続エラー: {e}")
        sys.exit()

    # ラベル（名前）の入力
    label_name = input("被験者の名前（ラベル）をアルファベットで入力してください: ")
    print(f"{label_name} さんのデータを計測します。センサーに乗って静止してください...")
    
    # 準備時間のカウントダウン
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("計測開始！")

    collected_data = []

    try:
        while len(collected_data) < SAMPLES_PER_SESSION:
            # バイナリデータの読み込み（前回の修正コードに対応）
            raw_data = ser.read(SENSOR_COUNT)
            ser.read_until(b'\n') # 改行読み捨て

            if len(raw_data) == SENSOR_COUNT:
                # バイト列を数値リスト(0-255)に変換
                pressure_values = list(raw_data)
                
                # 画面に簡易表示
                print(f"Data {len(collected_data)+1}/{SAMPLES_PER_SESSION}: {pressure_values}")
                
                # ラベルを先頭に追加してリスト作成: [Name, 100, 200, ...]
                row = [label_name] + pressure_values
                collected_data.append(row)

    except KeyboardInterrupt:
        print("\n中断しました。")
        ser.close()
        return

    print("計測終了。ファイルに保存します...")
    ser.close()

    # CSVファイルへの追記モード('a')での保存
    # ファイルがなければヘッダーを作成
    try:
        with open(CSV_FILENAME, 'a', newline='') as f:
            writer = csv.writer(f)
            # ファイルが空ならヘッダーを書き込む
            if f.tell() == 0:
                header = ["Label"] + [f"Sensor{i+1}" for i in range(SENSOR_COUNT)]
                writer.writerow(header)
            
            writer.writerows(collected_data)
            
        print(f"保存完了: {CSV_FILENAME}")
        
    except Exception as e:
        print(f"保存エラー: {e}")

if __name__ == "__main__":
    collect_data()