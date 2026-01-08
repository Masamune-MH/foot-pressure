import serial
import time
import csv
import sys
import numpy as np

# --- 設定 ---
SERIAL_PORT = "COM5"
BAUD_RATE = 9600  # もしArduino側を115200に上げられたらここも変えてください

SENSOR_COUNT = 16
CSV_FILENAME_BASE = "foot_pressure_data/" 

# 計測時間（秒）
MEASURE_DURATION_SEC = 30 

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
    print(f"{label_name} さんのデータを {MEASURE_DURATION_SEC} 秒間計測します。")
    print("センサーに乗って静止してください...")
    
    # 準備時間のカウントダウン
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("計測開始！")

    collected_data = []
    start_time = time.time()
    last_print_time = time.time()

    try:
        while (time.time() - start_time) < MEASURE_DURATION_SEC:
            
            # データの読み込み
            raw_data = ser.read(SENSOR_COUNT)
            # 改行読み捨て（Arduino側がSerial.writeだけで送ってるなら不要だが、念のため）
            ser.read_until(b'\n') 

            if len(raw_data) == SENSOR_COUNT:
                # 【重要修正】ここで値を反転させます！
                # 元のデータ: 255(浮いてる) ～ 0(重い)
                # 修正データ: 0(浮いてる) ～ 255(重い)
                pressure_values = [255 - val for val in raw_data]
                
                # 経過時間を記録
                elapsed = round(time.time() - start_time, 4)

                if elapsed >= 1.0: 
                    row = [label_name, elapsed] + pressure_values
                    collected_data.append(row)
                    

                # 1秒ごとに経過を表示
                if time.time() - last_print_time > 1.0:
                    print(f"計測中... 残り {int(MEASURE_DURATION_SEC - elapsed)} 秒 (現在 {len(collected_data)} サンプル)")
                    last_print_time = time.time()

    except KeyboardInterrupt:
        print("\n中断しました。")
        ser.close()
        return

    print("計測終了。ファイルに保存します...")
    ser.close()

    # CSV保存
    try:
        save_path = f"{CSV_FILENAME_BASE}{label_name}.csv"
        
        with open(save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # ファイルが空ならヘッダーを書き込む
            if f.tell() == 0:
                header = ["Label", "Time"] + [f"Sensor{i+1}" for i in range(SENSOR_COUNT)]
                writer.writerow(header)
            
            writer.writerows(collected_data)
            
        print(f"保存完了: {save_path}")
        print(f"総サンプル数: {len(collected_data)}")

        if len(collected_data) > 0:
            print(f"平均サンプリングレート: {len(collected_data)/MEASURE_DURATION_SEC:.1f} Hz")
        
    except Exception as e:
        print(f"保存エラー: {e}")

if __name__ == "__main__":
    collect_data()