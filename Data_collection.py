import serial
import time
import csv
import sys
import numpy as np

# --- 設定 ---
SERIAL_PORT = "COM5"
BAUD_RATE = 9600
SENSOR_COUNT = 16
# ファイル名は動的に生成するため、ディレクトリのみ指定などの運用が良いですが、一旦元のままにします
CSV_FILENAME_BASE = "foot_pressure_data/" 

# 【変更点1】サンプル数指定ではなく「計測時間（秒）」で指定する
MEASURE_DURATION_SEC = 30  # 30秒〜60秒推奨

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
        # 【変更点2】指定した経過時間が過ぎるまでループする
        while (time.time() - start_time) < MEASURE_DURATION_SEC:
            
            # データの読み込み
            raw_data = ser.read(SENSOR_COUNT)
            # 改行コード等がArduino側から送られている場合は読み飛ばす処理が必要
            # 今回は ser.read(SENSOR_COUNT) で固定バイト取っているので、
            # Arduino側が println ではなく write で生データを送っている想定であれば改行読み捨ては不要かもしれません。
            # もしArduino側で Serial.println() しているなら以下は必要です。
            ser.read_until(b'\n') 

            if len(raw_data) == SENSOR_COUNT:
                # バイト列を数値リスト(0-255)に変換
                pressure_values = list(raw_data)
                
                # ラベルとタイムスタンプを追加
                # 【追加提案】解析時に「時間経過」が必要になるため、経過時間も保存しておくと便利です
                elapsed = round(time.time() - start_time, 4)
                row = [label_name, elapsed] + pressure_values
                collected_data.append(row)

                # 【変更点3】毎回printすると遅くなるので、1秒ごとに経過を表示
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
        # ファイル名を決定
        save_path = f"{CSV_FILENAME_BASE}{label_name}.csv"
        
        with open(save_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # ファイルが空ならヘッダーを書き込む
            if f.tell() == 0:
                # HeaderにTimeを追加
                header = ["Label", "Time"] + [f"Sensor{i+1}" for i in range(SENSOR_COUNT)]
                writer.writerow(header)
            
            writer.writerows(collected_data)
            
        print(f"保存完了: {save_path}")
        print(f"総サンプル数: {len(collected_data)}")
        # サンプリングレートの目安を表示（解析時に重要）
        if len(collected_data) > 0:
            print(f"平均サンプリングレート: {len(collected_data)/MEASURE_DURATION_SEC:.1f} Hz")
        
    except Exception as e:
        print(f"保存エラー: {e}")

if __name__ == "__main__":
    collect_data()