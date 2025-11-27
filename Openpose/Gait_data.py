from ultralytics import YOLO
import cv2
import csv
import numpy as np
import os

# ==========================================
# 設定エリア
# ==========================================
VIDEO_PATH = "IMG_7338.mp4"  # ここに動画ファイル名を入れる
OUTPUT_CSV = "gait_analysis_result.csv" # 保存するCSVファイル名
SHOW_WINDOW = True # 解析中の画面を表示するか (Falseなら高速化)
# ==========================================

def calculate_angle(a, b, c):
    """ 3点から角度を計算する関数 """
    a = np.array(a) # 腰
    b = np.array(b) # 膝
    c = np.array(c) # 足首
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def main():
    # 動画ファイルがあるか確認
    if not os.path.exists(VIDEO_PATH):
        print(f"エラー: '{VIDEO_PATH}' が見つかりません。")
        print("動画ファイルをこのプログラムと同じ場所に置いてください。")
        return

    # モデル読み込み
    print("モデルを読み込んでいます...")
    model = YOLO('yolov8n-pose.pt')

    # 動画読み込み
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) # 動画のフレームレート取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"解析開始: {VIDEO_PATH} (FPS: {fps}, 全フレーム数: {total_frames})")

    # CSV準備
    header = ["frame", "time_sec", 
              "L_Knee_Angle", "R_Knee_Angle", # 膝の角度
              "L_Hip_x", "L_Hip_y", "R_Hip_x", "R_Hip_y", 
              "L_Knee_x", "L_Knee_y", "R_Knee_x", "R_Knee_y", 
              "L_Ankle_x", "L_Ankle_y", "R_Ankle_x", "R_Ankle_y"]
    
    f = open(OUTPUT_CSV, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(header)

    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # 動画が終わったらループを抜ける

        # 推論実行
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()

        if results[0].keypoints is not None and results[0].keypoints.xy is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            if len(keypoints) > 0:
                p = keypoints[0] # 1人目

                # 各関節の座標を取得 (YOLO index: 11=L_Hip, 12=R_Hip, 13=L_Knee, 14=R_Knee, 15=L_Ankle, 16=R_Ankle)
                l_hip, r_hip = p[11], p[12]
                l_knee, r_knee = p[13], p[14]
                l_ankle, r_ankle = p[15], p[16]

                # 座標がすべて(0,0)でない場合のみ計算
                if np.all(l_hip) and np.all(l_knee) and np.all(l_ankle):
                    l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                else:
                    l_angle = 0

                if np.all(r_hip) and np.all(r_knee) and np.all(r_ankle):
                    r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                else:
                    r_angle = 0

                # CSVに書き込み
                time_sec = frame_count / fps # 経過時間(秒)
                row = [frame_count, f"{time_sec:.3f}", 
                       f"{l_angle:.1f}", f"{r_angle:.1f}",
                       l_hip[0], l_hip[1], r_hip[0], r_hip[1],
                       l_knee[0], l_knee[1], r_knee[0], r_knee[1],
                       l_ankle[0], l_ankle[1], r_ankle[0], r_ankle[1]]
                writer.writerow(row)

                # 画面に角度を表示
                cv2.putText(annotated_frame, f"L_Angle: {int(l_angle)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"R_Angle: {int(r_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 骨格描画
                for point in [l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]:
                     cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

        if SHOW_WINDOW:
            # スマホ動画は縦長で巨大な場合があるので、見やすくリサイズしてもOK
            # h, w = annotated_frame.shape[:2]
            # annotated_frame = cv2.resize(annotated_frame, (int(w/2), int(h/2)))
            
            cv2.imshow("Video Analysis", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"処理中... {frame_count}/{total_frames} フレーム")

    cap.release()
    cv2.destroyAllWindows()
    f.close()
    print(f"完了！ '{OUTPUT_CSV}' にデータを保存しました。")

if __name__ == "__main__":
    main()