from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt')

# カメラ番号 (スマホカメラがつながらない場合はここを調整)
cap = cv2.VideoCapture(1) 

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Pose", annotated_frame)

        # 変更点1: キーボードの 'q' で終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # 変更点2: マウスでウィンドウの「×」ボタンを押しても終了できるようにする
        # getWindowPropertyはウィンドウが開いているか確認します (0以下なら閉じている)
        if cv2.getWindowProperty("YOLOv8 Pose", cv2.WND_PROP_VISIBLE) < 1:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()