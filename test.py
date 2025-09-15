import cv2, time, argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

model = YOLO(args.weights); model.to('cpu')  # ép chạy CPU
cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

t_prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok: break
    res = model(frame, device='cpu', conf=0.35, imgsz=args.imgsz, verbose=False)[0]
    for b in res.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    fps = 1/(time.time()-t_prev); t_prev = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("YOLOv8 Plate", frame)
    if cv2.waitKey(1) == 27: break  # ESC

cap.release(); cv2.destroyAllWindows()
