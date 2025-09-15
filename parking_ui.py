# parking_ui.py  — YOLOv8 (detect) + EasyOCR (read) + GUI Qt/PySide6
# ---------------------------------------------------------------
# Yêu cầu: pip install PySide6 ultralytics easyocr opencv-python paho-mqtt
# ---------------------------------------------------------------

import os, re, cv2, sys, json, time, socket, random, string, datetime, subprocess, threading, collections
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from ultralytics import YOLO

# OCR (ưu tiên EasyOCR cho nhẹ — không cần Paddle)
import easyocr

# MQTT (optional)
try:
    from paho.mqtt import client as mqtt
except Exception:
    mqtt = None

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLineEdit, QPushButton, QStatusBar, QMessageBox,
    QSizePolicy, QDialog, QComboBox, QDialogButtonBox, QFormLayout, QCheckBox
)

# ======================== CẤU HÌNH CHUNG ========================

CFG_FILE = "config.json"

# (1) Đặt đúng đường dẫn weight YOLO của bạn
YOLO_MODEL_PATH = r"E:\FIRMWAVE\Automatic-License-Plate-Recognition-using-YOLOv8\license_plate_detector.pt"

# (2) Thư mục lưu ảnh
DIR_IN  = Path("plates/IN")
DIR_OUT = Path("plates/OUT")
DIR_IN.mkdir(parents=True, exist_ok=True)
DIR_OUT.mkdir(parents=True, exist_ok=True)

# (3) Tham số ALPR
YOLO_CONF     = 0.35
YOLO_IMGSZ    = 416
MIN_REL_AREA  = 0.010     # bbox >= 1% frame
MIN_SHARPNESS = 60.0      # nét tối thiểu (Laplacian var)
CAP_WIDTH, CAP_HEIGHT = 640, 480

# (4) Phí demo
FEE_FLAT = 3000

# Regex biển VN (rộng)
PLATE_RE = re.compile(r"[0-9]{2,3}[A-Z]{1,2}[-\s]?[0-9]{3,5}")

@dataclass
class UiConfig:
    cam_in_index: int = 0
    cam_out_index: int = -1
    total_slots: int = 50
    mqtt_enable: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    gate_id: str = "gate1"
    auto_start_broker: bool = True
    broker_exe: str = r"C:\Program Files\mosquitto\mosquitto.exe"
    broker_conf: str = r"E:\FIRMWAVE\project\mosquitto.conf"

def load_config() -> UiConfig:
    if os.path.exists(CFG_FILE):
        try:
            with open(CFG_FILE, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            return UiConfig(**{
                **UiConfig().__dict__,
                **{k: d.get(k, getattr(UiConfig(), k)) for k in UiConfig().__dataclass_fields__.keys()}
            })
        except Exception:
            pass
    return UiConfig()

def save_config(cfg: UiConfig):
    with open(CFG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg.__dict__, fh, ensure_ascii=False, indent=2)

# ======================== TIỆN ÍCH ẢNH/VIDEO =====================

def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def enhance_for_plate(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def np_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()

def set_pixmap_fit_no_upscale(label: QLabel, img: QImage):
    if label.width() <= 0 or label.height() <= 0 or img.isNull():
        return
    pix = QPixmap.fromImage(img)
    sw, sh = label.width() / pix.width(), label.height() / pix.height()
    scale = min(1.0, sw, sh)
    new_size = QSize(int(pix.width()*scale), int(pix.height()*scale))
    label.setPixmap(pix.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    label.setAlignment(Qt.AlignCenter)
    label.setScaledContents(False)

def list_cameras(max_index=8) -> List[int]:
    found = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found

# ======================== TIỆN ÍCH MẠNG/MQTT =====================

def is_port_open(host: str, port: int, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def get_local_ips() -> set:
    ips = {"127.0.0.1", "localhost", "0.0.0.0"}
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            ips.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0]); s.close()
    except Exception:
        pass
    return ips

# ======================== CAMERA THREAD ==========================

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    opened = Signal(bool)

    def __init__(self, source=0, width=CAP_WIDTH, height=CAP_HEIGHT, mirror=False, parent=None):
        super().__init__(parent)
        self.source, self.width, self.height, self.mirror = source, width, height, mirror
        self._running = False
        self._buffer = collections.deque(maxlen=25)
        self._buf_lock = threading.Lock()
        self.cap = None

    def run(self):
        self._running = True
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        try:
            self.cap = cv2.VideoCapture(self.source, backend)
            if self.width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok = self.cap.isOpened(); self.opened.emit(ok)
            if not ok: return

            target_dt = 1/25.0
            last_emit = 0.0
            while self._running:
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    QThread.msleep(50); continue
                if self.mirror: frame = cv2.flip(frame, 1)

                try:
                    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                    score = sharpness_score(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                except Exception:
                    score = 0.0
                with self._buf_lock:
                    self._buffer.append((score, frame.copy()))

                if time.time() - last_emit >= target_dt:
                    disp = frame
                    h0,w0 = frame.shape[:2]
                    if w0 > 640:
                        scale = 640 / w0
                        disp = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
                    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(qimg)
                    last_emit = time.time()

                rem = target_dt - (time.time() - t0)
                if rem > 0: QThread.msleep(int(rem*1000))
        finally:
            try:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass

    def stop(self):
        self._running = False
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.wait(1500)

    def best_recent_frame(self, min_score: float = MIN_SHARPNESS) -> Optional[np.ndarray]:
        with self._buf_lock:
            if not self._buffer:
                return None
            s, f = max(self._buffer, key=lambda t: t[0])
            return f.copy() if s >= min_score else None

# ======================== ALPR (YOLO + EasyOCR) ===================

def clean_plate_text(txt: str) -> str:
    t = txt.upper().replace("O", "0")
    t = re.sub(r"[^A-Z0-9\s-]", "", t)
    m = PLATE_RE.search(t.replace(" ", ""))
    if not m: return t.strip()
    raw = m.group(0)
    if "-" not in raw:
        if len(raw) > 3 and raw[2].isalpha():
            raw = raw[:2] + "-" + raw[2:]
        elif len(raw) > 4 and raw[3].isalpha():
            raw = raw[:3] + "-" + raw[3:]
    raw = re.sub(r"-([A-Z]{1,2})(\d+)", r"-\1 \2", raw)
    return raw.strip()

class ALPR:
    def __init__(self, weights: str, conf=YOLO_CONF, imgsz=YOLO_IMGSZ):
        self.model = YOLO(weights)
        self.model.to('cpu')
        self.conf = conf
        self.imgsz = imgsz
        self.reader = easyocr.Reader(['en'], gpu=False)

    def infer(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        """
        Detect biển số, crop + OCR, trả về (best_text, debug_img).
        """
        debug = frame.copy()
        H, W = frame.shape[:2]
        results = self.model(frame, device='cpu', conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        best_txt, best_score = None, -1.0
        if results.boxes is None or len(results.boxes) == 0:
            return None, debug

        # Sắp xếp bbox theo confidence giảm dần
        confs = results.boxes.conf.cpu().numpy()
        order = np.argsort(-confs)

        for idx in order:
            b = results.boxes[int(idx)]
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W-1,x2), min(H-1,y2)
            w,h = x2-x1, y2-y1
            if w<=1 or h<=1: continue
            rel_area = (w*h)/(W*H)
            if rel_area < MIN_REL_AREA: continue

            crop = frame[y1:y2, x1:x2]
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if sharpness_score(g) < MIN_SHARPNESS:  # bỏ crop mờ
                continue
            g = cv2.bilateralFilter(g, 7, 50, 50)
            thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 7)

            dets = self.reader.readtext(thr)
            text = " ".join([d[1] for d in dets]) if dets else ""
            text = clean_plate_text(text)

            score = float(b.conf.item()) + 0.05*len(text)
            if text and score > best_score:
                best_score = score
                best_txt = text

            cv2.rectangle(debug, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(debug, text if text else "?", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        return best_txt, debug

# ======================== UI PHỤ TRỢ =============================

def qlabel_video_placeholder(text=""):
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setMinimumSize(QSize(360, 240))
    lbl.setStyleSheet("QLabel{background:#1f1f1f;color:#cccccc;border:1px solid #3a3a3a;}")
    return lbl

# ======================== SETTINGS DIALOG =========================

class SettingsDialog(QDialog):
    def __init__(self, cfg: UiConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt"); self.resize(520, 380)

        cams = list_cameras()
        self.cb_in  = QComboBox()
        self.cb_out = QComboBox(); self.cb_out.addItem("— Tắt —", -1)
        if not cams:
            self.cb_in.addItem("Không tìm thấy camera", -1)
        else:
            for i in cams:
                self.cb_in.addItem(f"Camera {i}", i)
                self.cb_out.addItem(f"Camera {i}", i)

        if cams and cfg.cam_in_index in cams:
            self.cb_in.setCurrentIndex(cams.index(cfg.cam_in_index))
        if cfg.cam_out_index == -1:
            self.cb_out.setCurrentIndex(0)
        elif cfg.cam_out_index in cams:
            self.cb_out.setCurrentIndex(1 + cams.index(cfg.cam_out_index))

        self.ed_slots  = QLineEdit(str(cfg.total_slots))

        self.chk_mqtt  = QCheckBox("Bật MQTT"); self.chk_mqtt.setChecked(cfg.mqtt_enable)
        self.ed_host   = QLineEdit(cfg.mqtt_host)
        self.ed_port   = QLineEdit(str(cfg.mqtt_port))
        self.ed_gate   = QLineEdit(cfg.gate_id)

        self.chk_autob = QCheckBox("Tự khởi động Mosquitto nếu broker là máy này")
        self.chk_autob.setChecked(cfg.auto_start_broker)
        self.ed_bexe   = QLineEdit(cfg.broker_exe)
        self.ed_bconf  = QLineEdit(cfg.broker_conf)

        form = QFormLayout()
        form.addRow("Ngõ vào:", self.cb_in)
        form.addRow("Ngõ ra:", self.cb_out)
        form.addRow("SLOT TỔNG:", self.ed_slots)
        form.addRow(self.chk_mqtt)
        form.addRow("MQTT Host:", self.ed_host)
        form.addRow("MQTT Port:", self.ed_port)
        form.addRow("Gate ID:", self.ed_gate)
        form.addRow(self.chk_autob)
        form.addRow("mosquitto.exe:", self.ed_bexe)
        form.addRow("mosquitto.conf:", self.ed_bconf)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self); layout.addLayout(form); layout.addWidget(buttons)

    def values(self):
        return (
            self.cb_in.currentData(), self.cb_out.currentData(), int(self.ed_slots.text()),
            self.chk_mqtt.isChecked(), self.ed_host.text().strip(),
            int(self.ed_port.text()), self.ed_gate.text().strip(),
            self.chk_autob.isChecked(), self.ed_bexe.text().strip(), self.ed_bconf.text().strip()
        )

# ========================= MAIN WINDOW ============================

class MainWindow(QMainWindow):
    def __init__(self, cfg: UiConfig):
        super().__init__()
        self.cfg = cfg

        # 1) UI (nhẹ, dựng trước)
        self._build_ui()

        # 2) Trạng thái/biến dùng chung
        self._in_records = {}     # {plate: datetime_in}
        self._local_ips = get_local_ips()
        self._mqtt_connected = False
        self._esp_online = False
        self._esp_last_hb = 0.0
        self._hb_timeout = 10.0
        self._mosq_proc = None
        self.mqtt_client = None

        # 3) ALPR model & OCR (nặng — làm ngay để sẵn sàng)
        self._init_models()

        # 4) Camera (thread)
        self.cam_in_worker: Optional[CameraWorker] = None
        self.cam_out_worker: Optional[CameraWorker] = None
        self.start_cameras()

        # 5) MQTT/Broker
        self.ensure_broker_running()
        self.init_mqtt()

        # 6) Timers
        self._start_timers()

    # ---------- UI ----------
    def _build_ui(self):
        self.setWindowTitle("Phần mềm quản lý bãi gửi xe")
        self.resize(1280, 780)

        act_settings = QAction("Thiết lập", self); act_settings.triggered.connect(self.open_settings)
        act_full = QAction("Toàn màn hình", self, checkable=True); act_full.triggered.connect(self.toggle_fullscreen)
        menu = self.menuBar().addMenu("Cài đặt"); menu.addAction(act_settings); menu.addAction(act_full)

        self.lbl_cam_in  = qlabel_video_placeholder()
        self.lbl_img_in  = qlabel_video_placeholder("Ảnh xe vào")
        self.lbl_cam_out = qlabel_video_placeholder()
        self.lbl_img_out = qlabel_video_placeholder("Ảnh xe ra")

        grid = QGridLayout()
        grid.addWidget(self._group("Camera ngõ vào", self.lbl_cam_in), 0, 0)
        grid.addWidget(self._group("Ảnh xe vào", self.lbl_img_in),     0, 1)
        grid.addWidget(self._group("Camera ngõ ra", self.lbl_cam_out), 1, 0)
        grid.addWidget(self._group("Ảnh xe ra", self.lbl_img_out),     1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1);    grid.setRowStretch(1, 1)
        left = QWidget(); left.setLayout(grid)

        self.lbl_clock = QLabel("--:--:--"); self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setStyleSheet("font-size:22px;font-weight:600;")

        self.lbl_mqtt_state = QLabel("OFF"); self.lbl_mqtt_state.setStyleSheet("color:#bbb;font-weight:700;")
        self.lbl_mqtt_broker = QLabel("-"); self.lbl_mqtt_gate = QLabel("-"); self.lbl_mqtt_cid = QLabel("-")
        mqtt_form = QFormLayout()
        mqtt_form.addRow("Trạng thái:", self.lbl_mqtt_state)
        mqtt_form.addRow("Broker:", self.lbl_mqtt_broker)
        mqtt_form.addRow("Gate ID:", self.lbl_mqtt_gate)
        mqtt_form.addRow("Client ID:", self.lbl_mqtt_cid)
        box_mqtt = QGroupBox("Kết nối MQTT / ESP32"); wm = QWidget(); wm.setLayout(mqtt_form)
        lay_mqtt = QVBoxLayout(); lay_mqtt.addWidget(wm); box_mqtt.setLayout(lay_mqtt)

        self.ed_plate_cnt = self._count_box("0")
        self.ed_card  = self._ro_edit()
        self.ed_plate = self._ro_edit()
        self.ed_tin   = self._ro_edit()
        self.ed_tout  = self._ro_edit()
        self.ed_tdiff = self._ro_edit()
        self.ed_fee   = self._ro_edit()
        self.ed_slots_total = self._ro_edit()
        self.ed_slots_used  = self._ro_edit()
        self.ed_slots_free  = self._ro_edit()

        self.btn_in  = QPushButton("Chụp IN");  self.btn_in.clicked.connect(self.on_shoot_in)
        self.btn_out = QPushButton("Chụp OUT"); self.btn_out.clicked.connect(self.on_shoot_out)
        btn_sync  = QPushButton("Đồng bộ"); btn_sync.clicked.connect(self.on_sync)
        btn_clear = QPushButton("Xóa");     btn_clear.clicked.connect(self.on_clear)

        form = QGridLayout(); r=0
        for label, widget in [
            ("SỐ XE", self.ed_plate_cnt),
            ("MÃ THẺ", self.ed_card),
            ("BIỂN SỐ", self.ed_plate),
            ("T/G XE VÀO", self.ed_tin),
            ("T/G XE RA", self.ed_tout),
            ("T/G GỬI XE", self.ed_tdiff),
            ("PHÍ GỬI XE", self.ed_fee),
            ("SLOT TỔNG", self.ed_slots_total),
            ("ĐÃ ĐỖ", self.ed_slots_used),
            ("CÒN LẠI", self.ed_slots_free),
        ]:
            form.addWidget(QLabel(label), r, 0); form.addWidget(widget, r, 1); r += 1
        form.addWidget(self.btn_in, r,0); form.addWidget(self.btn_out, r,1); r += 1
        form.addWidget(btn_sync, r,0); form.addWidget(btn_clear, r,1)

        box_info = QGroupBox("Thông tin"); wi = QWidget(); wi.setLayout(form)
        lay_info = QVBoxLayout(); lay_info.addWidget(wi); box_info.setLayout(lay_info)

        right = QVBoxLayout()
        right.addWidget(self.lbl_clock); right.addWidget(box_mqtt); right.addWidget(box_info); right.addStretch(1)
        panel_right = QWidget(); panel_right.setLayout(right); panel_right.setMaximumWidth(460)

        central = QWidget(); h = QHBoxLayout(central); h.addWidget(left, 2); h.addWidget(panel_right, 1)
        self.setCentralWidget(central)

        sb = QStatusBar(); self.lbl_status_cam = QLabel("Camera: —")
        sb.addWidget(self.lbl_status_cam); self.setStatusBar(sb)

        # điền các giá trị ban đầu
        self.ed_slots_total.setText(str(load_config().total_slots))
        self._update_slot_counts()
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)

    def _start_timers(self):
        self.tmr = QTimer(self); self.tmr.timeout.connect(self._tick); self.tmr.start(1000)
        self.tmr_hb = QTimer(self); self.tmr_hb.timeout.connect(self._check_esp_alive); self.tmr_hb.start(1000)

    def _group(self, title, widget):
        gb = QGroupBox(title); v = QVBoxLayout(); v.setContentsMargins(6,8,6,6); v.addWidget(widget); gb.setLayout(v)
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); return gb

    def _ro_edit(self):
        e = QLineEdit(); e.setReadOnly(True)
        e.setStyleSheet("QLineEdit{background:#2a2a2a;color:#ddd;padding:6px;border:1px solid #3a3a3a;}")
        return e

    def _count_box(self, val="0"):
        e = QLineEdit(val); e.setReadOnly(True); e.setAlignment(Qt.AlignCenter)
        e.setStyleSheet("QLineEdit{background:#39d353;color:#0a0a0a;font-size:18px;border-radius:6px;padding:6px;font-weight:700;}")
        return e

    def _tick(self):
        self.lbl_clock.setText(time.strftime("%H:%M:%S  —  %a, %d/%m/%Y"))

    # ---------- MODEL/OCR ----------
    def _init_models(self):
        try:
            self.alpr = ALPR(YOLO_MODEL_PATH, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
        except Exception as e:
            self.alpr = None
            QMessageBox.critical(self, "ALPR", f"Không khởi tạo được YOLO/EasyOCR:\n{e}")

    # ---------- CAMERA ----------
    def start_cameras(self):
        self.stop_cameras()
        if self.cfg.cam_in_index >= 0:
            try:
                self.cam_in_worker = CameraWorker(self.cfg.cam_in_index, mirror=False)
                self.cam_in_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_in, img))
                self.cam_in_worker.opened.connect(lambda ok: self._cam_status(ok, "IN", self.cfg.cam_in_index))
                self.cam_in_worker.start()
            except Exception:
                self.lbl_status_cam.setText("Camera IN: LỖI khi mở")
        else:
            self.lbl_status_cam.setText("Camera IN: tắt")

        if self.cfg.cam_out_index >= 0:
            try:
                self.cam_out_worker = CameraWorker(self.cfg.cam_out_index, mirror=False)
                self.cam_out_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_out, img))
                self.cam_out_worker.opened.connect(lambda ok: self._cam_status(ok, "OUT", self.cfg.cam_out_index))
                self.cam_out_worker.start()
            except Exception:
                self.lbl_status_cam.setText(self.lbl_status_cam.text() + " | OUT: LỖI khi mở")
        else:
            cur = self.lbl_status_cam.text()
            self.lbl_status_cam.setText((cur + " | OUT: tắt") if cur and "—" not in cur else "Camera OUT: tắt")

    def stop_cameras(self):
        if getattr(self, "cam_in_worker", None):
            try: self.cam_in_worker.stop()
            except Exception: pass
            self.cam_in_worker = None
        if getattr(self, "cam_out_worker", None):
            try: self.cam_out_worker.stop()
            except Exception: pass
            self.cam_out_worker = None

    def _cam_status(self, ok: bool, tag: str, idx: int):
        self.lbl_status_cam.setText(f"Camera {tag} (index {idx}): {'OK' if ok else 'LỖI'}")

    # ---------- SLOT / RECORD ----------
    def _update_slot_counts(self):
        used = len(getattr(self, "_in_records", {}))
        total = int(self.cfg.total_slots)
        free = max(0, total - used)
        self.ed_slots_used.setText(str(used))
        self.ed_slots_free.setText(str(free))
        self.ed_slots_total.setText(str(total))

    def _ensure_alpr(self) -> bool:
        if self.alpr is None:
            QMessageBox.warning(self, "ALPR", "Model YOLO/EasyOCR chưa sẵn sàng.")
            return False
        return True

    # ---------- ACTIONS ----------
    def on_shoot_in(self):
        if not self._ensure_alpr(): return
        if not self.cam_in_worker:
            QMessageBox.warning(self, "Chụp IN", "Chưa có camera IN."); return
        frame = self.cam_in_worker.best_recent_frame()
        if frame is None:
            QMessageBox.warning(self, "Chụp IN", "Không lấy được khung hình rõ."); return

        plate, debug = self.alpr.infer(frame)
        set_pixmap_fit_no_upscale(self.lbl_img_in, np_to_qimage(debug))
        if not plate:
            self.ed_plate.setText("Không đọc được")
            QMessageBox.information(self, "Chụp IN", "Không nhận dạng được biển số.")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(DIR_IN / f"UNREAD_{ts}.jpg"), frame)
            return

        if plate in self._in_records:
            QMessageBox.warning(self, "Chụp IN", f"Biển {plate} đã có bản ghi IN!"); return

        now = datetime.datetime.now()
        self._in_records[plate] = now
        self.ed_plate.setText(plate)
        self.ed_tin.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
        self.ed_tout.clear(); self.ed_tdiff.clear(); self.ed_fee.setText("0")
        self._update_slot_counts()

        today = now.strftime("%Y-%m-%d")
        save_dir = DIR_IN / today
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{plate}.jpg"), frame)


    def on_shoot_out(self):
        if not self._ensure_alpr(): return
        if not self.cam_out_worker:
            QMessageBox.warning(self, "Chụp OUT", "Chưa có camera OUT."); return
        frame = self.cam_out_worker.best_recent_frame()
        if frame is None:
            QMessageBox.warning(self, "Chụp OUT", "Không lấy được khung hình rõ."); return

        plate, debug = self.alpr.infer(frame)
        set_pixmap_fit_no_upscale(self.lbl_img_out, np_to_qimage(debug))
        if not plate:
            QMessageBox.information(self, "Chụp OUT", "Không nhận dạng được biển số.")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(DIR_OUT / f"UNREAD_{ts}.jpg"), frame)
            return

        self.ed_plate.setText(plate)
        now = datetime.datetime.now()
        self.ed_tout.setText(now.strftime("%Y-%m-%d %H:%M:%S"))

        if plate not in self._in_records:
            QMessageBox.warning(self, "Chụp OUT", "Không thấy bản ghi IN tương ứng. Ảnh OUT đã lưu.")
            ts = now.strftime("%Y%m%d_%H%M%S"); cv2.imwrite(str(DIR_OUT / f"{plate}_{ts}.jpg"), frame)
            return

        t_in = self._in_records.pop(plate)
        diff = now - t_in
        mins = max(1, int(diff.total_seconds() // 60))
        fee  = FEE_FLAT  # demo

        self.ed_tin.setText(t_in.strftime("%Y-%m-%d %H:%M:%S"))
        self.ed_tdiff.setText(f"{mins} phút")
        self.ed_fee.setText(f"{fee:,}")
        self._update_slot_counts()

        today = now.strftime("%Y-%m-%d")
        save_dir = DIR_OUT / today
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{plate}.jpg"), frame)

    # ---------- Misc ----------
    def on_sync(self):
        QMessageBox.information(self, "Đồng bộ", "Các chức năng nâng cao sẽ thêm sau.")

    def on_clear(self):
        for w in [self.ed_card, self.ed_plate, self.ed_tin, self.ed_tout, self.ed_tdiff, self.ed_fee]:
            try: w.clear()
            except Exception: pass
        for lbl, text in [(self.lbl_img_in,"Ảnh xe vào"), (self.lbl_img_out, "Ảnh xe ra")]:
            lbl.clear(); lbl.setText(text)

    def toggle_fullscreen(self, checked: bool):
        self.showFullScreen() if checked else self.showNormal()

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            cam_in, cam_out, slots, en_mqtt, host, port, gate, autob, bexe, bconf = dlg.values()
            if cam_in == -1:
                QMessageBox.warning(self, "Cài đặt", "Ngõ vào phải chọn 1 camera hợp lệ."); return
            self.cfg.cam_in_index = int(cam_in)
            self.cfg.cam_out_index = int(cam_out)
            self.cfg.total_slots = max(1, slots)
            self.cfg.mqtt_enable = bool(en_mqtt)
            self.cfg.mqtt_host   = host
            self.cfg.mqtt_port   = int(port)
            self.cfg.gate_id     = gate
            self.cfg.auto_start_broker = bool(autob)
            self.cfg.broker_exe  = bexe
            self.cfg.broker_conf = bconf
            save_config(self.cfg)

            self.start_cameras()
            self._update_slot_counts()
            self.restart_mqtt()
            self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
            self.lbl_mqtt_gate.setText(self.cfg.gate_id)

    # ---------- MQTT/BROKER ----------
    def _set_mqtt_state(self, text, color="#bbb"):
        self.lbl_mqtt_state.setText(text)
        self.lbl_mqtt_state.setStyleSheet(f"color:{color};font-weight:700;")

    def _refresh_conn_badge(self):
        mqtt_txt = "Đã kết nối" if self._mqtt_connected else "Mất kết nối"
        esp_txt  = "Online" if self._esp_online else "Offline"
        color = "#39d353" if (self._mqtt_connected and self._esp_online) else ("#f1c40f" if self._mqtt_connected else "#ff6b6b")
        self._set_mqtt_state(f"MQTT: {mqtt_txt} — ESP32: {esp_txt}", color)

    def _check_esp_alive(self):
        if not self._mqtt_connected:
            if self._esp_online:
                self._esp_online = False; self._refresh_conn_badge()
            return
        if self._esp_last_hb <= 0: return
        if (time.time() - self._esp_last_hb) >  self._hb_timeout and self._esp_online:
            self._esp_online = False; self._refresh_conn_badge()

    def ensure_broker_running(self):
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)

        if not (self.cfg.mqtt_enable and self.cfg.auto_start_broker):
            return
        host = (self.cfg.mqtt_host or "").strip()
        local_ips = self._local_ips or get_local_ips()
        if host not in local_ips:  # chỉ start khi broker là máy này
            return

        probe_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
        if is_port_open(probe_host, self.cfg.mqtt_port):
            return

        exe, conf = self.cfg.broker_exe, self.cfg.broker_conf
        if not os.path.exists(exe) or not os.path.exists(conf):
            self._set_mqtt_state("Không thấy mosquitto/conf", "#ff6b6b"); return

        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            exe_dir = os.path.dirname(exe) or None
            self._mosq_proc = subprocess.Popen(
                [exe, "-c", conf],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                creationflags=flags, cwd=exe_dir
            )
            self._set_mqtt_state("Đang khởi động broker…", "#f1c40f")
        except Exception as e:
            self._set_mqtt_state(f"Lỗi chạy broker: {e}", "#ff6b6b")
            self._mosq_proc = None

    def init_mqtt(self):
        if not self.cfg.mqtt_enable or mqtt is None:
            self._mqtt_connected = False; self._esp_online = False
            self._set_mqtt_state("OFF", "#bbb"); return
        try:
            cid = "ui-" + "".join(random.choice(string.ascii_lowercase+string.digits) for _ in range(6))
            self.lbl_mqtt_cid.setText(cid)
            self.mqtt_client = mqtt.Client(client_id=cid)
            try: self.mqtt_client.reconnect_delay_set(min_delay=0.5, max_delay=3)
            except Exception: pass

            def _on_connect(client, userdata, flags, rc, properties=None):
                self._mqtt_connected = (rc == 0)
                if rc == 0:
                    client.subscribe(f"parking/gate/{self.cfg.gate_id}/event", qos=1)
                    client.subscribe(f"parking/gate/{self.cfg.gate_id}/stats", qos=1)
                    client.subscribe(f"parking/gate/{self.cfg.gate_id}/status", qos=1)
                    client.subscribe(f"parking/gate/{self.cfg.gate_id}/heartbeat", qos=0)
                else:
                    self._esp_online = False
                self._refresh_conn_badge()

            def _on_disconnect(client, userdata, rc, properties=None):
                self._mqtt_connected = False
                self._esp_online = False
                self._refresh_conn_badge()

            def _on_message(client, userdata, msg):
                try:
                    topic = msg.topic
                    payload = {}
                    try:
                        payload = json.loads(msg.payload.decode("utf-8"))
                    except Exception:
                        pass

                    if topic.endswith("/status"):
                        online = bool(payload.get("online", False))
                        self._esp_online = online
                        if online:
                            self._esp_last_hb = time.time()
                        self._refresh_conn_badge()

                    elif topic.endswith("/heartbeat"):
                        self._esp_last_hb = time.time()
                        if not self._esp_online:
                            self._esp_online = True
                            self._refresh_conn_badge()

                    # TODO: xử lý thêm event/stats nếu cần
                except Exception:
                    pass

            self.mqtt_client.on_connect = _on_connect
            self.mqtt_client.on_disconnect = _on_disconnect
            self.mqtt_client.on_message = _on_message

            self._set_mqtt_state("Đang kết nối…", "#f1c40f")
            self.mqtt_client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=20)
            self.mqtt_client.loop_start()
        except Exception as e:
            self._mqtt_connected = False
            self._esp_online = False
            self._set_mqtt_state(f"Lỗi MQTT: {e}", "#ff6b6b")

    def restart_mqtt(self):
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception:
            pass
        self.mqtt_client = None
        self._mqtt_connected = False
        self._esp_online = False
        self._refresh_conn_badge()
        self.ensure_broker_running()
        self.init_mqtt()

    # ---------- Close ---------- 
    def closeEvent(self, e):
        self.stop_cameras()
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception:
            pass
        if self._mosq_proc:
            try: self._mosq_proc.terminate()
            except Exception: pass
        super().closeEvent(e)

# ============================= BOOT ===============================

def main():
    cfg = load_config()
    # đảm bảo file conf mặc định tồn tại
    if not os.path.exists(cfg.broker_conf):
        Path(cfg.broker_conf).parent.mkdir(parents=True, exist_ok=True)
        open(cfg.broker_conf, "w", encoding="utf-8").write(
            "listener 1883 0.0.0.0\nallow_anonymous true\npersistence false\n"
        )
    app = QApplication(sys.argv)
    w = MainWindow(cfg)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
