# parking_ui.py
# Full application file with EasyOCR + robust detect + fuzzy plate matching

# -*- coding: utf-8 -*-
import sys, time, json, os, threading, collections, random, string, datetime, csv, re, subprocess, shutil, difflib
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLineEdit, QPushButton, QStatusBar,
    QMessageBox, QSizePolicy, QDialog, QComboBox, QDialogButtonBox,
    QFormLayout, QCheckBox, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QFrame, QFileDialog, QInputDialog
)

import cv2
import numpy as np

# Optional MQTT
try:
    from paho.mqtt import client as mqtt
except Exception:
    mqtt = None

# Optional Tesseract OCR
try:
    import pytesseract
except Exception:
    pytesseract = None

# Optional EasyOCR
try:
    import easyocr
except Exception:
    easyocr = None

# ======================== CONFIG ========================
CFG_FILE = "config.json"

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
            return UiConfig(
                cam_in_index=int(d.get("cam_in_index", 0)),
                cam_out_index=int(d.get("cam_out_index", -1)),
                total_slots=int(d.get("total_slots", 50)),
                mqtt_enable=bool(d.get("mqtt_enable", True)),
                mqtt_host=str(d.get("mqtt_host", "127.0.0.1")),
                mqtt_port=int(d.get("mqtt_port", 1883)),
                gate_id=str(d.get("gate_id", "gate1")),
                auto_start_broker=bool(d.get("auto_start_broker", True)),
                broker_exe=str(d.get("broker_exe", r"C:\Program Files\mosquitto\mosquitto.exe")),
                broker_conf=str(d.get("broker_conf", r"E:\FIRMWAVE\project\mosquitto.conf")),
            )
        except Exception:
            pass
    return UiConfig()

def save_config(cfg: UiConfig):
    with open(CFG_FILE, "w", encoding="utf-8") as fh:
        json.dump(cfg.__dict__, fh, ensure_ascii=False, indent=2)

PLATES_DIR = Path("plates")
PLATES_DIR.mkdir(exist_ok=True)

# ======================== IMAGE/VIDEO HELPERS =====================
def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def enhance_for_plate(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    opened = Signal(bool)
    def __init__(self, source=0, width=640, height=0, mirror=False, parent=None):
        super().__init__(parent)
        self.source, self.width, self.height, self.mirror = source, width, height, mirror
        self._running, self.cap = False, None
        self._buf_lock = threading.Lock()
        self._buffer = collections.deque(maxlen=25)
    def run(self):
        self._running = True
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        try:
            self.cap = cv2.VideoCapture(self.source, backend)
            if self.width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok = self.cap.isOpened(); self.opened.emit(ok)
            if not ok:
                return
            target_dt = 1/25.0
            last_emit = 0.0
            while self._running:
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    QThread.msleep(50)
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                try:
                    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                    score = sharpness_score(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                except Exception:
                    score = 0.0
                with self._buf_lock:
                    self._buffer.append((score, frame.copy()))
                if time.time() - last_emit >= target_dt:
                    try:
                        display_w = min(self.width if self.width else 640, 640)
                        h0, w0 = frame.shape[:2]
                        if w0 > display_w:
                            scale = display_w / w0
                            disp = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
                        else:
                            disp = frame
                        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
                        self.frame_ready.emit(img)
                    except Exception:
                        pass
                    last_emit = time.time()
                elapsed = time.time() - t0
                rem = target_dt - elapsed
                if rem > 0:
                    QThread.msleep(int(rem * 1000))
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
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception:
            pass
        self.wait(1500)
    def best_recent_frame(self, min_score: float = 120.0) -> Optional[np.ndarray]:
        with self._buf_lock:
            if not self._buffer: return None
            s, f = max(self._buffer, key=lambda t: t[0])
            return f.copy() if s >= min_score else None

def qlabel_video_placeholder(text=""):
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lbl.setMinimumSize(QSize(360, 240))
    lbl.setStyleSheet("QLabel{background:#1f1f1f;color:#cccccc;border:1px solid #3a3a3a;}")
    return lbl

def set_pixmap_fit_no_upscale(label: QLabel, img: QImage):
    if label.width() <= 0 or label.height() <= 0: return
    pix = QPixmap.fromImage(img)
    if pix.isNull(): return
    sw, sh = label.width()/pix.width(), label.height()/pix.height()
    scale = min(1.0, sw, sh)
    new_size = QSize(int(pix.width()*scale), int(pix.height()*scale))
    scaled = pix.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label.setAlignment(Qt.AlignCenter); label.setScaledContents(False); label.setPixmap(scaled)

def np_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()

def list_cameras(max_index=8) -> List[int]:
    found=[]; backend = cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened(): found.append(i); cap.release()
    return found

 
# ---------------- Robust plate detection (replace older contour detect) ----------------
def detect_plate_robust(frame, min_area=600, aspect_min=2.0, aspect_max=6.5):
    h_img, w_img = frame.shape[:2]

    def find_from_edges(img_gray, low, high):
        blur = cv2.GaussianBlur(img_gray, (5,5), 0)
        edged = cv2.Canny(blur, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best=None; best_area=0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            if area < min_area or area > 0.9*w_img*h_img: continue
            aspect = (w/h) if h>0 else 0
            if aspect_min <= aspect <= aspect_max:
                if area > best_area:
                    best_area=area; best=(x,y,w,h)
        return best

    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best = find_from_edges(gray0, 50, 200)
    if best:
        x,y,w,h = best
        mx = int(0.06*w); my = int(0.10*h)
        x0 = max(0, x-mx); y0 = max(0, y-my); x1 = min(w_img, x+w+mx); y1 = min(h_img, y+h+my)
        crop = frame[y0:y1, x0:x1]
        if crop.shape[0] > 20 and crop.shape[1] > 30:
            return crop, (x0,y0,x1-x0,y1-y0)

    tries = [(30,150),(50,200),(80,200),(10,180)]
    for low,high in tries:
        best = find_from_edges(gray0, low, high)
        if best:
            x,y,w,h = best
            mx = int(0.06*w); my = int(0.10*h)
            x0 = max(0, x-mx); y0 = max(0, y-my); x1 = min(w_img, x+w+mx); y1 = min(h_img, y+h+my)
            crop = frame[y0:y1, x0:x1]
            if crop.shape[0] > 20 and crop.shape[1] > 30:
                return crop, (x0,y0,x1-x0,y1-y0)

    try:
        thr = cv2.adaptiveThreshold(gray0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best=None; best_area=0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt); area=w*h
            if area < min_area or area > 0.9*w_img*h_img: continue
            aspect = (w/h) if h>0 else 0
            if aspect_min <= aspect <= aspect_max:
                if area > best_area:
                    best_area=area; best=(x,y,w,h)
        if best:
            x,y,w,h = best
            mx = int(0.06*w); my = int(0.10*h)
            x0 = max(0, x-mx); y0 = max(0, y-my); x1 = min(w_img, x+w+mx); y1 = min(h_img, y+h+my)
            crop = frame[y0:y1, x0:x1]
            if crop.shape[0] > 20 and crop.shape[1] > 30:
                return crop, (x0,y0,x1-x0,y1-y0)
    except Exception:
        pass

    try:
        cw = int(w_img * 0.6)
        ch = int(h_img * 0.25)
        cx0 = (w_img - cw)//2
        cy0 = int(h_img * 0.55)
        crop = frame[cy0: min(h_img, cy0+ch), cx0: cx0+cw]
        if crop.size and crop.shape[0] > 20 and crop.shape[1] > 30:
            return crop, (cx0, cy0, cw, crop.shape[0])
    except Exception:
        pass

    return None, None

# ---------------- OCR helpers (Tesseract variants) ----------------
def preprocess_for_ocr(bgr_crop):
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    h, w = g.shape
    long_side = max(h,w)
    if long_side < 320:
        scale = 320.0 / long_side
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th

def ocr_with_tesseract(img_bin):
    if pytesseract is None:
        return "", 0
    try:
        cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
        data = pytesseract.image_to_data(img_bin, config=cfg, output_type=pytesseract.Output.DICT)
        texts=[]; confs=[]
        n = len(data.get('text', []))
        for i in range(n):
            t = str(data['text'][i]).strip()
            try:
                c = int(data['conf'][i])
            except:
                c = -1
            if t and c > -1:
                texts.append(t); confs.append(c)
        if not texts: return "", 0
        txt = " ".join(texts).upper()
        txt = re.sub(r'[^0-9A-Z\-\.\s]', '', txt).strip()
        avg = int(sum(confs)/len(confs)) if confs else 0
        return txt, avg
    except Exception:
        return "", 0

def _pre_variants(img_color):
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    med = cv2.medianBlur(clahe, 3)
    yield "clahe_med_adapt", med
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    yield "sharp", sharp
    bil = cv2.bilateralFilter(gray, 9, 75, 75)
    equ = cv2.equalizeHist(bil)
    yield "bilateral_eq", equ
    cla = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    yield "clahe2", cla
    h, w = gray.shape
    long = max(h,w)
    scale = 1.0
    if long < 600:
        scale = 600.0/long
    if scale != 1.0:
        big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        big = gray
    try:
        th = cv2.adaptiveThreshold(big,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        yield "big_adapt", th
    except Exception:
        pass

def _tesseract_data_for(img_gray, psm):
    if pytesseract is None:
        return "", 0, {}
    cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
    try:
        data = pytesseract.image_to_data(img_gray, config=cfg, output_type=pytesseract.Output.DICT)
        texts=[]; confs=[]
        n = len(data.get('text', []))
        for i in range(n):
            t = str(data['text'][i]).strip()
            try:
                c = int(data['conf'][i])
            except:
                c = -1
            if t and c > -1:
                texts.append(t); confs.append(c)
        if not texts:
            return "", 0, data
        txt = " ".join(texts).upper()
        txt = re.sub(r'[^0-9A-Z\-\.\s]', '', txt).strip()
        avg = int(sum(confs)/len(confs)) if confs else 0
        return txt, avg, data
    except Exception:
        return "", 0, {}

def ocr_best_of_variants(img_color) -> Tuple[str,int, np.ndarray]:
    if pytesseract is None:
        return "", 0, None
    best_text = ""
    best_conf = -1
    best_img = None
    psm_candidates = [7, 6, 11, 3]
    for name, gray in _pre_variants(img_color):
        for psm in psm_candidates:
            txt, conf, data = _tesseract_data_for(gray, psm)
            if txt and conf >= 0:
                if conf > best_conf or (conf == best_conf and len(txt) > len(best_text)):
                    best_text = txt; best_conf = conf
                    if len(gray.shape) == 2:
                        best_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        best_img = gray.copy()
            if not txt:
                try:
                    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                    m = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker)
                    txt2, conf2, _ = _tesseract_data_for(m, psm)
                    if txt2 and conf2 >= 0 and conf2 > best_conf:
                        best_text = txt2; best_conf = conf2
                        best_img = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) if len(m.shape)==2 else m.copy()
                except Exception:
                    pass
    if best_conf < 0:
        try:
            h,w = img_color.shape[:2]
            long = max(h,w)
            scale = 1.0
            if long < 800:
                scale = 800.0/long
            raw = cv2.resize(img_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            txt, conf, _ = _tesseract_data_for(gray, 7)
            if txt:
                best_text, best_conf, best_img = txt, conf, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass
    if best_conf < 0:
        return "", 0, None
    return best_text.strip(), int(best_conf), best_img

# ---------------- EasyOCR helpers (lazy init) ----------------
_EASY_OCR_READER = None
def _init_easyocr(gpu: bool = False, lang_list: list = ["en"]):
    global _EASY_OCR_READER
    if easyocr is None:
        return None
    if _EASY_OCR_READER is None:
        try:
            _EASY_OCR_READER = easyocr.Reader(lang_list, gpu=gpu)
            print("[EasyOCR] initialized (gpu=%s)" % gpu)
        except Exception as e:
            print("[EasyOCR] init error:", e)
            _EASY_OCR_READER = None
    return _EASY_OCR_READER

def ocr_with_easyocr(img_color: np.ndarray, lang_list: list = ["en"]) -> tuple:
    if easyocr is None:
        return "", 0, None
    reader = _init_easyocr(gpu=False, lang_list=lang_list)
    if reader is None:
        return "", 0, None
    try:
        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        results = reader.readtext(img_rgb, detail=1, paragraph=False)
        if not results:
            return "", 0, None
        texts, confs = [], []
        for bbox, txt, prob in results:
            t = str(txt).strip().upper()
            if t:
                t = re.sub(r'[^0-9A-Z\-\.\s]', '', t)
                texts.append(t)
                try:
                    c = int(prob * 100)
                except:
                    try: c = int(float(prob)*100)
                    except: c = 0
                confs.append(max(0, min(100, c)))
        if not texts:
            return "", 0, None
        joined = " ".join(texts).strip()
        avg = int(sum(confs)/len(confs)) if confs else 0
        preview = img_color.copy()
        try:
            bbox0 = results[0][0]
            pts = np.array(bbox0, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(preview, [pts], True, (0,255,0), 2)
        except Exception:
            pass
        return joined, avg, preview
    except Exception as e:
        print("[EasyOCR] read error:", e)
        return "", 0, None

# ======================== BROKER HELPERS ==========================
def is_port_open(host: str, port: int, timeout=0.5) -> bool:
    try:
        with __import__("socket").create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def get_local_ips() -> set:
    ips = {"127.0.0.1", "localhost", "0.0.0.0"}
    try:
        hostname = __import__("socket").gethostname()
        for ip in __import__("socket").gethostbyname_ex(hostname)[2]:
            ips.add(ip)
    except:
        pass
    try:
        s = __import__("socket").socket(__import__("socket").AF_INET, __import__("socket").SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except:
        pass
    return ips

# ================= Normalization & fuzzy matching for plates =================
def normalize_plate(s: str) -> str:
    if not s:
        return ""
    s = str(s).upper().strip()
    s = re.sub(r'[\s\.\,_:]', '', s)
    s = re.sub(r'[^0-9A-Z\-]', '', s)
    s = re.sub(r'\-+', '-', s)
    return s

# ---------------- Post-processing for Vietnamese plates ----------------
def _fix_confusable_characters(s: str) -> str:
    if not s:
        return ""
    mapping = {
        'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2',
        'S': '5', 'B': '8', 'G': '6', 'T': '7'
    }
    out = []
    for ch in s:
        out.append(mapping.get(ch, ch))
    return ''.join(out)

def _strip_to_allowed(s: str) -> str:
    s = s.upper()
    s = re.sub(r'[^0-9A-Z\-\.\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def format_vn_plate(raw: str) -> str:
    if not raw:
        return ""
    s = _strip_to_allowed(raw).replace(' ', '')
    s = _fix_confusable_characters(s)
    s = re.sub(r'[\.-]+', '', s)
    m = re.match(r'^(\d{2}[A-Z]{1,2})(\d{5})$', s)
    if m:
        head, tail = m.groups()
        return f"{head}-{tail[:3]}.{tail[3:]}"
    m = re.match(r'^(\d{2}[A-Z]{1,2})(\d{4})$', s)
    if m:
        head, tail = m.groups()
        return f"{head}-{tail}"
    s2 = _strip_to_allowed(raw).replace(' ', '')
    s2 = _fix_confusable_characters(s2)
    s2 = re.sub(r'\-+', '-', s2)
    s2 = re.sub(r'\.+', '.', s2)
    m = re.match(r'^(\d{2}[A-Z]{1,2})[-\.]?(\d{3})[\.]?(\d{2})$', s2)
    if m:
        a, b, c = m.groups()
        return f"{a}-{b}.{c}"
    return normalize_plate(raw)

def postprocess_plate_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    txt = raw_text.upper().strip()
    txt = re.sub(r'[^0-9A-Z\-\.\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    tokens = [t for t in txt.split(' ') if t]
    candidate = ''.join(tokens)
    formatted = format_vn_plate(candidate)
    if re.match(r'^\d{2}[A-Z]{1,2}', formatted or ''):
        return formatted
    return normalize_plate(candidate)

def plates_match(plate_a: str, plate_b: str) -> bool:
    if not plate_a or not plate_b:
        return False
    a = normalize_plate(plate_a)
    b = normalize_plate(plate_b)
    if not a or not b:
        return False
    if a == b:
        return True
    nums_a = re.findall(r'\d+', a)
    nums_b = re.findall(r'\d+', b)
    if nums_a and nums_b:
        match_numeric = any(na == nb for na in nums_a for nb in nums_b)
        if not match_numeric:
            return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= 0.75

# =================== Plate history dialog (search) ===================
class PlateHistoryDialog(QDialog):
    def __init__(self, parent, plate: str):
        super().__init__(parent)
        self.setWindowTitle(f"Lịch sử biển: {plate}")
        self.resize(900, 600)
        self.plate = plate.upper()
        self.listw = QListWidget(); self.listw.setMinimumWidth(420)
        self.lbl_img = QLabel("Chọn một bản ghi để xem ảnh"); self.lbl_img.setAlignment(Qt.AlignCenter); self.lbl_img.setMinimumSize(320,240)
        self.lbl_img.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.txt_info = QTextEdit(); self.txt_info.setReadOnly(True)
        btn_open = QPushButton("Mở ảnh bằng ứng dụng mặc định"); btn_open.clicked.connect(self.open_selected_image)
        right_layout = QVBoxLayout(); right_layout.addWidget(self.lbl_img); right_layout.addWidget(self.txt_info); right_layout.addWidget(btn_open)
        splitter = QSplitter(Qt.Horizontal)
        left_wrap = QWidget(); left_l = QVBoxLayout(left_wrap); left_l.addWidget(self.listw)
        splitter.addWidget(left_wrap)
        right_wrap = QWidget(); right_wrap.setLayout(right_layout)
        splitter.addWidget(right_wrap)
        layout = QVBoxLayout(self); layout.addWidget(splitter)
        self.records = self._load_records_for_plate(self.plate)
        if not self.records:
            self.listw.addItem("Không tìm thấy bản ghi cho biển: " + plate)
        else:
            for rec in self.records:
                date = rec.get("date", ""); action = rec.get("action", ""); fee = rec.get("fee", ""); img = rec.get("img", "")
                display = f"{date} — {action} — Phí: {fee} — {os.path.basename(img)}"
                it = QListWidgetItem(display); it.setData(Qt.UserRole, rec); self.listw.addItem(it)
            self.listw.currentItemChanged.connect(self._on_item_changed); self.listw.setCurrentRow(0)

    def _load_records_for_plate(self, plate):
        csvf = PLATES_DIR / "plates.csv"; out = []
        if not csvf.exists(): return out
        try:
            with open(csvf, "r", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                for row in rdr:
                    p = (row.get("plate") or "").strip().upper()
                    if p == plate:
                        out.append(row)
        except Exception:
            pass
        return out

    def _on_item_changed(self, cur, prev=None):
        if cur is None: return
        rec = cur.data(Qt.UserRole); img_path = rec.get("img", ""); info_lines = []
        for k,v in rec.items(): info_lines.append(f"{k}: {v}")
        self.txt_info.setText("\n".join(info_lines))
        try:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    qimg = np_to_qimage(img)
                    set_pixmap_fit_no_upscale(self.lbl_img, qimg)
                    return
        except Exception:
            pass
        self.lbl_img.setText("Không mở được ảnh")

    def open_selected_image(self):
        it = self.listw.currentItem()
        if it is None: return
        rec = it.data(Qt.UserRole); img_path = rec.get("img", "")
        if not img_path or not os.path.exists(img_path):
            QMessageBox.warning(self, "Mở ảnh", "Không tìm thấy tập tin ảnh."); return
        try:
            if os.name == "nt":
                os.startfile(img_path)
            else:
                subprocess.Popen(["xdg-open", img_path])
        except Exception as e:
            QMessageBox.warning(self, "Mở ảnh", f"Lỗi mở ảnh: {e}")

# ======================== SETTINGS DIALOG =========================
class SettingsDialog(QDialog):
    def __init__(self, cfg: UiConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt"); self.resize(480, 360)
        cams = list_cameras()
        self.cb_in  = QComboBox(); self.cb_out = QComboBox(); self.cb_out.addItem("— Tắt —", -1)
        if not cams:
            self.cb_in.addItem("Không tìm thấy camera", -1)
        else:
            for i in cams:
                self.cb_in.addItem(f"Camera {i}", i); self.cb_out.addItem(f"Camera {i}", i)
        if cams and cfg.cam_in_index in cams:
            self.cb_in.setCurrentIndex(cams.index(cfg.cam_in_index))
        if cfg.cam_out_index == -1:
            self.cb_out.setCurrentIndex(0)
        elif cfg.cam_out_index in cams:
            self.cb_out.setCurrentIndex(1 + cams.index(cfg.cam_out_index))
        self.ed_slots  = QLineEdit(str(cfg.total_slots))
        self.chk_mqtt  = QCheckBox("Bật MQTT"); self.chk_mqtt.setChecked(cfg.mqtt_enable)
        self.ed_host   = QLineEdit(cfg.mqtt_host); self.ed_port   = QLineEdit(str(cfg.mqtt_port)); self.ed_gate   = QLineEdit(cfg.gate_id)
        self.chk_autob = QCheckBox("Tự khởi động Mosquitto (nếu host là máy này)"); self.chk_autob.setChecked(cfg.auto_start_broker)
        self.ed_bexe   = QLineEdit(cfg.broker_exe); self.ed_bconf  = QLineEdit(cfg.broker_conf)
        form = QFormLayout()
        form.addRow("Ngõ vào:", self.cb_in); form.addRow("Ngõ ra:", self.cb_out); form.addRow("SLOT TỔNG:", self.ed_slots)
        form.addRow(self.chk_mqtt); form.addRow("MQTT Host:", self.ed_host); form.addRow("MQTT Port:", self.ed_port)
        form.addRow("Gate ID:", self.ed_gate); form.addRow(self.chk_autob); form.addRow("mosquitto.exe:", self.ed_bexe); form.addRow("mosquitto.conf:", self.ed_bconf)
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
        self._records = []
        self.setWindowTitle("Phần mềm quản lý bãi gửi xe"); self.resize(1280, 760)

        act_settings = QAction("Thiết lập", self); act_settings.triggered.connect(self.open_settings)
        act_full = QAction("Toàn màn hình", self, checkable=True); act_full.triggered.connect(self.toggle_fullscreen)
        menubar = self.menuBar(); menu = menubar.addMenu("Cài đặt"); menu.addAction(act_settings); menu.addAction(act_full)

        self.lbl_cam_in  = qlabel_video_placeholder(); self.lbl_img_in  = qlabel_video_placeholder("Ảnh xe vào")
        self.lbl_cam_out = qlabel_video_placeholder(); self.lbl_img_out = qlabel_video_placeholder("Ảnh xe ra")

        grid = QGridLayout()
        grid.addWidget(self._group("Camera ngõ vào", self.lbl_cam_in), 0, 0)
        grid.addWidget(self._group("Ảnh xe vào", self.lbl_img_in),     0, 1)
        grid.addWidget(self._group("Camera ngõ ra", self.lbl_cam_out), 1, 0)
        grid.addWidget(self._group("Ảnh xe ra", self.lbl_img_out),     1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        left = QWidget(); left.setLayout(grid)

        self.lbl_clock = QLabel("--:--:--"); self.lbl_clock.setAlignment(Qt.AlignCenter); self.lbl_clock.setStyleSheet("font-size:22px;font-weight:600;")
        self.lbl_mqtt_state = QLabel("OFF"); self.lbl_mqtt_state.setStyleSheet("color:#bbb;font-weight:700;")
        self.lbl_mqtt_broker = QLabel("-"); self.lbl_mqtt_gate   = QLabel("-"); self.lbl_mqtt_cid    = QLabel("-")
        mqtt_form = QFormLayout(); mqtt_form.addRow("Trạng thái:", self.lbl_mqtt_state); mqtt_form.addRow("Broker:", self.lbl_mqtt_broker); mqtt_form.addRow("Gate ID:", self.lbl_mqtt_gate); mqtt_form.addRow("Client ID:", self.lbl_mqtt_cid)
        box_mqtt = QGroupBox("Kết nối MQTT / ESP32"); w_mqtt = QWidget(); w_mqtt.setLayout(mqtt_form); lay_mqtt = QVBoxLayout(); lay_mqtt.addWidget(w_mqtt); box_mqtt.setLayout(lay_mqtt)

        self.ed_plate_cnt = self._count_box("0"); self.ed_card  = self._ro_edit(); self.ed_plate = self._ro_edit()
        self.ed_tin  = self._ro_edit(); self.ed_tout  = self._ro_edit(); self.ed_tdiff = self._ro_edit(); self.ed_fee = self._ro_edit()
        self.ed_slots_total = self._ro_edit(); self.ed_slots_total.setText(str(self.cfg.total_slots))
        self.ed_slots_used  = self._ro_edit(); self.ed_slots_free  = self._ro_edit()
        self._refresh_slot_labels()

        btn_sync = QPushButton("Đồng bộ"); btn_sync.clicked.connect(self.on_sync)
        btn_capture_in = QPushButton("Chụp IN"); btn_capture_in.clicked.connect(self.on_capture_in)
        btn_capture_out = QPushButton("Chụp OUT"); btn_capture_out.clicked.connect(self.on_capture_out)
        btn_manual = QPushButton("Nhập biển"); btn_manual.clicked.connect(self.on_manual_entry)
        btn_clear = QPushButton("Xóa"); btn_clear.clicked.connect(self.on_clear)
        self.ed_search_plate = QLineEdit(); self.ed_search_plate.setPlaceholderText("Nhập biển (VD: 30A12345)")
        btn_search = QPushButton("Tìm biển"); btn_search.clicked.connect(self.on_search_plate)

        form = QGridLayout(); r=0
        form.addWidget(QLabel("SỐ XE"), r,0); form.addWidget(self.ed_plate_cnt, r,1); r+=1
        form.addWidget(QLabel("MÃ THẺ"), r,0); form.addWidget(self.ed_card, r,1); r+=1
        form.addWidget(QLabel("BIỂN SỐ"), r,0); form.addWidget(self.ed_plate, r,1); r+=1
        form.addWidget(QLabel("T/G XE VÀO"), r,0); form.addWidget(self.ed_tin, r,1); r+=1
        form.addWidget(QLabel("T/G XE RA"), r,0);  form.addWidget(self.ed_tout, r,1); r+=1
        form.addWidget(QLabel("T/G GỬI XE"), r,0); form.addWidget(self.ed_tdiff, r,1); r+=1
        form.addWidget(QLabel("PHÍ GỬI XE"), r,0); form.addWidget(self.ed_fee, r,1); r+=1
        form.addWidget(QLabel("SLOT TỔNG"), r,0);  form.addWidget(self.ed_slots_total, r,1); r+=1
        form.addWidget(QLabel("ĐÃ ĐỖ"),     r,0);  form.addWidget(self.ed_slots_used,  r,1); r+=1
        form.addWidget(QLabel("CÒN LẠI"),   r,0);  form.addWidget(self.ed_slots_free,  r,1); r+=1
        form.addWidget(btn_sync, r,0); form.addWidget(btn_capture_in, r,1); r+=1
        form.addWidget(btn_capture_out, r,0); form.addWidget(btn_manual, r,1); r+=1
        form.addWidget(btn_clear, r,0); form.addWidget(QLabel(""), r,1); r+=1
        form.addWidget(QLabel("TÌM BIỂN:"), r,0); form.addWidget(self.ed_search_plate, r,1); r+=1
        form.addWidget(btn_search, r,0); r+=1

        box_info = QGroupBox("Thông tin"); w_info = QWidget(); w_info.setLayout(form)
        lay_info = QVBoxLayout(); lay_info.addWidget(w_info); box_info.setLayout(lay_info)

        right = QVBoxLayout()
        right.addWidget(self.lbl_clock)
        right.addWidget(box_mqtt)
        right.addWidget(box_info)
        right.addStretch(1)
        panel_right = QWidget(); panel_right.setLayout(right); panel_right.setMaximumWidth(420)

        central = QWidget()
        h = QHBoxLayout(central); h.addWidget(left, 2); h.addWidget(panel_right, 1)
        self.setCentralWidget(central)

        sb = QStatusBar(); self.lbl_status_cam = QLabel("Camera: —"); sb.addWidget(self.lbl_status_cam); self.setStatusBar(sb)
        self.tmr = QTimer(self); self.tmr.timeout.connect(self._tick); self.tmr.start(1000)

        self.cam_in_worker: Optional[CameraWorker] = None
        self.cam_out_worker: Optional[CameraWorker] = None

        self.start_cameras()

        self._mosq_proc = None; self.mqtt_client = None; self._local_ips = get_local_ips()
        self._mqtt_connected = False; self._esp_online = False; self._esp_last_hb = 0.0; self._hb_timeout = 10.0
        self.tmr_hb = QTimer(self); self.tmr_hb.timeout.connect(self._check_esp_alive); self.tmr_hb.start(1000)

        self.ensure_broker_running(); self.init_mqtt()

    def center_on_screen(self):
        try:
            pass
        except Exception:
            pass

    # UI helpers
    def _group(self, title, widget):
        gb = QGroupBox(title); v = QVBoxLayout(); v.setContentsMargins(6,8,6,6); v.addWidget(widget); gb.setLayout(v)
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); return gb
    def _ro_edit(self):
        e = QLineEdit(); e.setReadOnly(True); e.setStyleSheet("QLineEdit{background:#2a2a2a;color:#ddd;padding:6px;border:1px solid #3a3a3a;}"); return e
    def _count_box(self, val="0"):
        e = QLineEdit(val); e.setReadOnly(True); e.setAlignment(Qt.AlignCenter)
        e.setStyleSheet("QLineEdit{background:#39d353;color:#0a0a0a;font-size:18px;border-radius:6px;padding:6px;font-weight:700;}"); return e
    def _tick(self):
        self.lbl_clock.setText(time.strftime("%H:%M:%S  —  %a, %d/%m/%Y"))

    # Cameras
    def start_cameras(self):
        self.stop_cameras()
        if self.cfg.cam_in_index >= 0:
            self.cam_in_worker = CameraWorker(self.cfg.cam_in_index, mirror=False)
            self.cam_in_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_in, img))
            self.cam_in_worker.opened.connect(lambda ok: self._cam_status(ok, "IN", self.cfg.cam_in_index))
            self.cam_in_worker.start()
        else:
            self.lbl_status_cam.setText("Camera IN: tắt")
        if self.cfg.cam_out_index >= 0:
            self.cam_out_worker = CameraWorker(self.cfg.cam_out_index, mirror=False)
            self.cam_out_worker.frame_ready.connect(lambda img: set_pixmap_fit_no_upscale(self.lbl_cam_out, img))
            self.cam_out_worker.opened.connect(lambda ok: self._cam_status(ok, "OUT", self.cfg.cam_out_index))
            self.cam_out_worker.start()
        else:
            self.lbl_status_cam.setText(self.lbl_status_cam.text() + " | OUT: tắt")
    def stop_cameras(self):
        if self.cam_in_worker:  self.cam_in_worker.stop();  self.cam_in_worker = None
        if self.cam_out_worker: self.cam_out_worker.stop(); self.cam_out_worker = None
        try:
            self._tmr_detect_in.stop(); self._tmr_render_in.stop()
        except Exception:
            pass
        try:
            self._tmr_detect_out.stop(); self._tmr_render_out.stop()
        except Exception:
            pass
    def _cam_status(self, ok: bool, tag: str, idx: int):
        self.lbl_status_cam.setText(f"Camera {tag} (index {idx}): {'OK' if ok else 'LỖI'}")

    # Slots
    def _refresh_slot_labels(self):
        total = self.cfg.total_slots; used = 0
        for r in self._records:
            if r.get('in_ts') and not r.get('out_ts'):
                used += 1
        free = total - used
        self.ed_slots_used.setText(str(used)); self.ed_slots_free.setText(str(free)); self.ed_slots_total.setText(str(total))

    def on_sync(self):
        QMessageBox.information(self, "Đồng bộ", "Các chức năng nâng cao sẽ thêm sau.")
    def on_clear(self):
        for w in [self.ed_card, self.ed_plate, self.ed_tin, self.ed_tout, self.ed_tdiff, self.ed_fee, self.ed_search_plate]:
            try: w.clear()
            except: pass
        for lbl, text in [(self.lbl_img_in,"Ảnh xe vào"), (self.lbl_img_out, "Ảnh xe ra")]:
            lbl.clear(); lbl.setText(text)

    # Image save & records
    def _ensure_plates_dir(self, action: str = "IN") -> Path:
        act = (action or "IN").upper()
        sub = "IN" if act == "IN" else "OUT"
        d = PLATES_DIR / time.strftime("%Y%m%d") / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_image_and_record(self, bgr_img: np.ndarray, plate_text: str, action: str, conf: int, fee: int=0):
        try:
            d = self._ensure_plates_dir(action)
            ts = int(time.time())
            plate_safe = (plate_text or "").strip().upper()
            if plate_safe:
                plate_safe = re.sub(r'[^0-9A-Z\-_.]', '_', plate_safe)[:50]
                fname = d / f"{plate_safe}_{ts}_{action}_{conf}.jpg"
            else:
                fname = d / f"{ts}_{self.cfg.gate_id}_{action}_{conf}.jpg"
            cv2.imwrite(str(fname), bgr_img)
            csvf = PLATES_DIR / "plates.csv"
            exists = csvf.exists()
            with open(csvf, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if not exists:
                    w.writerow(["ts", "date", "gate", "action", "plate", "conf", "img", "fee"])
                w.writerow([ts, time.strftime("%Y-%m-%d %H:%M:%S"), self.cfg.gate_id, action, plate_text, conf, str(fname), fee])
            return True, str(fname)
        except Exception as e:
            return False, str(e)

    # Capture IN (EasyOCR first, fallback tesseract)
    def on_capture_in(self):
        frame = None
        if self.cam_in_worker:
            frame = self.cam_in_worker.best_recent_frame(min_score=60.0)
        if frame is None:
            QMessageBox.warning(self, "Chụp IN", "Không có frame phù hợp từ camera IN."); return

        crop, bbox = detect_plate_robust(frame)
        if crop is None:
            crop = frame.copy(); used_img = frame.copy()
        else:
            used_img = crop.copy()

        plate_text = ""; conf = 0; used_preview = None

        if easyocr is not None:
            try:
                txt_e, conf_e, prev_e = ocr_with_easyocr(crop)
                if txt_e:
                    plate_text, conf, used_preview = txt_e, conf_e, prev_e
                    print("[OCR EasyOCR IN]", plate_text, conf)
            except Exception as e:
                print("EasyOCR IN error:", e)

        if not plate_text:
            try:
                if 'ocr_best_of_variants' in globals():
                    txt_t, conf_t, prev_t = ocr_best_of_variants(crop)
                    if txt_t:
                        plate_text, conf, used_preview = txt_t, conf_t, prev_t
                        print("[OCR Tesseract IN]", plate_text, conf)
                elif pytesseract is not None:
                    pre = preprocess_for_ocr(crop)
                    cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
                    t = pytesseract.image_to_string(pre, config=cfg).strip().upper()
                    t = re.sub(r'[^0-9A-Z\-\.\s]', '', t)
                    if t:
                        plate_text, conf, used_preview = t, 50, cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
                        print("[OCR pytesseract IN]", plate_text)
            except Exception as e:
                print("Fallback OCR IN error:", e)

        # Hậu xử lý chuẩn hoá biển số VN
        # (không hậu xử lý thêm)

        display_text = plate_text if plate_text else "(không đọc được)"

        try:
            if used_preview is not None:
                qimg = np_to_qimage(used_preview)
                set_pixmap_fit_no_upscale(self.lbl_img_in, qimg)
            else:
                qimg = np_to_qimage(cv2.cvtColor(preprocess_for_ocr(crop), cv2.COLOR_GRAY2BGR))
                set_pixmap_fit_no_upscale(self.lbl_img_in, qimg)
        except Exception:
            pass

        if plate_text:
            for r in self._records:
                if r.get('in_ts') and not r.get('out_ts') and r.get('in_gate') == self.cfg.gate_id:
                    if plates_match(r.get('plate',''), plate_text):
                        QMessageBox.warning(self, "Chụp IN", f"Biển {plate_text} đã ở trạng thái 'Đã vào' — không thể chụp IN trùng."); return

        ok, info = self._save_image_and_record(frame, plate_text or "", "IN", conf, fee=0)
        if not ok:
            QMessageBox.critical(self, "Chụp IN", f"Lưu ảnh thất bại: {info}"); return

        rec = {'plate': plate_text, 'in_ts': int(time.time()), 'in_img': info, 'in_gate': self.cfg.gate_id, 'out_ts': None, 'out_img': None, 'fee': 0}
        self._records.append(rec)

        try:
            self.ed_plate.setText(display_text); self.ed_tin.setText(time.strftime("%Y-%m-%d %H:%M:%S")); self.ed_fee.setText("0")
        except Exception:
            pass
        self._refresh_slot_labels()
        msg = f"IN saved. OCR: {display_text} (conf={conf})\nẢnh: {info}"
        print("[IN]", msg); QMessageBox.information(self, "Chụp IN", msg)

    # Capture OUT
    def on_capture_out(self):
        frame = None
        if self.cam_out_worker:
            frame = self.cam_out_worker.best_recent_frame(min_score=60.0)
        if frame is None and self.cam_in_worker:
            frame = self.cam_in_worker.best_recent_frame(min_score=60.0)
        if frame is None:
            QMessageBox.warning(self, "Chụp OUT", "Không có frame phù hợp từ camera OUT."); return

        crop, bbox = detect_plate_robust(frame)
        if crop is None:
            crop = frame.copy(); used_img = frame.copy()
        else:
            used_img = crop.copy()

        plate_text = ""; conf = 0; used_preview = None

        if easyocr is not None:
            try:
                txt_e, conf_e, prev_e = ocr_with_easyocr(crop)
                if txt_e:
                    plate_text, conf, used_preview = txt_e, conf_e, prev_e
                    print("[OCR EasyOCR OUT]", plate_text, conf)
            except Exception as e:
                print("EasyOCR OUT error:", e)

        if not plate_text:
            try:
                if 'ocr_best_of_variants' in globals():
                    txt_t, conf_t, prev_t = ocr_best_of_variants(crop)
                    if txt_t:
                        plate_text, conf, used_preview = txt_t, conf_t, prev_t
                        print("[OCR Tesseract OUT]", plate_text, conf)
                elif pytesseract is not None:
                    pre = preprocess_for_ocr(crop)
                    cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
                    t = pytesseract.image_to_string(pre, config=cfg).strip().upper()
                    t = re.sub(r'[^0-9A-Z\-\.\s]', '', t)
                    if t:
                        plate_text, conf, used_preview = t, 50, cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
                        print("[OCR pytesseract OUT]", plate_text)
            except Exception as e:
                print("Fallback OCR OUT error:", e)

        # Hậu xử lý chuẩn hoá biển số VN
        # (không hậu xử lý thêm)

        try:
            if used_preview is not None:
                qimg = np_to_qimage(used_preview)
                set_pixmap_fit_no_upscale(self.lbl_img_out, qimg)
            else:
                qimg = np_to_qimage(cv2.cvtColor(preprocess_for_ocr(crop), cv2.COLOR_GRAY2BGR))
                set_pixmap_fit_no_upscale(self.lbl_img_out, qimg)
        except Exception:
            pass

        if not plate_text:
            ok_dbg, info_dbg = self._save_image_and_record(frame, "", "OUT", conf, fee=0)
            print("[OUT] No plate read. Saved debug image:", info_dbg)
            QMessageBox.warning(self, "Chụp OUT", "Không nhận được biển số khi quét OUT — không thể đối soát. Ảnh đã lưu để debug."); return

        match = None
        for r in self._records:
            if r.get('in_ts') and not r.get('out_ts') and r.get('in_gate') == self.cfg.gate_id:
                if plates_match(r.get('plate',''), plate_text):
                    match = r; break

        if match is None:
            ok_dbg, info_dbg = self._save_image_and_record(frame, plate_text, "OUT", conf, fee=0)
            print("[OUT] No matching IN for plate:", plate_text, "Saved:", info_dbg)
            QMessageBox.warning(self, "Chụp OUT", f"Không tìm thấy bản ghi 'IN' tương ứng cho biển {plate_text} tại cổng này. Ảnh OUT đã lưu để debug."); return

        fee = 3000
        ok, info = self._save_image_and_record(frame, plate_text, "OUT", conf, fee=fee)
        if not ok:
            QMessageBox.critical(self, "Chụp OUT", f"Lưu ảnh OUT thất bại: {info}"); return
        match['out_ts'] = int(time.time()); match['out_img'] = info; match['fee'] = fee

        
        try:
            self.ed_plate.setText(plate_text); self.ed_tout.setText(time.strftime("%Y-%m-%d %H:%M:%S")); self.ed_fee.setText(f"{fee:,d} VND")
        except Exception:
            pass
        self._refresh_slot_labels()
        msg = f"OUT saved for {plate_text} (conf={conf}). Fee: {fee:,d} VND. Ảnh: {info}"
        print("[OUT]", msg); QMessageBox.information(self, "Chụp OUT", msg)

    # Manual entry
    def on_manual_entry(self):
        dlg = QDialog(self); dlg.setWindowTitle("Nhập biển thủ công"); dlg.resize(360,140)
        le = QLineEdit(); le.setPlaceholderText("Nhập biển (VD: 30A12345)")
        cb = QComboBox(); cb.addItems(["IN","OUT"])
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout = QVBoxLayout(dlg); form = QFormLayout(); form.addRow("Biển:", le); form.addRow("Hành động:", cb); layout.addLayout(form); layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        plate = le.text().strip().upper()
        action = cb.currentText()
        if not plate:
            QMessageBox.warning(self, "Nhập biển", "Vui lòng nhập biển."); return
        frame = None
        if action == "IN":
            if self.cam_in_worker:
                frame = self.cam_in_worker.best_recent_frame(min_score=50.0)
        else:
            if self.cam_out_worker:
                frame = self.cam_out_worker.best_recent_frame(min_score=50.0)
            if frame is None and self.cam_in_worker:
                frame = self.cam_in_worker.best_recent_frame(min_score=50.0)
        if frame is None:
            frame = np.zeros((240,320,3), dtype=np.uint8)
        if action == "IN":
            for r in self._records:
                if r.get('plate') == plate and r.get('in_ts') and not r.get('out_ts') and r.get('in_gate') == self.cfg.gate_id:
                    QMessageBox.warning(self, "Nhập IN", f"Biển {plate} đã ở trạng thái 'Đã vào'."); return
            ok, info = self._save_image_and_record(frame, plate, "IN", 0, fee=0)
            if not ok:
                QMessageBox.critical(self, "Nhập IN", f"Lưu ảnh thất bại: {info}"); return
            rec = {'plate': plate, 'in_ts': int(time.time()), 'in_img': info, 'in_gate': self.cfg.gate_id, 'out_ts': None, 'out_img': None, 'fee': 0}
            self._records.append(rec)
            self.ed_plate.setText(plate); self.ed_tin.setText(time.strftime("%Y-%m-%d %H:%M:%S")); self.ed_fee.setText("0"); self._refresh_slot_labels()
            QMessageBox.information(self, "Nhập IN", f"Đã lưu IN thủ công: {plate}")
        else:
            match = None
            for r in self._records:
                if r.get('in_ts') and not r.get('out_ts') and r.get('in_gate') == self.cfg.gate_id:
                    if plates_match(r.get('plate',''), plate):
                        match = r; break
            if match is None:
                QMessageBox.warning(self, "Nhập OUT", f"Không tìm thấy bản ghi IN tương ứng cho {plate}"); return
            fee = 3000
            ok, info = self._save_image_and_record(frame, plate, "OUT", 0, fee=fee)
            if not ok:
                QMessageBox.critical(self, "Nhập OUT", f"Lưu ảnh OUT thất bại: {info}"); return
            match['out_ts'] = int(time.time()); match['out_img'] = info; match['fee'] = fee
            self.ed_plate.setText(plate); self.ed_tout.setText(time.strftime("%Y-%m-%d %H:%M:%S")); self.ed_fee.setText(f"{fee:,d} VND"); self._refresh_slot_labels()
            QMessageBox.information(self, "Nhập OUT", f"Đã lưu OUT thủ công cho {plate}. Phí: {fee:,d} VND")

    # Search handler
    def on_search_plate(self):
        plate = self.ed_search_plate.text().strip().upper()
        if not plate:
            QMessageBox.information(self, "Tìm biển", "Vui lòng nhập biển cần tìm."); return
        dlg = PlateHistoryDialog(self, plate); dlg.exec()

    # Broker auto-start
    def ensure_broker_running(self):
        self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}")
        self.lbl_mqtt_gate.setText(self.cfg.gate_id)
        if not (self.cfg.mqtt_enable and self.cfg.auto_start_broker): return
        host = (self.cfg.mqtt_host or "").strip(); local_ips = self._local_ips or get_local_ips()
        if host not in local_ips: return
        probe_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
        if is_port_open(probe_host, self.cfg.mqtt_port): return
        exe, conf = self.cfg.broker_exe, self.cfg.broker_conf
        if not os.path.exists(exe) or not os.path.exists(conf):
            self._set_mqtt_state("Không thấy mosquitto/conf", "#ff6b6b"); return
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            exe_dir = os.path.dirname(exe) or None
            self._mosq_proc = subprocess.Popen([exe, "-c", conf], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, creationflags=flags, cwd=exe_dir)
            self._set_mqtt_state("Đang khởi động broker…", "#f1c40f")
        except Exception as e:
            self._set_mqtt_state(f"Lỗi chạy broker: {e}", "#ff6b6b"); self._mosq_proc = None

    def stop_broker_if_started(self):
        if self._mosq_proc:
            try: self._mosq_proc.terminate()
            except Exception: pass
            self._mosq_proc = None

    # MQTT
    def _set_mqtt_state(self, text, color="#bbb"):
        self.lbl_mqtt_state.setText(text); self.lbl_mqtt_state.setStyleSheet(f"color:{color};font-weight:700;")
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
        if (time.time() - self._esp_last_hb) > self._hb_timeout and self._esp_online:
            self._esp_online = False; self._refresh_conn_badge()

    def init_mqtt(self):
        if not self.cfg.mqtt_enable or mqtt is None:
            self._mqtt_connected = False; self._esp_online = False; self._set_mqtt_state("OFF", "#bbb"); return
        try:
            cid = "ui-" + "".join(random.choice(string.ascii_lowercase+string.digits) for _ in range(6))
            self.lbl_mqtt_cid.setText(cid)
            self.mqtt_client = mqtt.Client(client_id=cid)
            try:
                self.mqtt_client.reconnect_delay_set(min_delay=0.5, max_delay=3)
            except Exception:
                pass
            def _on_connect(client, userdata, flags, rc, properties=None):
                try:
                    self._mqtt_connected = (rc == 0)
                    if rc == 0:
                        client.subscribe(f"parking/gate/{self.cfg.gate_id}/event", qos=1)
                        client.subscribe(f"parking/gate/{self.cfg.gate_id}/stats", qos=1)
                        client.subscribe(f"parking/gate/{self.cfg.gate_id}/status", qos=1)
                        client.subscribe(f"parking/gate/{self.cfg.gate_id}/heartbeat", qos=0)
                    else:
                        self._esp_online = False
                except Exception:
                    pass
                self._refresh_conn_badge()
            def _on_disconnect(client, userdata, rc, properties=None):
                self._mqtt_connected = False; self._esp_online = False; self._refresh_conn_badge()
            def _on_message(client, userdata, msg):
                try:
                    top = msg.topic
                    try:
                        payload = json.loads(msg.payload.decode("utf-8"))
                    except Exception:
                        payload = {}
                    if top.endswith("/status"):
                        online = bool(payload.get("online", False)); self._esp_online = online
                        if online: self._esp_last_hb = time.time(); self._refresh_conn_badge()
                    elif top.endswith("/heartbeat"):
                        self._esp_last_hb = time.time()
                        if not self._esp_online:
                            self._esp_online = True; self._refresh_conn_badge()
                except Exception:
                    pass
            self.mqtt_client.on_connect = _on_connect; self.mqtt_client.on_disconnect = _on_disconnect; self.mqtt_client.on_message = _on_message
            self._set_mqtt_state("Đang kết nối…", "#f1c40f")
            self.mqtt_client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=20); self.mqtt_client.loop_start()
        except Exception as e:
            self._mqtt_connected = False; self._esp_online = False; self._set_mqtt_state(f"Lỗi MQTT: {e}", "#ff6b6b")

    def restart_mqtt(self):
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop(); self.mqtt_client.disconnect()
        except Exception:
            pass
        self.mqtt_client = None; self._mqtt_connected = False; self._esp_online = False
        self._refresh_conn_badge(); self.ensure_broker_running(); self.init_mqtt()

    # Settings
    def toggle_fullscreen(self, checked: bool):
        self.showFullScreen() if checked else self.showNormal()
    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted:
            cam_in, cam_out, slots, en_mqtt, host, port, gate, autob, bexe, bconf = dlg.values()
            if cam_in == -1:
                QMessageBox.warning(self, "Cài đặt", "Ngõ vào phải chọn 1 camera hợp lệ."); return
            self.cfg.cam_in_index = int(cam_in); self.cfg.cam_out_index = int(cam_out); self.cfg.total_slots = max(1, slots)
            self.cfg.mqtt_enable = bool(en_mqtt); self.cfg.mqtt_host = host; self.cfg.mqtt_port = int(port); self.cfg.gate_id = gate
            self.cfg.auto_start_broker = bool(autob); self.cfg.broker_exe = bexe; self.cfg.broker_conf = bconf
            save_config(self.cfg); self.start_cameras(); self._refresh_slot_labels(); self.restart_mqtt()
            self.lbl_mqtt_broker.setText(f"{self.cfg.mqtt_host}:{self.cfg.mqtt_port}"); self.lbl_mqtt_gate.setText(self.cfg.gate_id)

    def closeEvent(self, e):
        self.stop_cameras()
        try:
            if self.mqtt_client:
                self.mqtt_client.loop_stop(); self.mqtt_client.disconnect()
        except Exception:
            pass
        self.stop_broker_if_started()
        super().closeEvent(e)

# BOOT
def main():
    cfg = load_config()
    if not os.path.exists(cfg.broker_conf):
        Path(cfg.broker_conf).parent.mkdir(parents=True, exist_ok=True)
        open(cfg.broker_conf, "w", encoding="utf-8").write("listener 1883 0.0.0.0\nallow_anonymous true\npersistence false\n")
    app = QApplication(sys.argv)
    w = MainWindow(cfg); w.show();
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
