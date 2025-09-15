import sys, subprocess, time, os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import Fore, Style, init

init(autoreset=True)  # bật màu cho log

APP_FILE = "parking_ui.py"
DEBOUNCE_TIME = 1.5   # giây

class Handler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith(APP_FILE):
            now = time.time()
            # tránh restart liên tục khi file bị lưu nhiều lần
            if now - self.last_modified < DEBOUNCE_TIME:
                return
            self.last_modified = now
            print(Fore.YELLOW + f"[Watcher] {APP_FILE} thay đổi, restart app...")
            restart_app()

p = None

def start_app():
    global p
    if p is not None:
        return
    print(Fore.GREEN + "[Runner] Bắt đầu chạy app...")
    p = subprocess.Popen([sys.executable, APP_FILE])

def restart_app():
    global p
    if p is not None:
        print(Fore.CYAN + "[Runner] Dừng app cũ...")
        p.terminate()
        try:
            p.wait(timeout=3)  # đợi camera được giải phóng
        except subprocess.TimeoutExpired:
            print(Fore.RED + "[Runner] App không dừng kịp, kill...")
            p.kill()
            p.wait()
        time.sleep(1)  # nghỉ cho chắc
    print(Fore.GREEN + "[Runner] Chạy lại app...")
    p = subprocess.Popen([sys.executable, APP_FILE])

def stop_app():
    global p
    if p is not None:
        print(Fore.CYAN + "[Runner] Tắt app...")
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

if __name__ == "__main__":
    if not os.path.exists(APP_FILE):
        print(Fore.RED + f"[Error] Không tìm thấy file {APP_FILE}")
        sys.exit(1)

    start_app()
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        stop_app()

    observer.join()
