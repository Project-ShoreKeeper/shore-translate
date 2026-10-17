import os
import sys
import queue
import signal
import threading

while len(sys.argv) < 2:
    sys.argv.append("shore-translate")

import glob

# Configure Linux display and audio environment
if "XDG_RUNTIME_DIR" not in os.environ:
    uid_runtime = f"/run/user/{os.getuid()}"
    if os.path.exists(uid_runtime):
        os.environ["XDG_RUNTIME_DIR"] = uid_runtime
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"
if "WAYLAND_DISPLAY" not in os.environ:
    os.environ["WAYLAND_DISPLAY"] = "wayland-0"

# GNOME Wayland compositor intentionally disables WindowStaysOnTopHint for native Wayland clients.
# Auto-discovering XAUTHORITY and using xcb (Xwayland) enables full EWMH _NET_WM_STATE_ABOVE
# so the overlay stays permanently floating on top of Chrome, media players, and all desktop apps.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
    if "XAUTHORITY" not in os.environ:
        uid = os.getuid()
        auth_candidates = [
            f"/run/user/{uid}/.Xauthority",
            os.path.expanduser("~/.Xauthority"),
        ]
        auth_candidates.extend(glob.glob(f"/run/user/{uid}/.mutter-Xwaylandauth.*"))
        for candidate in auth_candidates:
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                os.environ["XAUTHORITY"] = candidate
                break

    if os.environ.get("XAUTHORITY") and os.path.exists(os.environ["XAUTHORITY"]):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

# Suppress harmless Qt Wayland textinput debug warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.textinput=false"

from PySide6.QtWidgets import QApplication

from audio import capture_loopback
from vad import run_vad_worker
from asr import run_asr_worker
from translate import run_translation_worker
from overlay import SubtitleOverlay


def main():
    print("=" * 60)
    print(" Shore-Translate: Real-Time Japanese Livestream Subtitles")
    print("=" * 60)

    # Thread communication queues
    audio_queue: queue.Queue = queue.Queue()
    speech_queue: queue.Queue = queue.Queue()
    transcript_queue: queue.Queue = queue.Queue()
    subtitle_queue: queue.Queue = queue.Queue()

    stop_event = threading.Event()

    # Worker threads
    threads: list[threading.Thread] = []

    # 1. Audio Capture Thread
    t_audio = threading.Thread(
        target=capture_loopback,
        args=(audio_queue, stop_event),
        name="AudioCaptureThread",
        daemon=True,
    )
    threads.append(t_audio)

    # 2. VAD Segmentation Thread (CPU, ONNX Runtime)
    t_vad = threading.Thread(
        target=run_vad_worker,
        args=(audio_queue, speech_queue, stop_event),
        name="VADThread",
        daemon=True,
    )
    threads.append(t_vad)

    # 3. ASR Transcription Thread (CUDA, faster-whisper / CTranslate2)
    t_asr = threading.Thread(
        target=run_asr_worker,
        args=(speech_queue, transcript_queue, stop_event),
        name="ASRThread",
        daemon=True,
    )
    threads.append(t_asr)

    # 4. Translation Thread (Local LLM / llama-server)
    t_trans = threading.Thread(
        target=run_translation_worker,
        args=(transcript_queue, subtitle_queue, stop_event),
        name="TranslationThread",
        daemon=True,
    )
    threads.append(t_trans)

    # Start backend worker threads
    for t in threads:
        t.start()
        print(f"[pipeline] Started worker thread: {t.name}")

    # Initialize PySide6 GUI Application
    app = QApplication(sys.argv)
    overlay = SubtitleOverlay(subtitle_queue=subtitle_queue, stop_event=stop_event)
    overlay.show()
    overlay.set_subtitle("Live Translation Active (Esc or ✕ to close)")

    # Handle Ctrl+C from terminal
    signal.signal(signal.SIGINT, lambda sig, frame: app.quit())

    try:
        app_exit_code = app.exec()
    finally:
        print("[pipeline] Shutting down pipeline threads...")
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        print("[pipeline] All threads stopped. Goodbye!")

    sys.exit(app_exit_code)


if __name__ == "__main__":
    main()
