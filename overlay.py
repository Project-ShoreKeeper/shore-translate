import os
import sys
import time
import queue
import textwrap
import threading
from typing import Optional

import glob

# Ensure Wayland / X11 and PipeWire access
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
# so the overlay is guaranteed to stay floating on top of Chrome, media players, and all other apps.
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

# Suppress harmless Qt textinput debug warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.textinput=false"

from PySide6.QtCore import Qt, QTimer, QPoint, QRectF
from PySide6.QtGui import (
    QPainter,
    QPainterPath,
    QFont,
    QColor,
    QPen,
    QBrush,
    QFontMetrics,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)

from config import SUBTITLE_CLEAR_DELAY_SEC, OVERLAY_MAX_LINES


class OutlinedSubtitleWidget(QWidget):
    """
    Renders text with a semi-transparent dark background,
    white text (final) or amber italic text (live draft), and a crisp dark outline. Limits display to at most 2 lines.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.text: str = ""
        self.is_draft: bool = False
        self.font = QFont("Sans-Serif", 22, QFont.Weight.Bold)
        self.draft_font = QFont("Sans-Serif", 21, QFont.Weight.Medium)
        self.draft_font.setItalic(True)

    def set_subtitle_text(self, text: str, is_draft: bool = False) -> None:
        self.text = text
        self.is_draft = is_draft
        self.update()

    def clear_text(self) -> None:
        self.text = ""
        self.is_draft = False
        self.update()

    def mousePressEvent(self, event) -> None:
        win = self.window()
        if hasattr(win, "handle_drag_start"):
            win.handle_drag_start(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        win = self.window()
        if hasattr(win, "handle_drag_move"):
            win.handle_drag_move(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        win = self.window()
        if hasattr(win, "handle_drag_end"):
            win.handle_drag_end(event)
        else:
            super().mouseReleaseEvent(event)

    def _format_lines(self, width: int) -> list[str]:
        if not self.text.strip():
            return []

        active_font = self.draft_font if self.is_draft else self.font
        fm = QFontMetrics(active_font)
        # Average character width estimation for text wrapping
        avg_char_w = max(fm.averageCharWidth(), 12)
        max_chars_per_line = max(int(width / avg_char_w), 20)

        raw_lines = self.text.split("\n")
        wrapped = []
        for rl in raw_lines:
            wrapped.extend(textwrap.wrap(rl, width=max_chars_per_line))

        # Keep at most OVERLAY_MAX_LINES (last 2 lines if longer)
        if len(wrapped) > OVERLAY_MAX_LINES:
            wrapped = wrapped[-OVERLAY_MAX_LINES:]

        return wrapped

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Semi-transparent background
        if self.text.strip():
            bg_brush = QBrush(QColor(15, 15, 18, 175))
            painter.setBrush(bg_brush)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 12, 12)

            lines = self._format_lines(rect.width() - 40)
            if not lines:
                return

            active_font = self.draft_font if self.is_draft else self.font
            fm = QFontMetrics(active_font)
            line_height = fm.height() + 6
            total_text_height = len(lines) * line_height
            start_y = (rect.height() - total_text_height) // 2 + fm.ascent()

            painter.setFont(active_font)

            # Fill color: soft cream/amber for live draft, pure white for final commit
            fill_color = QColor(255, 235, 175, 235) if self.is_draft else QColor(255, 255, 255, 255)

            # Draw each line centered horizontally with black stroke + fill color
            for idx, line in enumerate(lines):
                line_w = fm.horizontalAdvance(line)
                x = (rect.width() - line_w) // 2
                y = start_y + idx * line_height

                path = QPainterPath()
                path.addText(x, y, active_font, line)

                # Black outline
                outline_pen = QPen(QColor(0, 0, 0, 255), 4, Qt.PenStyle.SolidLine)
                painter.strokePath(path, outline_pen)

                # Inner text
                painter.fillPath(path, QBrush(fill_color))
        else:
            # Subtle idle indicator so user can locate and drag the window
            idle_font = QFont("Sans-Serif", 11, QFont.Weight.Medium)
            painter.setFont(idle_font)
            fm = QFontMetrics(idle_font)
            idle_text = "Shore-Translate Active • Drag to position • Esc to close"
            text_w = fm.horizontalAdvance(idle_text) + 24
            text_h = fm.height() + 8
            pill_rect = QRectF(
                (rect.width() - text_w) / 2,
                (rect.height() - text_h) / 2,
                text_w,
                text_h,
            )
            painter.setBrush(QBrush(QColor(20, 20, 25, 90)))
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
            painter.drawRoundedRect(pill_rect, 6, 6)
            painter.setPen(QPen(QColor(200, 200, 200, 140)))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, idle_text)


class SubtitleOverlay(QWidget):
    """
    Floating, frameless, always-on-top, draggable subtitle overlay window.
    """

    def __init__(
        self,
        subtitle_queue: Optional[queue.Queue] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__()
        self.subtitle_queue = subtitle_queue
        self.stop_event = stop_event
        self._drag_pos: Optional[QPoint] = None
        self._current_text: str = ""
        self._last_update_ts: float = 0.0

        # Window flags: frameless, always-on-top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Set default size and position near bottom center of screen
        self.resize(880, 130)
        self._center_near_bottom()

        # UI Layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)

        # Top bar with a drag grip and small close button
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(6, 0, 6, 0)

        self.drag_handle = QLabel("⠿ Drag to reposition", self)
        self.drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self.drag_handle.setStyleSheet(
            """
            QLabel {
                color: rgba(220, 220, 220, 150);
                background-color: rgba(40, 40, 45, 130);
                border-radius: 9px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QLabel:hover {
                color: #ffffff;
                background-color: rgba(80, 80, 95, 200);
            }
            """
        )
        self.drag_handle.mousePressEvent = self.handle_drag_start
        self.drag_handle.mouseMoveEvent = self.handle_drag_move
        self.drag_handle.mouseReleaseEvent = self.handle_drag_end
        top_bar.addWidget(self.drag_handle)
        top_bar.addStretch()

        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(60, 60, 60, 140);
                color: #cccccc;
                border: none;
                border-radius: 11px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(220, 50, 50, 200);
                color: white;
            }
            """
        )
        self.close_btn.clicked.connect(self.close)
        top_bar.addWidget(self.close_btn)
        root_layout.addLayout(top_bar)

        # Subtitle display area
        self.subtitle_widget = OutlinedSubtitleWidget(self)
        root_layout.addWidget(self.subtitle_widget)

        # Clear timer (~6s after last update)
        self.clear_timer = QTimer(self)
        self.clear_timer.setInterval(int(SUBTITLE_CLEAR_DELAY_SEC * 1000))
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self._on_clear_timeout)

        # Queue polling timer (50ms interval)
        if self.subtitle_queue is not None:
            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(50)
            self.poll_timer.timeout.connect(self._poll_queue)
            self.poll_timer.start()

    def _center_near_bottom(self):
        screen = QApplication.primaryScreen()
        if screen:
            screen_geo = screen.geometry()
            x = (screen_geo.width() - self.width()) // 2
            y = screen_geo.height() - self.height() - 90
            self.move(x, y)

    def set_subtitle(self, text: str, is_final: bool = True) -> None:
        """
        Update subtitle text.
        Live drafts render progressively with an ellipsis in italic amber.
        Final commits snap into solid crisp white and reset the clear timer.
        """
        display_text = text if is_final else f"{text}…"
        self._current_text = display_text
        self._last_update_ts = time.time()
        self.subtitle_widget.set_subtitle_text(display_text, is_draft=(not is_final))
        if is_final:
            self.clear_timer.start(int(SUBTITLE_CLEAR_DELAY_SEC * 1000))
        else:
            self.clear_timer.start(4000)
        self.raise_()

    def _on_clear_timeout(self) -> None:
        self._current_text = ""
        self.subtitle_widget.clear_text()

    def _poll_queue(self) -> None:
        if self.subtitle_queue is None:
            return
        while True:
            try:
                item = self.subtitle_queue.get_nowait()
            except queue.Empty:
                break
            if len(item) == 4:
                en_text, ja_text, latency, is_final = item
            else:
                en_text, ja_text, latency = item
                is_final = True

            if en_text:
                self.set_subtitle(en_text, is_final=is_final)

    # Cross-platform dragging support (Wayland startSystemMove + X11 fallback)
    def handle_drag_start(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            wh = self.windowHandle()
            if wh is not None:
                try:
                    if wh.startSystemMove():
                        event.accept()
                        return
                except Exception:
                    pass
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def handle_drag_move(self, event) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def handle_drag_end(self, event) -> None:
        self._drag_pos = None
        event.accept()

    def mousePressEvent(self, event) -> None:
        self.handle_drag_start(event)

    def mouseMoveEvent(self, event) -> None:
        self.handle_drag_move(event)

    def mouseReleaseEvent(self, event) -> None:
        self.handle_drag_end(event)

    # Key press: Esc closes the overlay
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = SubtitleOverlay()
    overlay.show()
    overlay.set_subtitle("Live Translation Subtitles Ready.\nDrag to move | Press Esc to close.")
    sys.exit(app.exec())
