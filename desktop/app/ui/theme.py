# Centralized dark theme for the desktop app.
# Restrained professional look — не Dribbble-концепт, а рабочий инструмент.

from __future__ import annotations

# ── цвета ─────────────────────────────────────────────────────────────
BG_WINDOW = "#1a1b1e"
BG_CARD = "#232428"
BG_INPUT = "#2c2d31"
BG_HEADER = "#1e1f23"
BORDER = "#38393e"
BORDER_FOCUS = "#5b8def"
TEXT = "#d1d1d6"
TEXT_DIM = "#8e8e93"
TEXT_HEADING = "#f0f0f2"
ACCENT = "#5b8def"
ACCENT_HOVER = "#4a7de0"
ACCENT_PRESSED = "#3b6dd1"
SUCCESS = "#34c759"
DANGER = "#ff453a"
WARNING = "#ffd60a"
BADGE_MATCH_BG = "#1a3d2a"
BADGE_UNKNOWN_BG = "#3d1a1a"
TABLE_ROW_ALT = "#292a2e"


def app_stylesheet() -> str:
    return f"""
    /* ── база ─────────────────────────────────────────── */
    QMainWindow, QWidget {{
        background-color: {BG_WINDOW};
        color: {TEXT};
        font-family: "Segoe UI", Inter, system-ui, sans-serif;
        font-size: 13px;
    }}

    /* ── header bar ───────────────────────────────────── */
    #appHeader {{
        background-color: {BG_HEADER};
        border-bottom: 1px solid {BORDER};
        padding: 10px 16px;
    }}
    #appTitle {{
        font-size: 15px;
        font-weight: 600;
        color: {TEXT_HEADING};
    }}
    #appSubtitle {{
        font-size: 11px;
        color: {TEXT_DIM};
    }}

    /* ── tab bar ──────────────────────────────────────── */
    QTabWidget::pane {{
        border: none;
        background: {BG_WINDOW};
    }}
    QTabBar {{
        background: {BG_HEADER};
        border-bottom: 1px solid {BORDER};
    }}
    QTabBar::tab {{
        background: transparent;
        color: {TEXT_DIM};
        padding: 10px 20px;
        margin: 0;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 500;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        color: {ACCENT};
        border-bottom: 2px solid {ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT};
        background: rgba(91, 141, 239, 0.06);
    }}

    /* ── карточка (section card) ──────────────────────── */
    .card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px;
    }}

    /* ── inputs ───────────────────────────────────────── */
    QLineEdit, QSpinBox {{
        background: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 7px 10px;
        color: {TEXT};
        font-size: 13px;
        selection-background-color: {ACCENT};
    }}
    QLineEdit:focus, QSpinBox:focus {{
        border: 1px solid {BORDER_FOCUS};
    }}
    QLineEdit:disabled {{
        color: {TEXT_DIM};
    }}

    /* ── buttons ──────────────────────────────────────── */
    QPushButton {{
        background: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 7px 16px;
        color: {TEXT};
        font-weight: 500;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: #35363b;
        border-color: #4a4b50;
    }}
    QPushButton:pressed {{
        background: #2a2b30;
    }}
    QPushButton:disabled {{
        color: {TEXT_DIM};
        background: {BG_INPUT};
    }}

    /* primary action */
    QPushButton#primary {{
        background: {ACCENT};
        border: 1px solid {ACCENT};
        color: #fff;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background: {ACCENT_HOVER};
    }}
    QPushButton#primary:pressed {{
        background: {ACCENT_PRESSED};
    }}
    QPushButton#primary:disabled {{
        background: #3a4a6e;
        border-color: #3a4a6e;
        color: #8a9ec0;
    }}

    /* danger */
    QPushButton#danger {{
        border-color: {DANGER};
        color: {DANGER};
    }}
    QPushButton#danger:hover {{
        background: rgba(255, 69, 58, 0.12);
    }}

    /* ── таблицы ──────────────────────────────────────── */
    QTableWidget {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 6px;
        gridline-color: {BORDER};
        font-size: 12px;
    }}
    QTableWidget::item {{
        padding: 6px 8px;
        border: none;
    }}
    QTableWidget::item:selected {{
        background: rgba(91, 141, 239, 0.18);
        color: {TEXT};
    }}
    QHeaderView::section {{
        background: {BG_HEADER};
        color: {TEXT_DIM};
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        padding: 7px 8px;
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
    }}
    QHeaderView::section:last {{
        border-right: none;
    }}

    /* ── scrollbar ────────────────────────────────────── */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: #44454a;
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #5a5b60;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    /* ── statusbar ────────────────────────────────────── */
    QStatusBar {{
        background: {BG_HEADER};
        color: {TEXT_DIM};
        border-top: 1px solid {BORDER};
        font-size: 11px;
        padding: 2px 8px;
    }}

    /* ── section heading ──────────────────────────────── */
    .sectionHeading {{
        font-size: 13px;
        font-weight: 600;
        color: {TEXT_HEADING};
        padding: 0;
        margin: 0;
    }}

    /* ── dim label ────────────────────────────────────── */
    .dimLabel {{
        color: {TEXT_DIM};
        font-size: 12px;
    }}

    /* ── badges ───────────────────────────────────────── */
    #badgeMatch {{
        background: {BADGE_MATCH_BG};
        color: {SUCCESS};
        border: 1px solid {SUCCESS};
        border-radius: 4px;
        padding: 4px 12px;
        font-weight: 700;
        font-size: 14px;
    }}
    #badgeUnknown {{
        background: {BADGE_UNKNOWN_BG};
        color: {DANGER};
        border: 1px solid {DANGER};
        border-radius: 4px;
        padding: 4px 12px;
        font-weight: 700;
        font-size: 14px;
    }}

    /* ── info value (stats) ───────────────────────────── */
    .infoValue {{
        font-size: 22px;
        font-weight: 700;
        color: {TEXT_HEADING};
    }}
    .infoKey {{
        font-size: 11px;
        color: {TEXT_DIM};
        font-weight: 500;
    }}

    /* ── text edit (readonly response) ────────────────── */
    QTextEdit {{
        background: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px;
        color: {TEXT};
        font-family: "Cascadia Code", "Consolas", monospace;
        font-size: 12px;
    }}

    /* ── image preview ────────────────────────────────── */
    #imagePreview {{
        background: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        min-height: 120px;
    }}

    /* ── tooltip ──────────────────────────────────────── */
    QToolTip {{
        background: {BG_CARD};
        color: {TEXT};
        border: 1px solid {BORDER};
        padding: 4px 8px;
        font-size: 12px;
    }}
    """
