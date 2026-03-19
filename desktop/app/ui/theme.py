# Централизованная тема для desktop-клиента
from __future__ import annotations

# Палитра
BG_WINDOW = "#1e1f22"
BG_CARD = "#27282c"
BG_INPUT = "#2c2d31"
BG_HOVER = "#35363b"
BG_TABLE_ALT = "#24252a"
BORDER = "#3a3b40"
TEXT = "#c8cad0"
TEXT_DIM = "#8b8d93"
TEXT_HEADING = "#e1e2e6"
ACCENT = "#5b8def"
ACCENT_HOVER = "#7ba4f7"
GREEN = "#3ba55d"
RED = "#ed4245"
ORANGE = "#faa61a"


def app_stylesheet() -> str:
    return f"""
    /* ── глобальные ── */
    QWidget {{
        background-color: {BG_WINDOW};
        color: {TEXT};
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
    }}

    /* ── карточки ── */
    QFrame.card {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    /* ── инпуты ── */
    QLineEdit, QSpinBox, QTextEdit, QComboBox {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 6px 8px;
        color: {TEXT};
        selection-background-color: {ACCENT};
    }}
    QLineEdit:focus, QSpinBox:focus, QTextEdit:focus, QComboBox:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT};
        color: {TEXT};
    }}

    /* ── кнопки ── */
    QPushButton {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 6px 16px;
        color: {TEXT};
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {BG_HOVER};
        border-color: {TEXT_DIM};
    }}
    QPushButton:pressed {{
        background-color: {BORDER};
    }}
    QPushButton:disabled {{
        color: {TEXT_DIM};
        background-color: {BG_WINDOW};
    }}
    QPushButton#primary {{
        background-color: {ACCENT};
        border-color: {ACCENT};
        color: #ffffff;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background-color: {ACCENT_HOVER};
    }}
    QPushButton#danger {{
        background-color: transparent;
        border-color: {RED};
        color: {RED};
    }}
    QPushButton#danger:hover {{
        background-color: {RED};
        color: #ffffff;
    }}

    /* ── таблицы ── */
    QTableWidget {{
        background-color: {BG_CARD};
        alternate-background-color: {BG_TABLE_ALT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        gridline-color: {BORDER};
        selection-background-color: {ACCENT};
        selection-color: #ffffff;
    }}
    QHeaderView::section {{
        background-color: {BG_INPUT};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 5px 8px;
        font-weight: 600;
        color: {TEXT_DIM};
    }}

    /* ── вкладки ── */
    QTabWidget::pane {{
        border: none;
        background-color: {BG_WINDOW};
    }}
    QTabBar::tab {{
        background-color: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 20px;
        color: {TEXT_DIM};
        font-weight: 500;
    }}
    QTabBar::tab:selected {{
        color: {ACCENT};
        border-bottom-color: {ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT};
        border-bottom-color: {TEXT_DIM};
    }}

    /* ── бейджи ── */
    QLabel#badgeMatch {{
        background-color: {GREEN};
        color: #ffffff;
        border-radius: 4px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 15px;
    }}
    QLabel#badgeUnknown {{
        background-color: {RED};
        color: #ffffff;
        border-radius: 4px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 15px;
    }}
    QLabel#badgeCompare {{
        background-color: {ORANGE};
        color: #111111;
        border-radius: 4px;
        padding: 4px 14px;
        font-weight: 700;
        font-size: 15px;
    }}

    /* ── скроллбар ── */
    QScrollBar:vertical {{
        background: {BG_WINDOW};
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* ── статусбар ── */
    QStatusBar {{
        background-color: {BG_CARD};
        border-top: 1px solid {BORDER};
        color: {TEXT_DIM};
        font-size: 12px;
        padding: 2px 8px;
    }}

    /* ── тултипы ── */
    QToolTip {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        color: {TEXT};
        padding: 4px 8px;
    }}
    """
