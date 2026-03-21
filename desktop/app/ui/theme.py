from __future__ import annotations

BG_WINDOW = "#0a0f16"
BG_PANEL = "#111821"
BG_PANEL_2 = "#151d27"
BG_ELEVATED = "#1a222d"
BG_INPUT = "#0d141c"
BG_HOVER = "#1d2733"
BG_ACTIVE = "#202c3a"
BORDER = "#243241"
BORDER_STRONG = "#2b3f55"
TEXT = "#d7e3ee"
TEXT_SOFT = "#8ca2b6"
TEXT_MUTED = "#6b7f91"
ACCENT = "#22e7ff"
ACCENT_SOFT = "#b9f5ff"
SUCCESS = "#37f3bb"
WARN = "#ffcc73"
ERROR = "#ff6b88"


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {BG_WINDOW};
        color: {TEXT};
        font-family: "Bahnschrift SemiCondensed", "Segoe UI";
        font-size: 13px;
    }}

    QMainWindow {{
        background-color: {BG_WINDOW};
    }}

    QFrame#sidebar {{
        background-color: {BG_PANEL};
        border-right: 1px solid {BORDER};
    }}

    QFrame#topbar {{
        background-color: rgba(9, 14, 20, 0.92);
        border-bottom: 1px solid {BORDER};
    }}

    QFrame#card[variant="default"], QFrame#card[variant="metric"] {{
        background-color: {BG_PANEL_2};
        border: 1px solid {BORDER};
        border-radius: 4px;
    }}

    QFrame#card[variant="metric"] {{
        border-top: 1px solid {ACCENT};
    }}

    QLabel#sectionHeading {{
        color: {ACCENT_SOFT};
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1px;
        background: transparent;
    }}

    QLabel#dimLabel {{
        color: {TEXT_SOFT};
        font-size: 11px;
        letter-spacing: 1px;
        background: transparent;
    }}

    QLabel#microLabel {{
        color: {TEXT_MUTED};
        font-size: 10px;
        letter-spacing: 1px;
        background: transparent;
    }}

    QLabel#infoValue {{
        color: {TEXT};
        font-size: 16px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#metricTitle {{
        color: {TEXT_MUTED};
        font-size: 10px;
        letter-spacing: 1px;
        background: transparent;
    }}

    QLabel#metricValue {{
        color: {TEXT};
        font-size: 22px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#metricDetail {{
        color: {ACCENT};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#brandTitle {{
        color: {ACCENT_SOFT};
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 1px;
        background: transparent;
    }}

    QLabel#brandSubtitle, QLabel#operatorName, QLabel#operatorMeta, QLabel#topbarTitle {{
        background: transparent;
    }}

    QLabel#brandSubtitle {{
        color: {TEXT_MUTED};
        font-size: 11px;
        letter-spacing: 1px;
    }}

    QLabel#operatorName {{
        color: {TEXT};
        font-size: 14px;
        font-weight: 700;
    }}

    QLabel#operatorMeta {{
        color: {TEXT_SOFT};
        font-size: 11px;
        letter-spacing: 1px;
    }}

    QLabel#topbarTitle {{
        color: {TEXT};
        font-size: 14px;
        font-weight: 700;
    }}

    QLineEdit, QTextEdit, QComboBox, QSpinBox, QTableWidget {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 2px;
        color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: #061018;
    }}

    QLineEdit, QComboBox, QSpinBox {{
        padding: 8px 10px;
    }}

    QTextEdit {{
        padding: 10px;
    }}

    QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border-color: {ACCENT};
    }}

    QComboBox::drop-down {{
        width: 24px;
        border: none;
    }}

    QComboBox QAbstractItemView {{
        background-color: {BG_PANEL_2};
        border: 1px solid {BORDER};
        color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: #061018;
    }}

    QPushButton#navButton {{
        background-color: transparent;
        border: none;
        border-left: 3px solid transparent;
        color: {TEXT_SOFT};
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 12px 18px;
        text-align: left;
    }}

    QPushButton#navButton:hover {{
        background-color: rgba(34, 231, 255, 0.06);
        color: {TEXT};
    }}

    QPushButton#navButton:checked {{
        background-color: rgba(255, 255, 255, 0.06);
        border-left-color: {ACCENT};
        color: {ACCENT};
    }}

    QPushButton#primaryButton, QPushButton#secondaryButton, QPushButton#toolbarButton {{
        min-height: 34px;
        padding: 9px 16px;
        border-radius: 2px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    QPushButton#primaryButton {{
        background-color: {ACCENT};
        color: #041017;
        border: 1px solid {ACCENT};
    }}

    QPushButton#primaryButton:hover {{
        background-color: {ACCENT_SOFT};
        border-color: {ACCENT_SOFT};
    }}

    QPushButton#secondaryButton, QPushButton#toolbarButton {{
        background-color: transparent;
        color: {TEXT};
        border: 1px solid {BORDER_STRONG};
    }}

    QPushButton#secondaryButton:hover, QPushButton#toolbarButton:hover {{
        background-color: {BG_HOVER};
        border-color: {ACCENT};
        color: {ACCENT_SOFT};
    }}

    QPushButton#toolbarButton:checked {{
        background-color: rgba(34, 231, 255, 0.12);
        border-color: {ACCENT};
        color: {ACCENT_SOFT};
    }}

    QPushButton:disabled {{
        color: {TEXT_MUTED};
        border-color: {BORDER};
        background-color: rgba(255, 255, 255, 0.02);
    }}

    QLabel[state="ok"], QLabel[state="warn"], QLabel[state="error"], QLabel[state="idle"] {{
        padding: 6px 12px;
        border-radius: 11px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        border: 1px solid {BORDER};
    }}

    QLabel[state="ok"] {{
        color: {SUCCESS};
        background-color: rgba(55, 243, 187, 0.08);
        border-color: rgba(55, 243, 187, 0.3);
    }}

    QLabel[state="warn"] {{
        color: {WARN};
        background-color: rgba(255, 204, 115, 0.08);
        border-color: rgba(255, 204, 115, 0.3);
    }}

    QLabel[state="error"] {{
        color: {ERROR};
        background-color: rgba(255, 107, 136, 0.08);
        border-color: rgba(255, 107, 136, 0.3);
    }}

    QLabel[state="idle"] {{
        color: {TEXT_SOFT};
        background-color: rgba(140, 162, 182, 0.08);
    }}

    QFrame#imageDropZone {{
        background-color: #0c1219;
        border: 1px solid {BORDER_STRONG};
        border-radius: 4px;
    }}

    QLabel#dropZoneTitle {{
        color: {TEXT};
        font-size: 24px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#dropZoneSubtitle {{
        color: {TEXT_SOFT};
        font-size: 12px;
        line-height: 150%;
        background: transparent;
    }}

    QLabel#dropZoneImage {{
        color: {TEXT_MUTED};
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        background: transparent;
        border: 1px dashed {BORDER_STRONG};
    }}

    QFrame#resultCard {{
        background-color: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 2px;
    }}

    QLabel#resultThumb {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        color: {TEXT_MUTED};
        font-size: 11px;
        letter-spacing: 1px;
    }}

    QLabel#resultTitle {{
        color: {TEXT};
        font-size: 14px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#resultSubtitle, QLabel#resultMeta, QLabel#resultPipeline, QLabel#resultDistance {{
        color: {TEXT_SOFT};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#resultScore {{
        color: {ACCENT_SOFT};
        font-size: 24px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#compactPreview {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        color: {TEXT_MUTED};
        font-size: 11px;
        letter-spacing: 1px;
    }}

    QFrame#liveFaceLine {{
        background-color: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 2px;
    }}

    QLabel#liveFaceTitle {{
        color: {TEXT};
        font-size: 13px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#liveFaceMeta {{
        color: {TEXT_SOFT};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#liveFaceStatus {{
        min-width: 78px;
        padding: 6px 10px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        border: 1px solid {BORDER};
    }}

    QLabel#liveFaceStatus[state="ok"] {{
        color: {SUCCESS};
        background-color: rgba(55, 243, 187, 0.08);
        border-color: rgba(55, 243, 187, 0.3);
    }}

    QLabel#liveFaceStatus[state="warn"] {{
        color: {WARN};
        background-color: rgba(255, 204, 115, 0.08);
        border-color: rgba(255, 204, 115, 0.3);
    }}

    QLabel#liveFaceStatus[state="error"] {{
        color: {ERROR};
        background-color: rgba(255, 107, 136, 0.08);
        border-color: rgba(255, 107, 136, 0.3);
    }}

    QFrame#personCard {{
        background-color: {BG_ELEVATED};
        border: 1px solid {BORDER};
        border-radius: 3px;
    }}

    QLabel#personCardTitle {{
        color: {ACCENT_SOFT};
        font-size: 15px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#personCardMeta {{
        color: {TEXT_SOFT};
        font-size: 11px;
        background: transparent;
    }}

    QTextEdit#consoleView {{
        background-color: #0b1117;
        color: #c7f7ff;
        border: 1px solid {BORDER};
        font-family: "Consolas", "Courier New";
        font-size: 12px;
    }}

    QTableWidget {{
        alternate-background-color: {BG_ELEVATED};
        gridline-color: {BORDER};
    }}

    QHeaderView::section {{
        background-color: {BG_PANEL};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 8px 10px;
        color: {TEXT_SOFT};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    QScrollBar:vertical {{
        background: {BG_WINDOW};
        width: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {BORDER_STRONG};
        min-height: 24px;
        border-radius: 4px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {ACCENT};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
        height: 0;
    }}

    QMessageBox {{
        background-color: {BG_PANEL};
    }}
    """
