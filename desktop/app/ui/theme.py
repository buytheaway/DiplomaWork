from __future__ import annotations

BG_WINDOW = "#0c1117"
BG_PANEL = "#11171f"
BG_PANEL_2 = "#151d26"
BG_ELEVATED = "#18212c"
BG_INPUT = "#0f151d"
BG_HOVER = "#1b2632"
BG_ACTIVE = "#213142"
BORDER = "#25313d"
BORDER_SOFT = "#1c2732"
BORDER_STRONG = "#314252"
TEXT = "#d7e1ea"
TEXT_SOFT = "#94a5b5"
TEXT_MUTED = "#6e8092"
ACCENT = "#4ccde8"
ACCENT_SOFT = "#b4eef8"
SUCCESS = "#53d8b2"
WARN = "#e6c06b"
ERROR = "#ef7d95"


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {BG_WINDOW};
        color: {TEXT};
        font-family: "Segoe UI", "Bahnschrift";
        font-size: 13px;
    }}

    QMainWindow {{
        background-color: {BG_WINDOW};
    }}

    QFrame#sidebar {{
        background-color: {BG_PANEL};
        border-right: 1px solid {BORDER_SOFT};
    }}

    QFrame#topbar {{
        background-color: {BG_WINDOW};
        border-bottom: 1px solid {BORDER_SOFT};
    }}

    QFrame#card[variant="default"],
    QFrame#card[variant="metric"],
    QFrame#card[variant="subtle"] {{
        background-color: {BG_PANEL_2};
        border: 1px solid {BORDER_SOFT};
        border-radius: 8px;
    }}

    QFrame#card[variant="metric"] {{
        background-color: {BG_PANEL};
    }}

    QFrame#card[variant="subtle"] {{
        background-color: rgba(255, 255, 255, 0.02);
    }}

    QLabel#sectionHeading {{
        color: {TEXT};
        font-size: 16px;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#dimLabel {{
        color: {TEXT_SOFT};
        font-size: 12px;
        line-height: 150%;
        background: transparent;
    }}

    QLabel#microLabel {{
        color: {TEXT_MUTED};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#infoValue {{
        color: {TEXT};
        font-size: 16px;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#decisionText {{
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        background-color: rgba(148, 165, 181, 0.08);
        border: 1px solid {BORDER_SOFT};
    }}

    QLabel#decisionText[state="ok"] {{
        color: {SUCCESS};
        background-color: rgba(83, 216, 178, 0.08);
        border-color: rgba(83, 216, 178, 0.22);
    }}

    QLabel#decisionText[state="warn"] {{
        color: {WARN};
        background-color: rgba(230, 192, 107, 0.08);
        border-color: rgba(230, 192, 107, 0.22);
    }}

    QLabel#decisionText[state="error"] {{
        color: {ERROR};
        background-color: rgba(239, 125, 149, 0.08);
        border-color: rgba(239, 125, 149, 0.22);
    }}

    QLabel#decisionText[state="idle"] {{
        color: {TEXT_SOFT};
    }}

    QLabel#metricTitle {{
        color: {TEXT_MUTED};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#metricValue {{
        color: {TEXT};
        font-size: 24px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#metricDetail {{
        color: {TEXT_SOFT};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#brandTitle {{
        color: {TEXT};
        font-size: 22px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#brandSubtitle,
    QLabel#operatorName,
    QLabel#operatorMeta,
    QLabel#topbarTitle {{
        background: transparent;
    }}

    QLabel#brandSubtitle {{
        color: {TEXT_MUTED};
        font-size: 11px;
    }}

    QLabel#operatorName {{
        color: {TEXT_SOFT};
        font-size: 12px;
        font-weight: 600;
    }}

    QLabel#operatorMeta {{
        color: {TEXT_MUTED};
        font-size: 11px;
    }}

    QLabel#topbarTitle {{
        color: {TEXT};
        font-size: 18px;
        font-weight: 600;
    }}

    QLineEdit, QTextEdit, QComboBox, QSpinBox, QTableWidget {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: #061018;
    }}

    QLineEdit, QComboBox, QSpinBox {{
        padding: 7px 10px;
        min-height: 18px;
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
        selection-background-color: {BG_ACTIVE};
        selection-color: {TEXT};
    }}

    QPushButton#navButton {{
        background-color: transparent;
        border: none;
        border-left: 2px solid transparent;
        color: {TEXT_SOFT};
        font-size: 13px;
        font-weight: 600;
        padding: 10px 12px;
        text-align: left;
    }}

    QPushButton#navButton:hover {{
        background-color: rgba(76, 205, 232, 0.06);
        color: {TEXT};
    }}

    QPushButton#navButton:checked {{
        background-color: rgba(76, 205, 232, 0.08);
        border-left-color: {ACCENT};
        color: {TEXT};
    }}

    QPushButton#primaryButton,
    QPushButton#secondaryButton,
    QPushButton#toolbarButton,
    QPushButton#collapseButton {{
        min-height: 34px;
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: 600;
    }}

    QPushButton#primaryButton {{
        background-color: {ACCENT};
        color: #071218;
        border: 1px solid {ACCENT};
    }}

    QPushButton#primaryButton:hover {{
        background-color: #67d8ee;
        border-color: #67d8ee;
    }}

    QPushButton#secondaryButton,
    QPushButton#toolbarButton,
    QPushButton#collapseButton {{
        background-color: transparent;
        color: {TEXT};
        border: 1px solid {BORDER};
    }}

    QPushButton#secondaryButton:hover,
    QPushButton#toolbarButton:hover,
    QPushButton#collapseButton:hover {{
        background-color: {BG_HOVER};
        border-color: {BORDER_STRONG};
    }}

    QPushButton#toolbarButton:checked {{
        background-color: rgba(76, 205, 232, 0.08);
        border-color: {ACCENT};
        color: {TEXT};
    }}

    QPushButton#collapseButton {{
        text-align: left;
        padding-left: 10px;
        color: {TEXT_SOFT};
    }}

    QPushButton:disabled {{
        color: {TEXT_MUTED};
        border-color: {BORDER_SOFT};
        background-color: rgba(255, 255, 255, 0.02);
    }}

    QLabel[state="ok"],
    QLabel[state="warn"],
    QLabel[state="error"],
    QLabel[state="idle"] {{
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid {BORDER};
    }}

    QLabel[state="ok"] {{
        color: {SUCCESS};
        background-color: rgba(83, 216, 178, 0.08);
        border-color: rgba(83, 216, 178, 0.24);
    }}

    QLabel[state="warn"] {{
        color: {WARN};
        background-color: rgba(230, 192, 107, 0.08);
        border-color: rgba(230, 192, 107, 0.24);
    }}

    QLabel[state="error"] {{
        color: {ERROR};
        background-color: rgba(239, 125, 149, 0.08);
        border-color: rgba(239, 125, 149, 0.24);
    }}

    QLabel[state="idle"] {{
        color: {TEXT_SOFT};
        background-color: rgba(148, 165, 181, 0.08);
    }}

    QFrame#imageDropZone {{
        background-color: {BG_INPUT};
        border: 1px dashed {BORDER_STRONG};
        border-radius: 10px;
    }}

    QLabel#dropZoneTitle {{
        color: {TEXT};
        font-size: 18px;
        font-weight: 600;
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
        font-weight: 600;
        background: transparent;
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    QFrame#resultCard {{
        background-color: {BG_ELEVATED};
        border: 1px solid {BORDER_SOFT};
        border-radius: 8px;
    }}

    QLabel#resultThumb {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        color: {TEXT_MUTED};
        font-size: 11px;
    }}

    QLabel#resultTitle {{
        color: {TEXT};
        font-size: 14px;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#resultSubtitle,
    QLabel#resultMeta,
    QLabel#resultPipeline,
    QLabel#resultDistance {{
        color: {TEXT_SOFT};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#resultScore {{
        color: {ACCENT_SOFT};
        font-size: 22px;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#compactPreview {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        color: {TEXT_MUTED};
        font-size: 11px;
    }}

    QFrame#liveFaceLine,
    QFrame#personCard {{
        background-color: {BG_ELEVATED};
        border: 1px solid {BORDER_SOFT};
        border-radius: 8px;
    }}

    QLabel#liveFaceTitle,
    QLabel#personCardTitle {{
        color: {TEXT};
        font-size: 13px;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#liveFaceMeta,
    QLabel#personCardMeta {{
        color: {TEXT_SOFT};
        font-size: 11px;
        background: transparent;
    }}

    QLabel#liveFaceStatus {{
        min-width: 74px;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid {BORDER};
    }}

    QLabel#liveFaceStatus[state="ok"] {{
        color: {SUCCESS};
        background-color: rgba(83, 216, 178, 0.08);
        border-color: rgba(83, 216, 178, 0.24);
    }}

    QLabel#liveFaceStatus[state="warn"] {{
        color: {WARN};
        background-color: rgba(230, 192, 107, 0.08);
        border-color: rgba(230, 192, 107, 0.24);
    }}

    QLabel#liveFaceStatus[state="error"] {{
        color: {ERROR};
        background-color: rgba(239, 125, 149, 0.08);
        border-color: rgba(239, 125, 149, 0.24);
    }}

    QTextEdit#consoleView {{
        background-color: {BG_INPUT};
        color: #c9d7e4;
        border: 1px solid {BORDER};
        font-family: "Consolas", "Courier New";
        font-size: 12px;
    }}

    QTableWidget {{
        alternate-background-color: rgba(255, 255, 255, 0.02);
        gridline-color: {BORDER_SOFT};
    }}

    QHeaderView::section {{
        background-color: {BG_PANEL};
        border: none;
        border-bottom: 1px solid {BORDER_SOFT};
        padding: 8px 10px;
        color: {TEXT_SOFT};
        font-size: 11px;
        font-weight: 600;
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

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
        height: 0;
    }}

    QMessageBox {{
        background-color: {BG_PANEL};
    }}
    """
