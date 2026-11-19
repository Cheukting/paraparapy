# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller spec file for ParaParaPy.

Build commands:
    macOS:  pyinstaller paraparapy.spec
    Windows: pyinstaller paraparapy.spec

This produces a directory-based app (not single-file) for better performance
with MediaPipe models. The app uses multiprocessing with spawn mode for the
capture worker process.

Key configuration:
- Bundle identifier: io.paraparapy.app
- Camera permission: "ParaParaPy uses your camera to track your dance movements in real time."
- Icons: paraparapy.icns (macOS), paraparapy.ico (Windows)
- Assets: branding PNGs and catalog.json are bundled; songs are user-managed
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ---------------------------------------------------------------------------
# MediaPipe requires its model files to be bundled
# ---------------------------------------------------------------------------
# Collect all MediaPipe data files (pose models, etc.)
mediapipe_datas = collect_data_files(
    "mediapipe",
    include_py_files=True,
)

# MediaPipe has many optional submodules; collect them to avoid import errors
mediapipe_hidden = collect_submodules("mediapipe")

# ---------------------------------------------------------------------------
# Assets to bundle
# ---------------------------------------------------------------------------
# Branding assets (logo, wordmark, palette) — needed for the game's About/credits
# Catalog.json — demo song metadata (songs themselves are user-managed)
assets_datas = [
    ("assets/branding/paraparapy-logo.png", "assets/branding/"),
    ("assets/branding/paraparapy-wordmark.png", "assets/branding/"),
    ("assets/branding/PALETTE.md", "assets/branding/"),
    ("assets/songs/catalog.json", "assets/songs/"),
]

# ---------------------------------------------------------------------------
# Data files: MediaPipe models + branding assets
# ---------------------------------------------------------------------------
datas = mediapipe_datas + assets_datas

# ---------------------------------------------------------------------------
# Hidden imports: modules that are imported dynamically
# ---------------------------------------------------------------------------
hiddenimports = [
    # MediaPipe and dependencies
    *mediapipe_hidden,
    "mediapipe.tasks.vision",
    "mediapipe.tasks.vision.pose_landmarker",
    "numpy",
    "numpy.core",
    "numpy.core.multiarray",
    # pygame-ce
    "pygame",
    "pygame.ce",
    # ParaParaPy's own modules (for the worker process bootstrap)
    "parapara.capture.worker",
    "parapara.game.app",
    "parapara.__main__",
]

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
# The main game entry point
# All other console scripts (parapara-record, etc.) are CLI tools that may
# be excluded from the bundled app; users run them from the source tree.
a = Analysis(
    ["src/parapara/game/app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=["pyinstaller_hooks/rthook_parapara.py"],
    excludes=[
        # Exclude test frameworks and dev tools
        "pytest",
        "pytest_testmon",
        "ruff",
        "pyrefly",
        "prek",
        "pip_audit",
        "librosa",  # offline beat-map tool only
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# Platform-specific configuration
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    # macOS .app bundle
    # For macOS, we use EXE -> BUNDLE pattern. The BUNDLE creates the .app
    # and COLLECT gathers the supporting binaries and data files.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        name="ParaParaPy",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=False,  # No terminal window
        icon="assets/branding/paraparapy.icns",
    )
    app = BUNDLE(
        exe,
        name="ParaParaPy.app",
        icon="assets/branding/paraparapy.icns",
        bundle_identifier="io.paraparapy.app",
        info_plist={
            # Camera permission — required since macOS Catalina
            "NSCameraUsageDescription": (
                "ParaParaPy uses your camera to track your dance movements in real time."
            ),
            # Optional: microphone permission for future clap-by-sound feature
            # "NSMicrophoneUsageDescription": (
            #     "ParaParaPy uses your microphone to detect clap timing."
            # ),
            # Minimum system version
            "LSMinimumSystemVersion": "10.15",
        },
        codesign=None,
        entitlements_file=None,
    )

elif sys.platform == "win32":
    # Windows executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="ParaParaPy",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=False,  # No console window
        icon="assets/branding/paraparapy.ico",
        uac_admin=False,
        uac_uiaccess=False,
    )

else:
    # Linux (AppImage or directory)
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="ParaParaPy",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=False,
    )
