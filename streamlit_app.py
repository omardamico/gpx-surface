"""
Avvio dell'app come eseguibile (PyInstaller). In sviluppo resta valido: streamlit run app.py
"""
import os
import sys
import threading
import webbrowser

from streamlit.web import cli as stcli


def resource_path(relative: str) -> str:
    # Nell'eseguibile i file stanno nella cartella temporanea/_internal di PyInstaller
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


PORT = 8501


if __name__ == "__main__":
    # Headless evita il prompt email del primo avvio; il browser lo apro io appena il server è su
    threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    sys.argv = [
        "streamlit", "run", resource_path("app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
        # Tema inline: nell'eseguibile .streamlit/config.toml non viene cercato accanto all'app
        "--theme.base=light",
        "--theme.primaryColor=#007bff",
        "--theme.backgroundColor=#ffffff",
        "--theme.secondaryBackgroundColor=#f8f9fa",
        "--theme.textColor=#212529",
    ]
    sys.exit(stcli.main())