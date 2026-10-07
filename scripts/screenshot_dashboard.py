"""Save a full-page screenshot of the running dashboard to docs/img/dashboard.png.

Usage: start the dashboard (thermo-fr dashboard), then
    python scripts/screenshot_dashboard.py [http://localhost:8501] [docs/img/dashboard.png]
Needs seleniumbase (pip install seleniumbase) and a local Chrome.
"""

import sys
import time
from pathlib import Path

from seleniumbase import Driver


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img/dashboard.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    driver = Driver(browser="chrome", headless=True)
    try:
        # Streamlit scrolls inside its own container, so a tall window is the simple way to a full-page capture.
        driver.set_window_size(1500, 3600)
        driver.get(url)
        time.sleep(15)  # let Streamlit render the charts
        driver.save_screenshot(str(out))
        print(f"saved {out}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
