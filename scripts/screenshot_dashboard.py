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
        driver.set_window_size(1500, 1000)
        driver.get(url)
        time.sleep(12)  # let Streamlit render the charts
        height = driver.execute_script("return document.documentElement.scrollHeight")
        driver.set_window_size(1500, min(int(height) + 100, 4200))
        time.sleep(3)
        driver.save_screenshot(str(out))
        print(f"saved {out} ({height}px tall)")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
