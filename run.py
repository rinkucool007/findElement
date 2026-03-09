"""
run.py — End-to-end launcher for the Playwright + Ollama selector explorer.

What this script does:
  1. Checks that Python >= 3.10 is available.
  2. Installs missing pip packages (playwright, langchain-ollama).
  3. Installs Playwright browsers if not already present.
  4. Confirms Ollama is running (starts a human-readable error if not).
  5. Pulls the chosen model if it is not cached locally.
  6. Runs explore_website() with the user-defined steps.
  7. Pretty-prints a summary and opens selectors.json when done.

Usage:
    python run.py                        # uses defaults below
    python run.py --model qwen3:14b      # override model
    python run.py --url https://...      # override target URL
    python run.py --headless             # run browser headlessly
"""

import argparse
import json
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ─── User configuration ───────────────────────────────────────────────────────
DEFAULT_URL   = "https://shoppingkart-niwf.onrender.com"
DEFAULT_MODEL = "qwen3.5:27b"   # change to qwen3:14b if you prefer

USER_STEPS = [
    "Click on the login button",
    "Fill in the username field with 'admin' and password field with 'password'",
    "Explore all links on the page",
]

OUTPUT_FILE = "selectors.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── ANSI colours (safe on Windows 10+ with ANSI enabled) ─────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"{GREEN}  [OK]  {msg}{RESET}")
def warn(msg: str) -> None: print(f"{YELLOW}  [!!]  {msg}{RESET}")
def err(msg: str)  -> None: print(f"{RED}  [ERR] {msg}{RESET}")
def info(msg: str) -> None: print(f"{CYAN}  [ ->] {msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Python version
# ─────────────────────────────────────────────────────────────────────────────

def check_python() -> None:
    print(f"\n{BOLD}[1/5] Python version{RESET}")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        err(f"Python {major}.{minor} found — need ≥ 3.10")
        sys.exit(1)
    ok(f"Python {major}.{minor}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — pip packages
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "playwright":       "playwright",
    "langchain_ollama": "langchain-ollama",
}

def ensure_packages() -> None:
    print(f"\n{BOLD}[2/5] Python packages{RESET}")
    missing = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
            ok(pip_name)
        except ImportError:
            warn(f"{pip_name} not found — will install")
            missing.append(pip_name)

    if missing:
        info(f"Running: pip install {' '.join(missing)}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing],
            capture_output=False,
        )
        if result.returncode != 0:
            err("pip install failed — check your internet connection or venv.")
            sys.exit(1)
        ok(f"Installed: {', '.join(missing)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Playwright browsers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_playwright_browsers() -> None:
    print(f"\n{BOLD}[3/5] Playwright browsers{RESET}")
    # If chromium executable exists, skip; otherwise install
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # Accessing the executable path will raise if not installed
            _ = p.chromium.executable_path
        ok("Chromium already installed")
    except Exception:
        info("Installing Playwright Chromium browser …")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "--with-deps"],
            capture_output=False,
        )
        if result.returncode != 0:
            err("playwright install failed.")
            sys.exit(1)
        ok("Chromium installed")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Ollama health check
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"

def check_ollama() -> None:
    print(f"\n{BOLD}[4/5] Ollama service{RESET}")
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as resp:
            if resp.status == 200:
                ok("Ollama is running at localhost:11434")
                return
    except urllib.error.URLError:
        pass

    err("Ollama is NOT running.")
    print(
        f"\n  {YELLOW}Start Ollama and try again:{RESET}\n"
        "    Windows:  double-click the Ollama tray icon, or run\n"
        "              'ollama serve' in a separate terminal\n"
        "    WSL/Linux: ollama serve &\n"
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Model availability
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model(model_name: str) -> None:
    print(f"\n{BOLD}[5/5] Ollama model: {model_name}{RESET}")

    # Ask Ollama which models are already pulled
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        local_models = [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        warn(f"Could not list local models: {exc}")
        local_models = []

    # Normalise: "qwen2.5-coder:14b" == "qwen2.5-coder:14b"
    if any(model_name in m or m in model_name for m in local_models):
        ok(f"Model '{model_name}' already cached locally")
        return

    warn(f"Model '{model_name}' not found locally — pulling now …")
    info("This downloads several GB on first run. Please wait …\n")
    result = subprocess.run(["ollama", "pull", model_name], capture_output=False)
    if result.returncode != 0:
        err(f"Failed to pull '{model_name}'. Check the model name and try:\n"
            f"    ollama pull {model_name}")
        sys.exit(1)
    ok(f"Model '{model_name}' ready")


# ─────────────────────────────────────────────────────────────────────────────
# Run the explorer
# ─────────────────────────────────────────────────────────────────────────────

def run_explorer(url: str, model: str, headless: bool) -> None:
    # Import here so we benefit from packages installed above
    import asyncio
    # get_selectors.py lives in the same directory as this script
    sys.path.insert(0, str(Path(__file__).parent))
    from get_selectors import explore_website

    print(f"\n{BOLD}{'-'*60}")
    print(f"{BOLD}  Starting explorer{RESET}")
    print(f"  URL   : {url}")
    print(f"  Model : {model}")
    print(f"  Steps : {len(USER_STEPS)}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{BOLD}{'-'*60}{RESET}\n")

    asyncio.run(
        explore_website(
            start_url=url,
            user_steps=USER_STEPS,
            model_name=model,
            max_attempts_per_step=3,
            output_file=str(Path(__file__).parent / OUTPUT_FILE),
            headless=headless,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Print final summary from selectors.json
# ─────────────────────────────────────────────────────────────────────────────

def print_summary() -> None:
    out_path = Path(__file__).parent / OUTPUT_FILE
    if not out_path.exists():
        warn("selectors.json not found — explorer may have crashed.")
        return

    data = json.loads(out_path.read_text(encoding="utf-8"))

    print(f"\n{BOLD}{'-'*60}")
    print("  RESULTS SUMMARY")
    print(f"{'-'*60}{RESET}")
    print(f"  CSS selectors  : {len(data.get('css',  []))}")
    print(f"  XPath exprs    : {len(data.get('xpath', []))}")
    print(f"  Role locators  : {len(data.get('role',  []))}")
    print(f"  Text locators  : {len(data.get('text',  []))}")
    print(f"  Links found    : {len(data.get('links', []))}")
    print(f"  Steps logged   : {len(data.get('history', []))}")
    print(f"{BOLD}{'-'*60}{RESET}")

    links = data.get("links", [])[:10]
    if links:
        print(f"\n  {BOLD}First 10 links discovered:{RESET}")
        for lk in links:
            text = (lk.get("text") or "").strip()[:40] or "(no text)"
            href = lk.get("href", "")
            print(f"    {GREEN}{text:<42}{RESET} {href}")

    css_sample = data.get("css", [])[:5]
    if css_sample:
        print(f"\n  {BOLD}CSS sample (first 5):{RESET}")
        for s in css_sample:
            print(f"    {s}")

    xpath_sample = data.get("xpath", [])[:5]
    if xpath_sample:
        print(f"\n  {BOLD}XPath sample (first 5):{RESET}")
        for s in xpath_sample:
            print(f"    {s}")

    print(f"\n  {GREEN}Full results → {out_path}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Enable ANSI colours on Windows
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )

    parser = argparse.ArgumentParser(
        description="Playwright + Ollama website selector explorer"
    )
    parser.add_argument("--url",      default=DEFAULT_URL,   help="Target website URL")
    parser.add_argument("--model",    default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--headless", action="store_true",   help="Run browser headlessly")
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip setup checks (faster if env is already set up)",
    )
    args = parser.parse_args()

    print(f"{BOLD}{'='*60}")
    print("  Playwright + Ollama Selector Explorer")
    print(f"{'='*60}{RESET}")

    if not args.skip_checks:
        check_python()
        ensure_packages()
        ensure_playwright_browsers()
        check_ollama()
        ensure_model(args.model)

    run_explorer(url=args.url, model=args.model, headless=args.headless)
    print_summary()


if __name__ == "__main__":
    main()
