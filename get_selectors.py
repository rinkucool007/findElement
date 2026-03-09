# get_selectors.py — Playwright + Ollama/LangChain website explorer
#
# Explores a website following user-defined steps, then extracts CSS
# selectors and XPath expressions into a JSON file.
#
# ── Recommended local Ollama models ─────────────────────────────────
#   qwen2.5-coder:14b  ★ BEST — purpose-built for code/automation
#                         ~10 GB VRAM, fast, highly accurate
#   qwen3:14b              Latest Qwen3 architecture, great reasoning
#                         ~9 GB VRAM
#   qwen2.5:32b            Highest quality, needs ~20 GB VRAM
#   deepseek-coder-v2:16b  Strong code model, ~10 GB VRAM
#   llama3.3:70b           Top tier quality, needs ~40 GB VRAM
#
# ── Install ──────────────────────────────────────────────────────────
#   pip install playwright langchain-ollama
#   playwright install
#   ollama pull qwen2.5-coder:14b
# ─────────────────────────────────────────────────────────────────────

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from playwright.async_api import async_playwright, Page
from langchain_ollama import OllamaLLM  # pip install langchain-ollama

# ── Recommended model — change to qwen3:14b or qwen2.5:32b if you
#    have more VRAM.  qwen2.5-coder:14b hits the sweet spot for
#    code / automation tasks on consumer hardware.
DEFAULT_MODEL = "qwen2.5-coder:14b"

# ─────────────────────────────────────────────────────────────────────
# Async-safe exec
# ─────────────────────────────────────────────────────────────────────

async def safe_execute_action(page: Page, code_snippet: str) -> str:
    """
    Execute a single ``await page.*`` Playwright expression safely.

    The standard ``exec()`` call cannot handle ``await`` outside an
    async context, causing the classic
    ``SyntaxError: 'await' outside function`` error.
    The fix: wrap the line in a tiny ``async def``, exec that
    definition, then *await* the resulting coroutine.
    """
    code_snippet = code_snippet.strip().rstrip(";")
    # Strip markdown code fences if the LLM wrapped the line
    code_snippet = re.sub(r"^```[a-z]*\n?|```$", "", code_snippet, flags=re.M).strip()

    if not code_snippet.startswith("await page."):
        return "Skipped: line does not start with 'await page.'"

    wrapper = (
        "async def _action(page):\n"
        f"    return {code_snippet}\n"
    )
    ns: Dict[str, Any] = {}
    try:
        exec(wrapper, ns)                           # defines _action in ns
        result = await ns["_action"](page)          # actually awaits it
        return f"Success: {result!r}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────
# DOM-level selector harvesting via JavaScript
# ─────────────────────────────────────────────────────────────────────

# Injected into the page; returns a structured object of real,
# DOM-verified selectors — no guessing from LLM output needed.
_DOM_EXTRACTOR_JS = """() => {
  const seen  = new Set();
  const out   = { css: [], xpath: [], role: [], text: [], links: [] };
  const uniq  = arr => [...new Set(arr.filter(Boolean))];

  // ── Unique CSS path ─────────────────────────────────────────────
  function cssPath(el) {
    if (el.id) return '#' + CSS.escape(el.id);
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentNode) {
      if (n.id) { parts.unshift('#' + CSS.escape(n.id)); break; }
      let seg = n.tagName.toLowerCase();
      if (n.className) {
        seg += [...n.classList].slice(0, 2)
                               .map(c => '.' + CSS.escape(c))
                               .join('');
      }
      const sibs = [...(n.parentNode?.children || [])]
                     .filter(s => s.tagName === n.tagName);
      if (sibs.length > 1) seg += ':nth-of-type(' + (sibs.indexOf(n) + 1) + ')';
      parts.unshift(seg);
      if (parts.length >= 4) break;
    }
    return parts.join(' > ');
  }

  // ── Minimal XPath ───────────────────────────────────────────────
  function xPath(el) {
    if (el.id) return `//*[@id="${el.id}"]`;
    const parts = [];
    for (let n = el; n && n.nodeType === 1; n = n.parentNode) {
      if (n.id) { parts.unshift(`//*[@id="${n.id}"]`); break; }
      const tag  = n.tagName.toLowerCase();
      const sibs = [...(n.parentNode?.children || [])]
                     .filter(s => s.tagName === n.tagName);
      parts.unshift(tag + (sibs.length > 1 ? `[${sibs.indexOf(n)+1}]` : ''));
      if (parts.length >= 5) break;
    }
    return (parts[0] || '').startsWith('//*')
      ? parts.join('/')
      : '/' + parts.join('/');
  }

  const SELECTORS = [
    'a[href]','button','input','textarea','select',
    '[role="button"]','[role="link"]','[role="tab"]',
    '[data-testid]','[aria-label]'
  ];

  document.querySelectorAll(SELECTORS.join(',')).forEach(el => {
    const key = el.tagName + '|' + el.id + '|' + el.className;
    if (seen.has(key)) return;
    seen.add(key);

    out.css.push(cssPath(el));
    out.xpath.push(xPath(el));

    const role  = el.getAttribute('role') || el.tagName.toLowerCase();
    const label = (el.getAttribute('aria-label') || el.innerText || '')
                    .trim().slice(0, 60);
    if (label) out.role.push({ role, label });

    const txt = (el.innerText || '').trim().slice(0, 80);
    if (txt && !['INPUT','TEXTAREA','SELECT'].includes(el.tagName))
      out.text.push(txt);

    if (el.tagName === 'A' && el.href && !el.href.startsWith('javascript'))
      out.links.push({ text: (el.innerText || '').trim(), href: el.href });
  });

  out.css   = uniq(out.css);
  out.xpath = uniq(out.xpath);
  out.text  = uniq(out.text);
  return out;
}"""


async def harvest_dom_selectors(page: Page) -> Dict[str, Any]:
    """Inject JS into the live page and return real, DOM-verified selectors."""
    try:
        return await page.evaluate(_DOM_EXTRACTOR_JS)
    except Exception as exc:
        print(f"  [DOM harvest error] {exc}")
        return {"css": [], "xpath": [], "role": [], "text": [], "links": []}


def _merge_dom(collected: Dict[str, Any], dom: Dict[str, Any]) -> None:
    """Merge a DOM harvest result into the running collected dict."""
    for key in ("css", "xpath", "text"):
        collected.setdefault(key, []).extend(dom.get(key, []))
    for item in dom.get("role", []):
        entry = f"role={item['role']!r} label={item['label']!r}"
        collected.setdefault("role", []).append(entry)
    for link in dom.get("links", []):
        if link.get("href"):
            collected.setdefault("links", []).append(link)


# ─────────────────────────────────────────────────────────────────────
# Heuristic extraction from LLM code snippets (bonus selectors)
# ─────────────────────────────────────────────────────────────────────

def extract_locators_from_snippet(code: str) -> Dict[str, List[str]]:
    sel: Dict[str, Set[str]] = {"css": set(), "xpath": set(), "role": set(), "text": set()}

    for m in re.finditer(r'get_by_role\(["\']([^"\']+)["\'].*?name=["\']([^"\']+)["\']', code):
        sel["role"].add(f"role={m.group(1)!r} name={m.group(2)!r}")
    for m in re.finditer(r'get_by_text\(["\']([^"\']+)["\']', code):
        sel["text"].add(m.group(1))
    for m in re.finditer(r'locator\(["\']([^"\']+)["\']', code):
        s = m.group(1).strip()
        if s.startswith(("//", "xpath=")):
            sel["xpath"].add(s.removeprefix("xpath="))
        else:
            sel["css"].add(s)

    return {k: sorted(v) for k, v in sel.items() if v}


# ─────────────────────────────────────────────────────────────────────
# Lightweight HTML summary (strips noise, keeps interactive markup)
# ─────────────────────────────────────────────────────────────────────

async def get_html_summary(page: Page, max_len: int = 4000) -> str:
    html = await page.content()
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    return html[:max_len] + ("…" if len(html) > max_len else "")


# ─────────────────────────────────────────────────────────────────────
# Main explorer
# ─────────────────────────────────────────────────────────────────────

async def explore_website(
    start_url: str,
    user_steps: List[str],
    model_name: str = DEFAULT_MODEL,
    max_attempts_per_step: int = 3,
    output_file: str = "selectors.json",
    headless: bool = False,
) -> None:
    """
    Navigate *start_url* following *user_steps*, ask Ollama LLM for each
    Playwright action, harvest real CSS/XPath from the live DOM after
    every action, and write everything to *output_file*.
    """
    llm = OllamaLLM(model=model_name, temperature=0.1)

    collected: Dict[str, Any] = {
        "css": [], "xpath": [], "role": [], "text": [], "links": [], "history": []
    }

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless, slow_mo=350)
        ctx     = await browser.new_context()
        page: Page = await ctx.new_page()

        await page.goto(start_url, wait_until="domcontentloaded", timeout=45_000)
        print(f"\n→ Loaded: {page.url}")

        # ── Initial DOM harvest before any navigation
        dom = await harvest_dom_selectors(page)
        _merge_dom(collected, dom)
        print(f"  Initial DOM: {len(dom['css'])} CSS, "
              f"{len(dom['xpath'])} XPath, {len(dom['links'])} links")

        for step_idx, instruction in enumerate(user_steps, 1):
            print(f"\n── Step {step_idx}/{len(user_steps)}: {instruction}")
            current_url   = page.url

            # ── Link / explore shortcut ─────────────────────────────
            if re.search(r"\blinks?\b|\bnavigate\b|\bexplore\b", instruction, re.I):
                print("  → Link-exploration mode (harvesting all <a> elements)")
                dom   = await harvest_dom_selectors(page)
                links = dom.get("links", [])
                _merge_dom(collected, dom)
                deduped = list({lk["href"]: lk for lk in collected["links"]}.values())
                collected["links"] = deduped
                collected["history"].append(
                    f"Step {step_idx}: Explored {len(deduped)} unique links"
                )
                print(f"  Collected {len(deduped)} unique links")
                continue

            # ── LLM-guided Playwright action ────────────────────────
            title        = await page.title()
            html_summary = await get_html_summary(page)
            recent_hist  = "\n".join(collected["history"][-4:]) or "(none)"

            prompt = f"""You are a Playwright Python automation expert.
Page: "{title}" ({current_url})
Task: {instruction}

Recent history:
{recent_hist}

Relevant page HTML (truncated):
{html_summary}

Write EXACTLY ONE Python line starting with `await page.` that performs the task.
Use stable locators in this priority:
1. page.get_by_role(role, name="...")
2. page.get_by_placeholder("...")  /  page.get_by_label("...")
3. page.get_by_text("...", exact=True)
4. page.locator("[data-testid=...]") or page.locator("#id")
5. page.locator("xpath=//...")  — only as last resort

Examples:
await page.get_by_role("button", name="Login").click()
await page.get_by_placeholder("Username").fill("admin")
await page.get_by_role("link", name="Products").click()

Your single line (no explanation, no markdown fences):"""

            success = False
            for attempt in range(1, max_attempts_per_step + 1):
                try:
                    raw = (await llm.ainvoke(prompt)).strip()

                    # Prefer the first line that looks like a Playwright call
                    lines = [
                        ln.strip()
                        for ln in raw.splitlines()
                        if ln.strip().startswith("await page.")
                    ]
                    code_line = lines[0] if lines else raw

                    print(f"  Attempt {attempt}: {code_line}")

                    result = await safe_execute_action(page, code_line)
                    print(f"  → {result}")

                    collected["history"].append(
                        f"Step {step_idx}.{attempt}: {code_line} → {result}"
                    )

                    # Extract heuristic locators from the LLM snippet
                    for k, v in extract_locators_from_snippet(code_line).items():
                        collected.setdefault(k, []).extend(v)

                    if "Error" not in result and "failed" not in result.lower():
                        success = True
                        # Wait for page to settle, then harvest updated DOM
                        try:
                            await page.wait_for_load_state("domcontentloaded",
                                                           timeout=6_000)
                        except Exception:
                            pass
                        dom = await harvest_dom_selectors(page)
                        _merge_dom(collected, dom)
                        break

                except Exception as exc:
                    print(f"  LLM/browser error: {exc}")
                    break

            if not success:
                print(f"  ⚠  Step {step_idx} did not succeed "
                      f"after {max_attempts_per_step} attempts")

            if page.url != current_url:
                print(f"  → Navigated to: {page.url}")

        await browser.close()

    # ── Deduplicate everything before saving ─────────────────────────
    for key in ("css", "xpath", "role", "text"):
        collected[key] = sorted(set(collected[key]))
    collected["links"] = list(
        {lk["href"]: lk for lk in collected.get("links", [])}.values()
    )

    out_path = Path(output_file)
    out_path.write_text(json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Selectors saved → {out_path.resolve()}")
    print(f"  CSS: {len(collected['css'])}  "
          f"XPath: {len(collected['xpath'])}  "
          f"Links: {len(collected['links'])}  "
          f"Role: {len(collected['role'])}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TARGET_URL = "https://shoppingkart-niwf.onrender.com"

    user_steps = [
        "Click on the login button",
        "Fill in the username field with 'admin' and password field with 'password'",
        "Explore all links on the page",
    ]

    asyncio.run(
        explore_website(
            start_url=TARGET_URL,
            user_steps=user_steps,
            model_name=DEFAULT_MODEL,   # swap to "qwen3:14b" or "qwen2.5:32b" as needed
            max_attempts_per_step=3,
            output_file="selectors.json",
            headless=False,             # set True for CI / server environments
        )
    )