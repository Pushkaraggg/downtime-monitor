"""
Agent 1 — UPI Downtime Monitor (3-Layer Detection)
====================================================
Visits downdetector.in every 5 mins, extracts chart data,
detects spikes using 3 layers, and alerts on Telegram.

Layer 1: Extract actual numbers from chart (Highcharts JS)
Layer 2: Statistical spike detection (math — never fails)
Layer 3: AI Vision confirmation (Groq — bonus, can fail safely)

Alert logic:
  - Layer 2 says spike → ALERT (even if AI is down)
  - Layer 2 + AI both say spike → HIGH CONFIDENCE alert
  - Only AI says spike, Layer 2 disagrees → ignore (hallucination)

Spike definition:
  A spike = sudden jump where the latest value is 5x+ higher
  than the average of recent readings, AND the jump happened 
  in one step (not gradual). Example:
    [0, 1, 0, 2, 1, 150] → SPIKE (sudden jump from ~1 to 150)
    [10, 20, 30, 60, 100] → NOT a spike (gradual rise)

Usage:
    python Down_alert.py --test
    python Down_alert.py
"""

import argparse
import base64
import json
import logging
import os
import re
import time
from datetime import datetime

# Load .env file if it exists (keeps keys out of code)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SERVICES = {
    "UPI": "https://downdetector.in/status/unified-payments-interface/",
    "HDFC": "https://downdetector.in/status/hdfc-bank/",
    "ICICI": "https://downdetector.in/status/icici-bank/",
    "SBI": "https://downdetector.in/status/state-bank-of-india-sbi/",
    "Axis": "https://downdetector.in/status/axisbank/",
    "Kotak": "https://downdetector.in/status/kotak-mahindra-bank/",
    "Google Pay": "https://downdetector.in/status/google-pay/",
}
CHECK_INTERVAL = 5 * 60                   # 5 minutes

# Spike detection config
SPIKE_MULTIPLIER = 5        # current must be 5x the recent average
SUDDEN_JUMP_RATIO = 3       # current vs previous reading (sudden = 3x+)
MIN_SPIKE_VALUE = 10         # ignore spikes below this many reports

# Groq Vision API (FREE — https://console.groq.com)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Telegram Bot (FREE — @BotFather)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [Agent-1]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Agent-1")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 2 — STATISTICAL SPIKE DETECTOR (the backbone, never fails)      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class SpikeDetector:
    """
    Detects sudden spikes in a series of numbers.

    Spike = BOTH conditions must be true:
      1. Current value >= 5x the recent average
      2. Jump was sudden (current >= 3x the previous reading)

    NOT a spike:
      - Gradual rise: 10, 20, 30, 60, 100 (each step is <3x previous)
      - Consistently high: 50, 55, 60, 58 (no sudden change)
    """

    def __init__(self):
        self.history = []        # recent chart values
        self.max_history = 20    # keep last 20 readings

    def add_values(self, values: list[float]):
        """Add extracted chart values to history."""
        self.history.extend(values)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def check(self, values: list[float]) -> dict:
        """
        Check a series of chart values for spikes.
        Returns: {is_spike, severity, reason, spike_value, baseline_avg}

        Spike = BOTH conditions true:
          1. Current value >= 5x the baseline average
          2. Jump was sudden (current >= 3x the previous reading)
        """
        result = {
            "is_spike": False,
            "severity": "none",
            "reason": "",
            "spike_value": 0,
            "baseline_avg": 0,
        }

        if len(values) < 3:
            return result

        # Get the recent values (last 5) and the baseline (everything before)
        recent = values[-5:] if len(values) >= 5 else values[-3:]
        baseline = values[:-5] if len(values) > 5 else values[:-3]

        latest = recent[-1]
        previous = recent[-2] if len(recent) >= 2 else 0

        # Calculate baseline average
        if baseline:
            avg = sum(baseline) / len(baseline)
        elif len(self.history) >= 3:
            avg = sum(self.history[-10:]) / len(self.history[-10:])
        else:
            avg = sum(recent[:-1]) / max(len(recent) - 1, 1)

        result["baseline_avg"] = round(avg, 1)
        result["spike_value"] = latest

        # --- Check 1: Is the latest value >= 5x baseline average? ---
        if avg > 0 and latest >= avg * SPIKE_MULTIPLIER:

            # --- Check 2: Was the jump sudden (>= 3x previous)? ---
            if previous > 0 and latest >= previous * SUDDEN_JUMP_RATIO:
                severity = self._get_severity(latest, avg)
                result.update({
                    "is_spike": True,
                    "severity": severity,
                    "reason": (
                        f"Sudden spike: {latest} reports "
                        f"(baseline avg: {avg:.1f}, previous: {previous}, "
                        f"jump: {latest/max(previous,1):.1f}x)"
                    ),
                })
                return result

            elif previous <= avg * 2:
                # Previous was near baseline, current shot up
                severity = self._get_severity(latest, avg)
                result.update({
                    "is_spike": True,
                    "severity": severity,
                    "reason": (
                        f"Spike from baseline: {latest} reports "
                        f"(avg was {avg:.1f}, jumped to {latest})"
                    ),
                })
                return result

        return result

    def _get_severity(self, value: float, avg: float) -> str:
        """Severity based on how big the jump is relative to baseline."""
        if avg <= 0:
            return "LOW"
        ratio = value / avg
        if ratio >= 50:
            return "CRITICAL"
        elif ratio >= 20:
            return "HIGH"
        elif ratio >= 10:
            return "MEDIUM"
        return "LOW"


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  AGENT 1  —  COMPLETE UPI MONITOR                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class Agent1:
    """
    Every 5 minutes:
      1. Opens downdetector in headless Chrome
      2. Dismisses cookie banner, waits for chart
      3. Extracts chart data numbers (Layer 1)
      4. Statistical spike detection on numbers (Layer 2)
      5. AI Vision confirmation on screenshot (Layer 3)
      6. Alert decision + Telegram alert with screenshot
    """

    def __init__(self):
        self.driver = None
        self.last_alert_time = {}           # per-service cooldown
        self.alert_cooldown = 10 * 60
        self.spike_detectors = {name: SpikeDetector() for name in SERVICES}

    # -----------------------------------------------------------------
    # BROWSER
    # -----------------------------------------------------------------
    def _start_browser(self):
        opts = Options()
        # Auto-detect: headless on Linux (GitHub Actions), off-screen on Windows (local)
        if os.name == "nt":
            # Windows: real window off-screen (bypasses Cloudflare)
            opts.add_argument("--window-position=-3000,-3000")
        else:
            # Linux/GitHub Actions: headless (no display available)
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        chrome_bin = os.environ.get("CHROME_BIN")
        if chrome_bin:
            opts.binary_location = chrome_bin
        self.driver = webdriver.Chrome(options=opts)
        # Anti-bot detection
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            """
        })
        log.info("Chrome started (minimized window)")

    def _stop_browser(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    # -----------------------------------------------------------------
    # COOKIE BANNER
    # -----------------------------------------------------------------
    def _dismiss_cookies(self):
        consent_texts = ["i consent", "accept", "agree", "accept all", "got it"]
        for sel in [
            "button.fc-cta-consent",
            "button[aria-label='Consent']",
            "button#onetrust-accept-btn-handler",
            ".cc-btn.cc-dismiss",
        ]:
            try:
                btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                btn.click()
                log.info(f"Cookie dismissed ({sel})")
                time.sleep(2)
                return
            except Exception:
                continue
        try:
            for btn in self.driver.find_elements(By.TAG_NAME, "button"):
                if btn.text.strip().lower() in consent_texts:
                    btn.click()
                    log.info(f"Cookie dismissed ('{btn.text}')")
                    time.sleep(2)
                    return
        except Exception:
            pass

    # -----------------------------------------------------------------
    # LAYER 1 — EXTRACT CHART DATA (actual numbers from Highcharts)
    # -----------------------------------------------------------------
    def _extract_chart_data(self) -> list[float]:
        """
        Pull actual report-count numbers from the Highcharts chart.
        Returns a list of y-values like [0, 1, 0, 2, 1, 0, 150, 200].
        This is the RAW DATA — no AI, no guessing.
        """
        # Strategy 1: React Fiber (Recharts — new Downdetector UI)
        try:
            data = self.driver.execute_script("""
                var wrapper = document.querySelector('.recharts-wrapper');
                if (!wrapper) return null;
                var keys = Object.keys(wrapper);
                var fiberKey = keys.find(function(k) {
                    return k.startsWith('__reactFiber') || k.startsWith('__reactInternalInstance');
                });
                if (!fiberKey) return null;
                var node = wrapper[fiberKey];
                for (var i = 0; i < 20; i++) {
                    if (!node) break;
                    if (node.memoizedProps && node.memoizedProps.data) {
                        var d = node.memoizedProps.data;
                        if (Array.isArray(d) && d.length > 5 && d[0].value !== undefined) {
                            return d.map(function(p) { return p.value || 0; });
                        }
                    }
                    node = node.return;
                }
                return null;
            """)
            if data and len(data) > 0:
                log.info(f"Layer 1: React Fiber → {len(data)} values, last 5: {data[-5:]}")
                return [float(v) for v in data]
        except Exception as e:
            log.debug(f"React Fiber extraction failed: {e}")

        # Strategy 2: Highcharts JS API (older Downdetector UI)
        try:
            data = self.driver.execute_script("""
                if (typeof Highcharts !== 'undefined' && Highcharts.charts) {
                    for (var i = 0; i < Highcharts.charts.length; i++) {
                        var c = Highcharts.charts[i];
                        if (c && c.series && c.series[0]) {
                            return c.series[0].data.map(function(p) {
                                return p.y || 0;
                            });
                        }
                    }
                }
                return null;
            """)
            if data and len(data) > 0:
                log.info(f"Layer 1: Highcharts → {len(data)} values, last 5: {data[-5:]}")
                return [float(v) for v in data]
        except Exception as e:
            log.debug(f"Highcharts extraction failed: {e}")

        log.warning("Layer 1: No chart data extracted")
        return []

    # -----------------------------------------------------------------
    # SCREENSHOT THE CHART
    # -----------------------------------------------------------------
    def _screenshot_chart(self) -> tuple[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"chart_{ts}.png")

        # Fix missing gradient + force chart colors (headless Chrome doesn't resolve CSS vars/gradients)
        self.driver.execute_script("""
            // Inject missing gradient into SVG defs
            var svg = document.querySelector('svg.recharts-surface');
            if (svg) {
                var defs = svg.querySelector('defs');
                if (defs) {
                    var grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
                    grad.setAttribute('id', 'gradientBlueLight');
                    grad.setAttribute('x1', '0'); grad.setAttribute('y1', '0');
                    grad.setAttribute('x2', '0'); grad.setAttribute('y2', '1');
                    var stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                    stop1.setAttribute('offset', '0%');
                    stop1.setAttribute('stop-color', '#22d3ee');
                    stop1.setAttribute('stop-opacity', '0.6');
                    var stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                    stop2.setAttribute('offset', '100%');
                    stop2.setAttribute('stop-color', '#22d3ee');
                    stop2.setAttribute('stop-opacity', '0.05');
                    grad.appendChild(stop1);
                    grad.appendChild(stop2);
                    defs.appendChild(grad);
                }
            }
            // Fix area stroke
            var curves = document.querySelectorAll('.recharts-area-curve');
            for (var i = 0; i < curves.length; i++) {
                curves[i].setAttribute('stroke', '#06b6d4');
                curves[i].setAttribute('stroke-width', '2');
            }
            // Fix baseline dashed line
            var lines = document.querySelectorAll('.recharts-line-curve');
            for (var i = 0; i < lines.length; i++) {
                lines[i].setAttribute('stroke', '#374151');
                lines[i].setAttribute('stroke-width', '2');
                lines[i].setAttribute('stroke-dasharray', '8 4');
            }
        """)
        time.sleep(1)

        # Scroll chart into view first
        self.driver.execute_script("""
            var wrapper = document.querySelector('.recharts-wrapper');
            if (wrapper) {
                wrapper.scrollIntoView({block: 'center'});
            }
        """)
        time.sleep(1)

        rect = self.driver.execute_script("""
            // Find the chart container (heading + recharts wrapper)
            var wrapper = document.querySelector('.recharts-wrapper');
            if (!wrapper) return null;

            // Find the heading above the chart
            var heading = null;
            var el = wrapper;
            for (var i = 0; i < 10; i++) {
                el = el.parentElement;
                if (!el) break;
                var texts = el.querySelectorAll('*');
                for (var j = 0; j < texts.length; j++) {
                    var t = (texts[j].innerText || '').trim();
                    if ((t.includes('problems reported in the last 24 hours') || t.includes('outages reported in the last 24 hours')) && t.length < 200) {
                        heading = texts[j];
                        break;
                    }
                }
                if (heading) break;
            }

            var wRect = wrapper.getBoundingClientRect();
            var top, left, width, bottom;

            if (heading) {
                var hRect = heading.getBoundingClientRect();
                top = hRect.top - 10;
                left = Math.min(hRect.left, wRect.left) - 10;
                width = Math.max(hRect.width, wRect.width) + 20;
            } else {
                top = wRect.top - 40;
                left = wRect.left - 10;
                width = wRect.width + 20;
            }
            bottom = wRect.bottom + 5;

            return {
                x: left + window.scrollX,
                y: top + window.scrollY,
                width: Math.min(width, window.innerWidth),
                height: bottom - top
            };
        """)

        if rect and rect.get("width", 0) > 100 and rect.get("height", 0) > 100:
            log.info(f"Chart box: {rect['width']:.0f}x{rect['height']:.0f}")
            cdp_result = self.driver.execute_cdp_cmd("Page.captureScreenshot", {
                "format": "png",
                "clip": {
                    "x": rect["x"],
                    "y": rect["y"],
                    "width": rect["width"],
                    "height": rect["height"],
                    "scale": 2,
                },
            })
            b64 = cdp_result["data"]
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            log.info(f"Screenshot: {path}")
            return path, b64

        self.driver.save_screenshot(path)
        log.info("Screenshot: full viewport (fallback)")
        return self._read_b64(path)

    def _read_b64(self, path: str) -> tuple[str, str]:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return path, b64

    # -----------------------------------------------------------------
    # LAYER 3 — AI CHART ANALYSIS (Groq Vision — bonus, can fail)
    # -----------------------------------------------------------------
    def _ai_analyse(self, screenshot_b64: str) -> dict:
        if not GROQ_API_KEY:
            return {}

        prompt = (
            "You are monitoring outages in India via downdetector.in.\n"
            "This chart shows outage reports over 24 hours.\n"
            "The dashed line is the normal baseline.\n\n"
            "CRITICAL — ONLY LOOK AT THE FAR RIGHT EDGE OF THE CHART:\n"
            "- The far RIGHT side = the CURRENT time (right now).\n"
            "- The LEFT and MIDDLE of the chart = OLD data (hours ago).\n"
            "- IGNORE any spikes in the left or middle — they already happened.\n"
            "- is_spike = true ONLY if the RIGHT EDGE is spiking RIGHT NOW.\n\n"
            "STRICT SPIKE RULES:\n"
            "- is_spike = true ONLY if the FAR RIGHT of the chart shows a MASSIVE spike.\n"
            "- The spike must be AT LEAST 5x higher than the average baseline.\n"
            "- Small bumps (0-10 reports) = NOT a spike. NEVER.\n"
            "- Values under 20 reports = NOT a spike.\n"
            "- If the spike happened earlier but the right edge is back to normal → is_spike: FALSE.\n\n"
            "EXAMPLES:\n"
            "- Spike at 11 AM but now at 7 PM chart is flat → is_spike: FALSE (old spike)\n"
            "- Right edge shooting up to 50+ right now → is_spike: TRUE\n"
            "- Baseline 0-3, right edge at 5 → is_spike: FALSE (noise)\n"
            "- Baseline 0-3, right edge at 100+ → is_spike: TRUE\n\n"
            "When in doubt, say is_spike: false. Better to miss than false alarm.\n\n"
            "Respond with ONLY JSON:\n"
            '{"is_spike": true/false, '
            '"right_edge_value": <current value at far right>, '
            '"peak_reports": <highest peak on chart>, '
            '"summary": "<1-2 sentences about what the RIGHT EDGE shows>"}'
        )

        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{screenshot_b64}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    "max_tokens": 400,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                log.info(f"Layer 3 AI: {raw[:200]}")
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    return json.loads(m.group())
            else:
                log.error(f"Layer 3 AI: HTTP {resp.status_code}")
        except Exception as e:
            log.error(f"Layer 3 AI failed: {e}")

        return {}

    # -----------------------------------------------------------------
    # TELEGRAM
    # -----------------------------------------------------------------
    def _send_telegram_message(self, message: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            log.error(f"Telegram error: {e}")

    def _send_telegram_photo(self, photo_path: str, caption: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            with open(photo_path, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "caption": caption,
                        "parse_mode": "HTML",
                    },
                    files={"photo": f},
                    timeout=15,
                )
        except Exception as e:
            log.error(f"Telegram photo error: {e}")

    # -----------------------------------------------------------------
    # ALERT DECISION + DISPATCH
    # -----------------------------------------------------------------
    def _alert(self, service_name: str, service_url: str, reason: str, severity: str, details: dict, screenshot_path: str):
        now = time.time()
        last = self.last_alert_time.get(service_name, 0.0)
        if now - last < self.alert_cooldown:
            log.info(f"Alert suppressed for {service_name} (cooldown)")
            return
        self.last_alert_time[service_name] = now

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Console
        print("\n" + "=" * 60)
        print(f"  *** {service_name} OUTAGE DETECTED ***")
        print(f"  Severity  : {severity}")
        print(f"  Reason    : {reason}")
        print(f"  Detected  : {details.get('detected_by', '?')}")
        print(f"  Time      : {ts}")
        print("=" * 60 + "\n")

        # Telegram — single message with screenshot embedded
        ai_summary = details.get("ai_summary", "Sudden spike detected")
        msg = (
            f"<b>🚨 {service_name} OUTAGE DETECTED</b>\n\n"
            f"<b>AI says:</b> {ai_summary}\n"
            f"<b>Time:</b> {ts}\n\n"
            f"🔗 <a href='{service_url}'>Check Downdetector</a>"
        )

        # Send screenshot with the alert as caption (single message)
        self._send_telegram_photo(screenshot_path, msg)
        log.info("Alert sent to Telegram")

    # -----------------------------------------------------------------
    # SINGLE CHECK CYCLE (all 3 layers)
    # -----------------------------------------------------------------
    def check_once(self, service_name: str, service_url: str) -> dict:
        log.info(f"[{service_name}] Loading {service_url}")
        self.driver.get(service_url)

        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(2)

        self._dismiss_cookies()
        time.sleep(2)

        # Scroll to chart to trigger lazy loading
        self.driver.execute_script("window.scrollTo(0, 400);")
        time.sleep(2)

        # Wait for Recharts chart to render
        chart_ready = False
        for attempt in range(20):
            chart_ready = self.driver.execute_script("""
                // Check for Recharts (new Downdetector UI)
                var rc = document.querySelector('.recharts-wrapper svg.recharts-surface');
                if (rc && rc.querySelectorAll('path').length > 0) return 'recharts';
                // Fallback: any large SVG with paths
                var svgs = document.querySelectorAll('svg');
                for (var i = 0; i < svgs.length; i++) {
                    var r = svgs[i].getBoundingClientRect();
                    if (r.width > 300 && r.height > 100 && svgs[i].querySelectorAll('path').length > 2) return 'svg';
                }
                return false;
            """)
            if chart_ready:
                log.info(f"[{service_name}] Chart rendered via {chart_ready} (attempt {attempt + 1})")
                break
            # Try dismissing cookies again mid-wait (may be blocking)
            if attempt == 5:
                self._dismiss_cookies()
                self.driver.execute_script("window.scrollTo(0, 400);")
            time.sleep(2)
        if not chart_ready:
            log.warning(f"[{service_name}] Chart did not render after 40s")
        # Extra wait for animations to finish
        time.sleep(3)

        # ========================
        # LAYER 1: Try to extract chart data (quick attempt)
        # ========================
        chart_values = []
        for attempt in range(3):
            chart_values = self._extract_chart_data()
            if chart_values:
                log.info(f"[{service_name}] Layer 1: Got {len(chart_values)} values (attempt {attempt + 1})")
                break
            time.sleep(2)
        if not chart_values:
            log.info(f"[{service_name}] Layer 1: No data — relying on AI")

        # ========================
        # LAYER 2: Statistical spike detection (5x avg + 3x previous)
        # ========================
        stats_result = {"is_spike": False}
        if chart_values:
            detector = self.spike_detectors[service_name]
            detector.add_values(chart_values)
            stats_result = detector.check(chart_values)
            if stats_result["is_spike"]:
                log.warning(f"[{service_name}] Layer 2 SPIKE: {stats_result['reason']}")
            else:
                log.info(f"[{service_name}] Layer 2: No spike (latest={stats_result['spike_value']}, avg={stats_result['baseline_avg']})")

        # Screenshot the chart
        ss_path, ss_b64 = self._screenshot_chart()

        # ========================
        # LAYER 3: AI Vision (bonus confirmation)
        # ========================
        ai_result = self._ai_analyse(ss_b64)
        ai_spike = ai_result.get("is_spike", False)
        if ai_spike:
            log.info(f"[{service_name}] Layer 3 AI: SPIKE — {ai_result.get('summary', '?')}")
        else:
            log.info(f"[{service_name}] Layer 3 AI: No spike — {ai_result.get('summary', 'N/A')}")

        # ========================
        # ALERT DECISION
        # Priority 1: Stats spike (Layer 2) → always alert
        # Priority 2: AI spike (Layer 3) when Layer 1 has no data → trust AI
        # ========================
        stats_spike = stats_result.get("is_spike", False)

        if stats_spike:
            detected_by = "Stats"
            if ai_spike:
                detected_by += " + AI"
            self._alert(
                service_name=service_name,
                service_url=service_url,
                reason=stats_result["reason"],
                severity=stats_result.get("severity", "LOW"),
                details={
                    "detected_by": detected_by,
                    "ai_summary": ai_result.get("summary", ""),
                },
                screenshot_path=ss_path,
            )
        elif ai_spike and not chart_values:
            # Layer 1 failed, but AI sees a spike — trust it
            log.warning(f"[{service_name}] AI detected spike (no chart data to verify)")
            self._alert(
                service_name=service_name,
                service_url=service_url,
                reason=f"AI detected spike: {ai_result.get('summary', 'Spike detected')}",
                severity="LOW",
                details={
                    "detected_by": "AI (no chart data)",
                    "ai_summary": ai_result.get("summary", ""),
                },
                screenshot_path=ss_path,
            )
        else:
            log.info(f"[{service_name}] All clear — no spike detected.")

        return {
            "stats": stats_result,
            "ai": ai_result,
            "chart_values_count": len(chart_values),
        }

    # -----------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------
    def run(self):
        log.info("=" * 50)
        log.info("Agent 1 — Downtime Monitor")
        log.info(f"Services : {', '.join(SERVICES.keys())}")
        log.info(f"Interval : {CHECK_INTERVAL // 60} minutes")
        log.info(f"AI       : {'ON' if GROQ_API_KEY else 'OFF'}")
        log.info(f"Telegram : {'ON' if TELEGRAM_BOT_TOKEN else 'OFF'}")
        log.info(f"Detection: Layer 1 (data) + Layer 2 (5x avg + 3x prev) + Layer 3 (AI)")
        log.info("=" * 50)

        self._start_browser()

        services_list = "\n".join(f"  • {name}" for name in SERVICES)
        self._send_telegram_message(
            "✅ <b>Downtime Monitor Online</b>\n\n"
            f"<b>Monitoring:</b>\n{services_list}\n\n"
            f"Checking every {CHECK_INTERVAL // 60} min\n"
            "3-layer detection: Data + Stats + AI\n"
            "You'll get alerts + chart screenshot if any service goes down."
        )

        try:
            while True:
                for name, url in SERVICES.items():
                    try:
                        self.check_once(name, url)
                    except Exception as e:
                        log.error(f"[{name}] Check failed: {e}")
                        self._stop_browser()
                        self._start_browser()
                    time.sleep(5)  # small delay between services

                log.info(f"Next check in {CHECK_INTERVAL // 60} min\n")
                time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        finally:
            self._stop_browser()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  TEST MODE                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_test():
    print("\n" + "=" * 60)
    print("  Agent 1 — TEST MODE")
    print("=" * 60)

    agent = Agent1()
    passed = 0
    failed = 0

    # --- Test 1: Spike Detection Logic ---
    print("\n  [1/3] Layer 2 — Statistical Spike Detection")
    print("  " + "-" * 40)
    try:
        detector = SpikeDetector()

        normal = [0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1]
        r1 = detector.check(normal)
        print(f"    Normal [0,1,0,2,1...] → spike={r1['is_spike']}  (expected: False)")

        gradual = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
        r2 = detector.check(gradual)
        print(f"    Gradual [5→100]       → spike={r2['is_spike']}  (expected: False)")

        spike = [0, 1, 0, 2, 1, 0, 1, 0, 2, 1, 150]
        r3 = detector.check(spike)
        print(f"    Spike [0,1,0→150]     → spike={r3['is_spike']}  (expected: True)")
        if r3["is_spike"]:
            print(f"    Severity: {r3['severity']}, Reason: {r3['reason']}")

        if not r1["is_spike"] and not r2["is_spike"] and r3["is_spike"]:
            print(f"    PASS — all 3 tests correct")
            passed += 1
        else:
            print(f"    FAIL — detection logic error")
            failed += 1
    except Exception as e:
        print(f"    Error : {e}")
        print(f"    FAIL")
        failed += 1

    # --- Test 2: Telegram Bot ---
    print("\n  [2/3] Telegram Bot Connection")
    print("  " + "-" * 40)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"    Not configured — SKIPPED")
    else:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
                timeout=10,
            )
            bot = r.json()
            if bot.get("ok"):
                print(f"    Bot : @{bot['result']['username']}  — PASS")
                passed += 1
            else:
                print(f"    Invalid token — FAIL")
                failed += 1
        except Exception as e:
            print(f"    Error : {e}")
            failed += 1

    # --- Test 3: All Services — Browser + Data + AI + Telegram Alert ---
    print(f"\n  [3/3] Testing ALL {len(SERVICES)} services")
    print("  " + "-" * 40)
    try:
        agent._start_browser()
        for svc_name, svc_url in SERVICES.items():
            print(f"\n    --- {svc_name} ---")
            try:
                agent.driver.get(svc_url)
                WebDriverWait(agent.driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(2)
                agent._dismiss_cookies()
                time.sleep(2)
                agent.driver.execute_script("window.scrollTo(0, 400);")
                time.sleep(2)

                for attempt in range(20):
                    chart_ready = agent.driver.execute_script("""
                        var rc = document.querySelector('.recharts-wrapper svg.recharts-surface');
                        if (rc && rc.querySelectorAll('path').length > 0) return true;
                        var svgs = document.querySelectorAll('svg');
                        for (var i = 0; i < svgs.length; i++) {
                            var r = svgs[i].getBoundingClientRect();
                            if (r.width > 300 && r.height > 100 && svgs[i].querySelectorAll('path').length > 2) return true;
                        }
                        return false;
                    """)
                    if chart_ready:
                        print(f"    Chart ready (attempt {attempt + 1})")
                        break
                    time.sleep(2)
                time.sleep(3)

                # Layer 1: Extract data
                chart_values = agent._extract_chart_data()
                print(f"    Data points : {len(chart_values)}")
                if chart_values:
                    print(f"    Last 5      : {chart_values[-5:]}")
                    print(f"    Max value   : {max(chart_values)}")

                # Layer 2: Spike check
                result = {"is_spike": False, "severity": "none", "spike_value": 0, "baseline_avg": 0}
                if chart_values:
                    result = agent.spike_detectors[svc_name].check(chart_values)
                    print(f"    Spike       : {result['is_spike']} (latest={result['spike_value']}, avg={result['baseline_avg']})")

                # Screenshot
                ss_path, ss_b64 = agent._screenshot_chart()
                print(f"    Screenshot  : {ss_path}")

                # Layer 3: AI
                ai = {}
                if GROQ_API_KEY and ss_b64:
                    ai = agent._ai_analyse(ss_b64)
                    print(f"    AI summary  : {ai.get('summary', 'N/A')}")

                # Send test alert to Telegram
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ai_summary = ai.get("summary", "No issues detected")
                test_caption = (
                    f"<b>🚨 {svc_name} OUTAGE DETECTED</b>\n"
                    f"<i>(TEST — not a real outage)</i>\n\n"
                    f"<b>AI says:</b> {ai_summary}\n"
                    f"<b>Time:</b> {ts}\n\n"
                    f"🔗 <a href='{svc_url}'>Check Downdetector</a>"
                )
                if ss_path:
                    agent._send_telegram_photo(ss_path, test_caption)
                else:
                    agent._send_telegram_message(test_caption)

                print(f"    Telegram    : sent")
                print(f"    PASS")
                passed += 1

            except Exception as e:
                print(f"    Error : {e}")
                print(f"    FAIL")
                failed += 1

            time.sleep(3)  # small delay between services

        agent._stop_browser()
    except Exception as e:
        print(f"    Browser error: {e}")
        failed += 1
        try:
            agent._stop_browser()
        except Exception:
            pass

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed == 0:
        print("  All systems GO! Run without --test to start monitoring.")
    else:
        print("  Fix failures above, then re-run --test")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 1 — UPI Downtime Monitor")
    parser.add_argument("--test", action="store_true", help="Run tests and exit")
    parser.add_argument("--once", action="store_true", help="Check all services once and exit (for GitHub Actions)")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.once:
        agent = Agent1()
        agent._start_browser()
        try:
            for name, url in SERVICES.items():
                try:
                    agent.check_once(name, url)
                except Exception as e:
                    log.error(f"[{name}] Check failed: {e}")
                    agent._stop_browser()
                    agent._start_browser()
                time.sleep(5)
        finally:
            agent._stop_browser()
        log.info("Single run complete.")
    else:
        Agent1().run()
