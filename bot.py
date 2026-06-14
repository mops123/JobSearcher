#!/usr/bin/env python3
"""
IT QA Job Scraper & Filter Bot
================================
Scrapes Austrian job portals (karriere.at, devjobs.at) for IT/Software QA
roles in Vienna, filters out manufacturing/non-IT QA via Google Gemini,
deduplicates re-posted listings, and sends Telegram notifications.

Usage:
    python bot.py

Required environment variables (see .env):
    GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (all secrets come from environment — never hard-coded)
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID: str = os.environ["TELEGRAM_CHAT_ID"]

DB_FILE = "jobs_db.json"
GEMINI_MODEL = "gemini-2.5-flash"

# Polite scraping delays (be a good citizen)
SCRAPE_DELAY_SEC = 2
TELEGRAM_DELAY_SEC = 1
REQUEST_TIMEOUT_SEC = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-AT,de;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# Search keyword configuration
# ---------------------------------------------------------------------------

# karriere.at: slug-based search — spaces become "-", special chars dropped.
# Each keyword is searched against the /wien location scope.
KARRIERE_KEYWORDS = [
    "qa",
    "quality-assurance",
    "software-tester",
    "softwaretester",
    "test-automation",
    "test-engineer",
    "testanalyst",
]

# devjobs.at: slug format is  <keyword>-wien-109166  (Wien location ID = 109166).
# Multi-word keywords are hyphenated; umlauts are transliterated by the portal.
DEVJOBS_KEYWORDS = [
    "qa",
    "quality-assurance",
    "software-tester",
    "softwaretester",
    "test-automation",
    "test-engineer",
    "testanalyst",
]
DEVJOBS_WIEN_ID = "109166"

# ---------------------------------------------------------------------------
# Gemini Classification Prompt  (bilingual DE/EN, temperature=0)
# ---------------------------------------------------------------------------

GEMINI_PROMPT_TEMPLATE = """\
You are a strict job classification expert for the Austrian IT job market.
Your ONLY task: decide whether the job below is an IT / Software QA role.

════════════════════════════════════════════════════════
✅  CLASSIFY AS "YES" — IT / Software QA  (ALLOW)
════════════════════════════════════════════════════════
Titles (English & German):
  Software Tester, Softwaretester, QA Engineer, QA Analyst,
  Quality Assurance Engineer, Test Automation Engineer, SDET,
  Testanalyst, Testingenieur (Software/IT), Agile Tester,
  Scrum Tester, Performance Tester, Security Tester (software),
  Test Manager, Testmanager, QA Lead, QA/QC Engineer (software dev),
  Quality Engineer (Software), Manual Tester, Functional Tester,
  Integration Tester (software context)

Strong YES signals — any of these keywords in the description:
  Selenium, Cypress, Playwright, Appium, WebdriverIO,
  JUnit, TestNG, NUnit, PyTest, Robot Framework,
  JIRA, TestRail, Xray, Zephyr, qTest,
  CI/CD, Jenkins, GitLab CI, GitHub Actions,
  API testing, REST testing, Postman, SoapUI,
  Python, Java, TypeScript, JavaScript (in a testing context),
  test cases, test plans, test strategy, bug reports,
  regression testing, smoke testing, exploratory testing,
  Agile, Scrum, Kanban (in software teams)

════════════════════════════════════════════════════════
❌  CLASSIFY AS "NO" — Manufacturing / Non-IT QA  (BLOCK)
════════════════════════════════════════════════════════
Industry context triggers immediate NO:
  Physical manufacturing, factory floors, heavy industry,
  construction, civil engineering, mechanical engineering,
  pharmaceutical production, food & beverage, medical devices
  (UNLESS the role is explicitly about SOFTWARE validation, e.g. CSV/GAMP)

German block-words — if the description contains ANY of these, answer NO:
  Produktion, Produktionslinie, Produktionsmitarbeiter,
  Fertigung, Fertigungsanlage, Fertigungsqualität,
  Fließband, Schichtarbeit, Schichtdienst,
  Bau, Bauingenieur, Baustelle, Bauqualität,
  Wareneingangsprüfung, Wareneingang,
  Endprüfung, Endkontrolle,
  Reklamationsbearbeitung, Reklamation (manufacturing),
  ISO 9001 (unless software-process context),
  ISO 13485, ISO 14001, GMP, HACCP,
  Laborant, Laborkontrolle, Lebensmittelkontrolle,
  Qualitätssicherung in der Produktion,
  Qualitätsmanager Fertigung

English block-words:
  production line, factory, manufacturing plant,
  construction site, civil engineering QA/QC,
  pharmaceutical manufacturing QA,
  incoming goods inspection, shift work (manufacturing)

════════════════════════════════════════════════════════
🔍  JOB TO CLASSIFY
════════════════════════════════════════════════════════
Title:       {title}
Company:     {company}
Location:    {location}
Description: {description}

════════════════════════════════════════════════════════
📌  YOUR RESPONSE
════════════════════════════════════════════════════════
Reply with exactly ONE word — either YES or NO.
Do not add any explanation, punctuation, or extra words.\
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    description: str
    source: str

    def text_fingerprint(self) -> str:
        """
        SHA-256 of a normalized string built from title + company + the first
        200 chars of description.  Catches re-posts where only the date/URL
        changes but the content is identical.
        """
        raw = (self.title + self.company + self.description[:200]).lower()
        normalized = re.sub(r"[\s\W]+", "", raw)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def url_fingerprint(self) -> str:
        """SHA-256 of the canonical URL (query params stripped)."""
        canonical = re.sub(r"[?#].*$", "", self.url.strip())
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Persistence  (jobs_db.json — committed back to Git by GitHub Actions)
# ---------------------------------------------------------------------------


def load_seen_hashes() -> set[str]:
    if not os.path.exists(DB_FILE):
        logger.info("No existing DB found — starting with empty hash set.")
        return set()
    try:
        with open(DB_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return set(data.get("seen_hashes", []))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s) — starting fresh.", DB_FILE, exc)
        return set()


def save_seen_hashes(hashes: set[str]) -> None:
    payload = {"seen_hashes": sorted(hashes)}
    with open(DB_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("DB saved: %d total hashes in %s", len(hashes), DB_FILE)


# ---------------------------------------------------------------------------
# HTTP utility
# ---------------------------------------------------------------------------


def fetch_page(url: str) -> Optional[BeautifulSoup]:
    """GET a URL and return a parsed BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s — %s", url, exc)
        return None


def _clean(element) -> str:
    """Extract and normalise whitespace from a BS4 element (or return '')."""
    if element is None:
        return ""
    return " ".join(element.get_text(separator=" ", strip=True).split())


# ---------------------------------------------------------------------------
# Scraper — karriere.at
# ---------------------------------------------------------------------------


def _karriere_urls() -> list[str]:
    """Build the full set of karriere.at search URLs from KARRIERE_KEYWORDS."""
    return [
        f"https://www.karriere.at/jobs/{kw}/wien"
        for kw in KARRIERE_KEYWORDS
    ]


def scrape_karriere_at() -> list[Job]:
    """
    Scrape karriere.at for QA jobs in Wien.

    Two parsing strategies are tried in order:
      1. Structured article-card selectors (preferred — survives minor CSS changes).
      2. Link-based fallback that finds all /jobs/<numeric-id> hrefs on the page.
    """
    seen_urls: set[str] = set()
    jobs: list[Job] = []

    for url in _karriere_urls():
        logger.info("karriere.at → %s", url)
        soup = fetch_page(url)
        if soup is None:
            time.sleep(SCRAPE_DELAY_SEC)
            continue

        # ── Strategy 1: article cards ────────────────────────────────────────
        cards = (
            soup.select("article[class*='jobsListItem']")
            or soup.select("article[class*='JobsListItem']")
            or soup.select("div[class*='jobsListItem']")
        )

        if cards:
            for card in cards:
                try:
                    link_el = card.select_one(
                        "h2 a[href*='/jobs/'], a[href*='/jobs/']"
                    )
                    if not link_el:
                        continue
                    href = link_el.get("href", "")
                    if not re.search(r"/jobs/\d+", href):
                        continue

                    job_url = (
                        href
                        if href.startswith("http")
                        else f"https://www.karriere.at{href}"
                    )
                    if job_url in seen_urls:
                        continue
                    seen_urls.add(job_url)

                    title = _clean(link_el)
                    company = _clean(
                        card.select_one(
                            "[class*='company'], [class*='Company'], "
                            "[class*='employer'], [class*='Employer']"
                        )
                    ) or "Unknown"
                    location = _clean(
                        card.select_one(
                            "[class*='location'], [class*='Location'], "
                            "[class*='address'], [class*='Address']"
                        )
                    ) or "Wien"
                    description = _clean(
                        card.select_one(
                            "[class*='description'], [class*='snippet'], "
                            "[class*='summary']"
                        )
                    ) or title

                    if title:
                        jobs.append(
                            Job(
                                title=title,
                                company=company,
                                location=location,
                                url=job_url,
                                description=description,
                                source="karriere.at",
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Card parse error: %s", exc)

        else:
            # ── Strategy 2: link-based fallback ─────────────────────────────
            logger.debug("No article cards on %s — using link fallback", url)
            for link in soup.find_all("a", href=re.compile(r"/jobs/\d+$")):
                href = link.get("href", "")
                job_url = (
                    href
                    if href.startswith("http")
                    else f"https://www.karriere.at{href}"
                )
                if job_url in seen_urls:
                    continue
                seen_urls.add(job_url)

                title = _clean(link)
                if not title or len(title) < 5:
                    continue

                # Grab surrounding parent text for company / location context
                parent_text = _clean(link.find_parent()) or ""
                jobs.append(
                    Job(
                        title=title,
                        company="",
                        location="Wien",
                        url=job_url,
                        description=parent_text or title,
                        source="karriere.at",
                    )
                )

        time.sleep(SCRAPE_DELAY_SEC)

    logger.info("karriere.at: %d jobs collected", len(jobs))
    return jobs


# ---------------------------------------------------------------------------
# Scraper — devjobs.at
# ---------------------------------------------------------------------------

# Job detail links on devjobs.at follow /job/<32-char hex hash>
_DEVJOBS_JOB_HREF = re.compile(r"/job/[a-f0-9]{20,}")


def _devjobs_urls() -> list[str]:
    """Build devjobs.at search URLs: slug format is <keyword>-wien-<location_id>."""
    return [
        f"https://devjobs.at/jobs/{kw}-wien-{DEVJOBS_WIEN_ID}"
        for kw in DEVJOBS_KEYWORDS
    ]


def scrape_devjobs_at() -> list[Job]:
    """
    Scrape devjobs.at for QA/testing jobs in Wien.
    devjobs.at embeds all card text inside a single <a> tag,
    so we parse that text heuristically.
    """
    seen_urls: set[str] = set()
    jobs: list[Job] = []

    for url in _devjobs_urls():
        logger.info("devjobs.at → %s", url)
        soup = fetch_page(url)
        if soup is None:
            time.sleep(SCRAPE_DELAY_SEC)
            continue

        for link in soup.find_all("a", href=_DEVJOBS_JOB_HREF):
            href = link.get("href", "")
            job_url = (
                href if href.startswith("http") else f"https://devjobs.at{href}"
            )
            if job_url in seen_urls:
                continue
            seen_urls.add(job_url)

            full_text = _clean(link)
            if not full_text or len(full_text) < 10:
                continue

            # devjobs card text layout (heuristic):
            # "[Title]  [Company]  [City]  [Description snippet]  [Salary] ..."
            segments = [s.strip() for s in re.split(r"\s{2,}", full_text) if s.strip()]
            title = segments[0] if segments else full_text[:80]
            company = segments[1] if len(segments) > 1 else "Unknown"

            jobs.append(
                Job(
                    title=title,
                    company=company,
                    location="Wien / AT",
                    url=job_url,
                    description=full_text,
                    source="devjobs.at",
                )
            )

        time.sleep(SCRAPE_DELAY_SEC)

    logger.info("devjobs.at: %d jobs collected", len(jobs))
    return jobs


# ---------------------------------------------------------------------------
# Gemini AI classification
# ---------------------------------------------------------------------------


def is_it_qa_role(job: Job, client: genai.Client) -> bool:
    """
    Ask Gemini whether this job is an IT/Software QA role.

    Returns True  → send notification.
    Returns False → skip (manufacturing / non-IT QA, or unrelated).
    Fails open on transient API errors to avoid silently dropping real jobs.
    """
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        title=job.title,
        company=job.company,
        location=job.location,
        description=job.description[:1500],  # stay well within token limits
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0,
            ),
        )
        verdict = response.text.strip().upper().rstrip(".").strip()
        logger.info(
            "Gemini %-55s → %s",
            f"'{job.title[:50]}'",
            verdict,
        )
        return verdict.startswith("YES")
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini API error for '%s': %s — passing through.", job.title, exc)
        return True  # fail open


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------

# MarkdownV2 requires escaping these characters
_MD_ESCAPE = re.compile(r"([_*\[\]()~`>#\+\-=|{}.!\\])")


def _esc(text: str) -> str:
    """Escape a plain string for Telegram MarkdownV2."""
    return _MD_ESCAPE.sub(r"\\\1", str(text))


def send_telegram_alert(job: Job, token: str, chat_id: str) -> bool:
    """Send a formatted MarkdownV2 job alert to the Telegram bot."""
    message = (
        "🆕 *New IT QA Job — Vienna\\!*\n\n"
        f"📌 *{_esc(job.title)}*\n"
        f"🏢 {_esc(job.company)}\n"
        f"📍 {_esc(job.location)}\n"
        f"🌐 Source: {_esc(job.source)}\n\n"
        f"[🔗 View Job]({job.url})"
    )

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": False,
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error("Telegram send failed for '%s': %s", job.title, exc)
        if hasattr(exc, "response") and exc.response is not None:
            logger.error("Telegram API response body: %s", exc.response.text)
        return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("IT QA Job Scraper — run started")
    logger.info("=" * 60)

    # ── Load deduplication database ─────────────────────────────────────────
    seen_hashes = load_seen_hashes()
    logger.info("Loaded %d previously seen hashes from DB", len(seen_hashes))

    # ── Initialise Gemini client ─────────────────────────────────────────────
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Scrape all sources ───────────────────────────────────────────────────
    all_jobs: list[Job] = []
    all_jobs.extend(scrape_karriere_at())
    all_jobs.extend(scrape_devjobs_at())
    logger.info("Total jobs scraped across all sources: %d", len(all_jobs))

    # ── Process each job ─────────────────────────────────────────────────────
    updated_hashes: set[str] = set(seen_hashes)
    sent = skipped_dup = skipped_filter = 0

    for job in all_jobs:
        t_hash = job.text_fingerprint()
        u_hash = job.url_fingerprint()

        # 1. Deduplication — check both text and URL fingerprints
        if t_hash in seen_hashes or u_hash in seen_hashes:
            logger.debug("Duplicate — skip: %s", job.title)
            skipped_dup += 1
            continue

        # 2. AI relevance filter
        if not is_it_qa_role(job, gemini_client):
            logger.info("Non-IT QA — filtered: %s", job.title)
            # Mark as seen so we don't waste Gemini quota re-checking it
            updated_hashes.add(t_hash)
            updated_hashes.add(u_hash)
            skipped_filter += 1
            continue

        # 3. Send Telegram notification
        success = send_telegram_alert(job, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if success:
            logger.info("✓ Sent: %s @ %s", job.title, job.company)
            sent += 1
        else:
            logger.warning("✗ Failed to send: %s", job.title)

        # Mark as seen regardless of send success to avoid re-sending on retry
        updated_hashes.add(t_hash)
        updated_hashes.add(u_hash)

        time.sleep(TELEGRAM_DELAY_SEC)  # stay under Telegram rate limit

    # ── Persist updated database ─────────────────────────────────────────────
    save_seen_hashes(updated_hashes)

    logger.info("=" * 60)
    logger.info(
        "Run complete — Sent: %d | Filtered (non-QA): %d | Duplicates skipped: %d",
        sent,
        skipped_filter,
        skipped_dup,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
