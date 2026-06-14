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
import random
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
# Batch processing: group new jobs before calling Gemini.
# Free tier = 15 RPM; one batch call per ~15 jobs is far cheaper than per-job.
GEMINI_BATCH_SIZE = 15        # jobs per Gemini call
GEMINI_BATCH_DELAY = 2        # seconds to sleep between consecutive batch calls

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-AT,de;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "DNT": "1",
}

# devjobs.at is more aggressive about bot detection — use a stricter header set
# with a Referer that mimics organic navigation from the homepage.
DEVJOBS_HEADERS = {
    **HEADERS,
    "Referer": "https://devjobs.at/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
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

GEMINI_BATCH_PROMPT = """\
You are a strict job-filter assistant for the Austrian IT job market.
You will receive a JSON array of job listings. Evaluate EACH one and return
the IDs of the jobs that are genuine IT / Software QA roles.

=== ALLOW (IT / Software QA) ===
Titles: Software Tester, Softwaretester, QA Engineer, QA Analyst,
  Quality Assurance Engineer, Test Automation Engineer, SDET, Testanalyst,
  Testingenieur (Software/IT), Agile Tester, Scrum Tester, Performance Tester,
  Security Tester (software), Test Manager, Testmanager, QA Lead,
  QA/QC Engineer (software dev), Quality Engineer (Software),
  Manual Tester, Functional Tester, Integration Tester (software context)

Strong ALLOW signals in description:
  Selenium, Cypress, Playwright, Appium, WebdriverIO,
  JUnit, TestNG, NUnit, PyTest, Robot Framework,
  JIRA, TestRail, Xray, Zephyr, qTest,
  CI/CD, Jenkins, GitLab CI, GitHub Actions,
  API testing, REST testing, Postman, SoapUI,
  Python, Java, TypeScript, JavaScript (testing context),
  test cases, test plans, test strategy, bug reports,
  regression testing, smoke testing, exploratory testing

=== BLOCK (Manufacturing / Non-IT QA) ===
Industry context alone is enough to BLOCK:
  Physical manufacturing, factory, heavy industry, construction,
  civil/mechanical engineering, pharma production, food & beverage,
  medical devices (UNLESS explicitly software/CSV/GAMP validation)

German BLOCK words — any single match = BLOCK:
  Produktion, Produktionslinie, Fertigung, Fertigungsanlage,
  Fließband, Schichtarbeit, Schichtdienst,
  Bau, Bauingenieur, Baustelle,
  Wareneingangsprüfung, Endprüfung, Endkontrolle,
  Reklamationsbearbeitung, ISO 13485, ISO 14001, GMP, HACCP,
  Laborant, Lebensmittelkontrolle, Qualitätssicherung in der Produktion

English BLOCK words:
  production line, factory, manufacturing plant, construction site,
  civil engineering QA/QC, pharmaceutical manufacturing QA,
  incoming goods inspection, shift work (manufacturing)

=== INPUT JOBS ===
{jobs_json}

=== OUTPUT RULES (CRITICAL) ===
- Return ONLY a raw JSON integer array containing the "id" values of matching jobs.
- Example: [0, 3, 7]
- If no jobs match, return: []
- Output ONLY the JSON array. No explanation, no markdown fences, no extra text.\
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


def fetch_page(url: str, headers: dict | None = None) -> Optional[BeautifulSoup]:
    """GET a URL and return a parsed BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(
            url,
            headers=headers if headers is not None else HEADERS,
            timeout=REQUEST_TIMEOUT_SEC,
        )
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
        soup = fetch_page(url, headers=DEVJOBS_HEADERS)
        if soup is None:
            time.sleep(random.uniform(2, 5))
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

        # Random delay between keyword requests to avoid 429 rate-limiting
        delay = random.uniform(2, 5)
        logger.debug("devjobs.at — sleeping %.1fs before next request", delay)
        time.sleep(delay)

    logger.info("devjobs.at: %d jobs collected", len(jobs))
    return jobs


# ---------------------------------------------------------------------------
# Gemini AI classification
# ---------------------------------------------------------------------------


def filter_jobs_batch(jobs: list[Job], client: genai.Client) -> set[int]:
    """
    Send a batch of jobs to Gemini in a single call.
    Returns the set of list-indices (0-based) whose jobs are IT/Software QA roles.
    Fails open — returns all indices — on any API or parse error, so real jobs
    are never silently dropped.
    """
    job_data = [
        {
            "id": i,
            "title": job.title,
            "company": job.company,
            "description": job.description[:400],
        }
        for i, job in enumerate(jobs)
    ]
    prompt = GEMINI_BATCH_PROMPT.format(
        jobs_json=json.dumps(job_data, ensure_ascii=False, indent=2)
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0.0,
            ),
        )
        raw = response.text.strip()
        # Accept either a bare array or one wrapped in markdown fences
        match = re.search(r"\[[\d\s,]*\]", raw, re.DOTALL)
        if not match:
            logger.warning("Gemini batch: unexpected response %r — passing all through.", raw[:200])
            return set(range(len(jobs)))
        ids: list[int] = json.loads(match.group())
        valid_ids = {i for i in ids if isinstance(i, int) and 0 <= i < len(jobs)}
        logger.info(
            "Gemini batch (%d jobs) → %d matched: IDs %s",
            len(jobs), len(valid_ids), sorted(valid_ids),
        )
        return valid_ids
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini batch API error: %s — passing all %d jobs through.", exc, len(jobs))
        return set(range(len(jobs)))  # fail open


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
    except requests.HTTPError as exc:
        # e.g. 400 Bad Request (wrong chat_id, bot not started by user, etc.)
        body = exc.response.text if exc.response is not None else "N/A"
        logger.error(
            "Telegram HTTP error for '%s': %s — API response: %s",
            job.title, exc, body,
        )
        return False
    except requests.RequestException as exc:
        logger.error("Telegram request failed for '%s': %s", job.title, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected Telegram error for '%s': %s", job.title, exc)
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
    # devjobs.at disabled — aggressive anti-scraping returns 429.
    # Uncomment the line below to re-enable once a workaround is in place:
    # all_jobs.extend(scrape_devjobs_at())
    logger.info("Total jobs scraped: %d", len(all_jobs))

    # ── Deduplication pass (done BEFORE calling Gemini to save quota) ────────
    updated_hashes: set[str] = set(seen_hashes)
    sent = skipped_dup = skipped_filter = 0
    new_jobs: list[Job] = []

    for job in all_jobs:
        if job.text_fingerprint() in seen_hashes or job.url_fingerprint() in seen_hashes:
            skipped_dup += 1
            continue
        new_jobs.append(job)

    logger.info(
        "New (unseen) jobs to classify: %d | Duplicates skipped: %d",
        len(new_jobs), skipped_dup,
    )

    # ── Gemini batch filtering ────────────────────────────────────────────────
    # Split new_jobs into batches of GEMINI_BATCH_SIZE and send one API call
    # per batch instead of one call per job — dramatically reduces RPM usage.
    approved_jobs: list[Job] = []
    total_batches = -(-len(new_jobs) // GEMINI_BATCH_SIZE) if new_jobs else 0  # ceiling div

    for batch_num, batch_start in enumerate(range(0, len(new_jobs), GEMINI_BATCH_SIZE), 1):
        batch = new_jobs[batch_start : batch_start + GEMINI_BATCH_SIZE]
        logger.info("Gemini: batch %d/%d — classifying %d jobs", batch_num, total_batches, len(batch))

        matched_indices = filter_jobs_batch(batch, gemini_client)

        for i, job in enumerate(batch):
            t_hash = job.text_fingerprint()
            u_hash = job.url_fingerprint()
            # Mark every job seen so we never re-classify it
            updated_hashes.add(t_hash)
            updated_hashes.add(u_hash)
            if i in matched_indices:
                approved_jobs.append(job)
            else:
                skipped_filter += 1
                logger.info("Non-IT QA — filtered: %s", job.title)

        # Sleep between batches (not after the last one)
        if batch_start + GEMINI_BATCH_SIZE < len(new_jobs):
            logger.debug("Sleeping %ss before next Gemini batch...", GEMINI_BATCH_DELAY)
            time.sleep(GEMINI_BATCH_DELAY)

    logger.info(
        "Gemini done — approved: %d | filtered (non-QA): %d",
        len(approved_jobs), skipped_filter,
    )

    # ── Send Telegram alerts ──────────────────────────────────────────────────
    for job in approved_jobs:
        success = send_telegram_alert(job, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        if success:
            logger.info("✓ Sent: %s @ %s", job.title, job.company)
            sent += 1
        else:
            logger.warning("✗ Failed to send: %s", job.title)
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
