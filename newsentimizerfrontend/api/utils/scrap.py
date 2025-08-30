"""
social_scraper.py – version 2025-04-27-c

Modular multi‑platform extractor with rotating proxies, stealth Playwright,
SQLite caching, step‑wise logging, and **full comment threads** for both
Instagram *and* X / Twitter.

Changes in this build
─────────────────────
✓ Added `TwitterScraper.comments(tweet_id, limit=100)` – deep‑scroll reply
  harvesting with progress markers.
✓ Expanded CLI demo to prove comment‑thread extraction on both platforms.

────────────────────────────────────────────────────────────────────────────
DISCLAIMER – Use responsibly. You are accountable for legal / ToS risks.
────────────────────────────────────────────────────────────────────────────
"""

import asyncio, os, random, sqlite3, time, json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from bs4 import BeautifulSoup

# ╭──────────────────────── Configuration ─────────────────────────╮
DEFAULT_PROXIES: List[str] = []
PROXY_FILE = Path(__file__).with_name("proxies.txt")
if PROXY_FILE.exists():
    DEFAULT_PROXIES = [p.strip() for p in PROXY_FILE.read_text().splitlines() if p.strip()]

USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/20D67 Safari",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

CACHE_DB = Path(__file__).with_name("scraper_cache.sqlite3")

# ╭────────────────────────── Logging ─────────────────────────────╮
_ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_ok = lambda m: print(f"[{_ts()}] ✅ {m}", flush=True)
log_err = lambda m: print(f"[{_ts()}] ❌ {m}", flush=True)

# ╭────────────────────────── Cache ───────────────────────────────╮
class Cache:
    def __init__(self, db_path: Path = CACHE_DB):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB, exp INTEGER)")
        self.conn.commit()

    def set(self, k: str, v: bytes, ttl: int):
        self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (k, v, int(time.time()) + ttl))
        self.conn.commit()

    def get(self, k: str):
        row = self.conn.execute("SELECT v, exp FROM kv WHERE k=?", (k,)).fetchone()
        if row and row[1] > int(time.time()):
            return row[0]
        return None

# ╭───────────────────────── Proxy rotator ────────────────────────╮
class ProxyRotator:
    def __init__(self, proxies: List[str] | None = None):
        self.proxies = proxies or DEFAULT_PROXIES
        self.i = 0
    def next(self):
        if not self.proxies: return None
        p = self.proxies[self.i % len(self.proxies)]; self.i += 1; return p

# ╭───────────────────────── Base scraper ─────────────────────────╮
class BaseScraper:
    def __init__(self, rot: ProxyRotator, cache: Cache, headless: bool = True):
        self.rot, self.cache, self.headless = rot, cache, headless

    @asynccontextmanager
    async def ctx(self):
        proxy = self.rot.next()
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=self.headless, proxy={"server": proxy} if proxy else None, args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS), viewport={"width":1280,"height":720}, locale="en-US")
        await context.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        try:
            yield context
        finally:
            await context.close(); await browser.close(); await pw.stop()

    async def _html(self, url:str, ttl:int=3600):
        ck=f"h::{url}"; cached=self.cache.get(ck)
        if cached:
            log_ok(f"CACHE → {url}"); return cached.decode()
        log_ok(f"FETCH → {url}")
        async with self.ctx() as c:
            p=await c.new_page(); await p.goto(url, timeout=60000); await p.wait_for_load_state("networkidle")
            h=await p.content(); self.cache.set(ck,h.encode(),ttl); return h

# ╭──────────────────── Instagram scraper ─────────────────────────╮
class InstagramScraper(BaseScraper):
    BASE="https://www.instagram.com"
    async def post_info(self, sc:str):
        log_ok(f"IG post_info {sc}")
        try:
            soup=BeautifulSoup(await self._html(f"{self.BASE}/p/{sc}/",1800),"html.parser")
            j=json.loads(soup.find("script",{"type":"application/ld+json"}).string)
            return {"id":j.get("identifier"),"caption":j.get("caption"),"image":j.get("image"),"uploadDate":j.get("uploadDate"),"author":j.get("author",{}).get("alternateName"),"commentCount":j.get("commentCount"),"likeCount":j.get("interactionStatistic",[{}])[0].get("userInteractionCount"),"url":f"{self.BASE}/p/{sc}/"}
        except Exception as e:
            log_err(f"IG post_info error {e}"); raise
    async def comments(self, sc:str, limit:int=100):
        log_ok(f"IG comments {sc} limit {limit}"); res=[]
        async with self.ctx() as c:
            p=await c.new_page(); await p.goto(f"{self.BASE}/p/{sc}/"); await p.wait_for_selector("ul > li")
            while len(res)<limit:
                lis=await p.query_selector_all("ul > li")
                for li in lis[len(res):]:
                    text=await li.inner_text(); author=await li.query_selector_eval("h3","e=>e.innerText") if await li.query_selector("h3") else None
                    res.append({"author":author,"text":text});
                    if len(res)>=limit:break
                await p.mouse.wheel(0,1000); await p.wait_for_timeout(300)
        return res
    async def search(self, q:str, limit:int=50):
        tag=q.lstrip("#"); log_ok(f"IG search #{tag} limit {limit}"); seen=set(); out=[]
        async with self.ctx() as c:
            p=await c.new_page(); await p.goto(f"{self.BASE}/explore/tags/{tag}/"); await p.wait_for_selector("article a[href^='/p/']")
            while len(out)<limit:
                anchors=await p.query_selector_all("article a[href^='/p/']")
                for a in anchors:
                    href=await a.get_attribute("href"); sc=href.split("/p/")[1].strip("/") if href else None
                    if sc and sc not in seen:
                        out.append({"shortcode":sc,"url":f"{self.BASE}{href}"}); seen.add(sc); log_ok(f"  ↳ {sc}")
                        if len(out)>=limit:break
                await p.mouse.wheel(0,1400); await p.wait_for_timeout(400)
        return out

# ╭────────────────────── Twitter scraper ─────────────────────────╮
class TwitterScraper(BaseScraper):
    BASE="https://x.com"
    async def tweet_info(self, tid:str):
        log_ok(f"TW tweet_info {tid}")
        try:
            soup=BeautifulSoup(await self._html(f"{self.BASE}/i/status/{tid}",1800),"html.parser")
            text=soup.find("meta",{"property":"og:description"})["content"]
            author=soup.find("meta",{"property":"og:title"})["content"].split(" on X")[0]
            js=soup.find_all("script")[-1].string or ""
            def val(k):
                key=f'"{k}":'
                if key in js:
                    try:return int(js[js.index(key)+len(key):].split(',',1)[0])
                    except:pass; return None
            return {"id":tid,"author":author,"text":text,"likes":val("favorite_count"),"retweets":val("retweet_count"),"bookmarks":val("bookmark_count"),"url":f"{self.BASE}/i/status/{tid}"}
        except Exception as e:
            log_err(f"TW tweet_info error {e}"); raise
    async def comments(self, tid:str, limit:int=100):
        log_ok(f"TW comments {tid} limit {limit}"); out=[]; seen=set()
        async with self.ctx() as c:
            p=await c.new_page(); await p.goto(f"{self.BASE}/i/status/{tid}"); await p.wait_for_selector("article[data-testid='tweet']")
            first=True
            while len(out)<limit:
                cards=await p.query_selector_all("article[data-testid='tweet']")
                for card in cards:
                    cid=await card.get_attribute("data-tweet-id")
                    if first: first=False; continue  # original tweet, skip
                    if cid and cid not in seen:
                        txt=await card.inner_text()
                        user=await card.query_selector_eval("div[dir='ltr'] span","e=>e.innerText") if await card.query_selector("div[dir='ltr'] span") else None
                        out.append({"id":cid,"author":user,"text":txt}); seen.add(cid); log_ok(f"  ↳ reply {cid}")
                        if len(out)>=limit:break
                await p.mouse.wheel(0,1400); await p.wait_for_timeout(500)
        return out
    async def search(self, q:str, limit:int=50):
        log_ok(f"TW search '{q}' limit {limit}"); out=[]; seen=set()
        async with self.ctx() as c:
            p=await c.new_page(); await p.goto(f"{self.BASE}/search?q={q}&src=typed_query&f=live"); await p.wait_for_selector("article[data-testid='tweet']")
            while len(out)<limit:
                cards=await p.query_selector_all("article[data-testid='tweet']")
                for card in cards:
                    tid=await card.get_attribute("data-tweet-id")
                    if tid and tid not in seen:
                        txt=await card.inner_text(); out.append({"id":tid,"text":txt}); seen.add(tid); log_ok(f"  ↳ {tid}")
                        if len(out)>=limit:break
                await p.mouse.wheel(0,1400); await p.wait_for_timeout(500)
        return out

# ╭────────────────────────── CLI demo ────────────────────────────╮
async def _demo():
    rot=ProxyRotator(); cache=Cache(); ig=InstagramScraper(rot,cache); tw=TwitterScraper(rot,cache)
    log_ok("--- Instagram demo ---")
    posts=await ig.search("sunset",limit=5)
    info=await ig.post_info(posts[0]['shortcode']); log_ok(f"Post → {info}")
    cmts=await ig.comments(posts[0]['shortcode'],limit=5); log_ok(f"Comments → {len(cmts)}")
    log_ok("--- Twitter demo ---")
    tweets=await tw.search("openai",limit=5)
    twinfo=await tw.tweet_info(tweets[0]['id']); log_ok(f"Tweet → {twinfo}")
    replies=await tw.comments(tweets[0]['id'],limit=5); log_ok(f"Replies → {len(replies)}")

if __name__=="__main__": asyncio.run(_demo())
