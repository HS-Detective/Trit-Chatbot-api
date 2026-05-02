import os
from dotenv import load_dotenv
import os, re, hashlib, time
from datetime import datetime
from email.utils import parsedate_to_datetime  # RFC822 날짜 파서
import requests
import pymysql
import html
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image

#  Jupyter에서 .env를 notebook과 같은 폴더에 둠
load_dotenv(dotenv_path="./.env")

#  환경 변수 불러오기
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")

#  로딩 확인용 출력 (디버깅용)
print("NAVER_CLIENT_ID:", NAVER_ID)
print("MYSQL_DB:", os.getenv("MYSQL_DB"))

#  DB 연결 설정
DB = dict(
    host=os.getenv("MYSQL_HOST"),
    port=os.getenv("MYSQL_PORT"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PW"),
    db=os.getenv("MYSQL_DB"),
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

# --- 설정 ---
QUERY = "화장품 실무 무역"
DISPLAY = 20
TIMEOUT = 8
DO_CROP = False
CROP_SIZE = 200
DEFAULT_THUMB = "/images/news/default.jpg"  # 기본 이미지 경로

#  Jupyter 기준 크롭 이미지 저장 경로
BASE = os.path.abspath(os.path.join(os.getcwd(), "..", "project"))
STATIC_DIR = os.path.join(BASE, "src", "main", "resources", "static", "images", "news")
PUBLIC_PREFIX = "/images/news"

#  HTML 태그 제거
def clean_html(s: str) -> str:
    return re.sub("<.*?>", "", s or "").strip()

#  날짜 변환: 네이버 pubDate → MySQL DATETIME
def to_mysql_datetime(s: str) -> str | None:
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)  # timezone-aware datetime
        return dt.strftime("%Y-%m-%d %H:%M:%S")  # MySQL DATETIME 형식
    except Exception:
        return None

#  네이버 뉴스 검색 API 호출
def naver_news_search(query, display=10, start=1, sort="date"):
    if not NAVER_ID or not NAVER_SECRET:
        raise RuntimeError(".env 설정 누락: NAVER_CLIENT_ID / NAVER_CLIENT_SECRET")
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET,
    }
    params = {"query": query, "display": display, "start": start, "sort": sort}
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("items", [])

#  썸네일 추출 (og:image)
def extract_og_image(link):
    try:
        html = requests.get(link, timeout=TIMEOUT).text
        soup = BeautifulSoup(html, "html.parser")
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return og["content"].strip()
    except Exception:
        pass
    return None

# 이미지 크롭 → static 저장
def crop_square_to_static(img_url):
    try:
        res = requests.get(img_url, timeout=TIMEOUT)
        res.raise_for_status()
        im = Image.open(BytesIO(res.content)).convert("RGB")
        w, h = im.size
        m = min(w, h)
        left = (w - m) // 2
        top = (h - m) // 2
        im = im.crop((left, top, left + m, top + m)).resize((CROP_SIZE, CROP_SIZE))

        os.makedirs(STATIC_DIR, exist_ok=True)
        name = hashlib.md5((img_url + str(time.time())).encode()).hexdigest() + ".jpg"
        save_path = os.path.join(STATIC_DIR, name)
        im.save(save_path, format="JPEG", quality=88)

        return f"{PUBLIC_PREFIX}/{name}"
    except Exception:
        return None

# html문구 제거
def clean_html(s: str) -> str:
    no_tags = re.sub("<.*?>", "", s or "").strip()
    return html.unescape(no_tags)

#  DB 저장
def save_news(rows):
    sql = """
    INSERT INTO trade_news
      (news_title, news_link, news_thumbnail, news_pubdate)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      news_title=VALUES(news_title),
      news_thumbnail=VALUES(news_thumbnail),
      news_pubdate=VALUES(news_pubdate)
    """
    conn = pymysql.connect(**DB)
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    finally:
        conn.close()

#  실행 함수
def run():
    try:
        items = naver_news_search(QUERY, display=DISPLAY)
        rows = []
        for it in items:
            title = clean_html(it.get("title", ""))
            link = (it.get("link") or "").strip()
            pubdate_raw = (it.get("pubDate") or "").strip()
            pubdate = to_mysql_datetime(pubdate_raw)

            if not link:
                continue

            thumb = extract_og_image(link) or DEFAULT_THUMB

            if DO_CROP and thumb != DEFAULT_THUMB:
                processed = crop_square_to_static(thumb)
                if processed:
                    thumb = processed

            rows.append((title, link, thumb, pubdate))

        if rows:
            save_news(rows)
            print(f" {len(rows)}건 저장 완료!")
        else:
            print(" 저장할 뉴스가 없습니다.")
    except Exception as e:
        print(" 에러 발생:", e)

# 실행
run()

