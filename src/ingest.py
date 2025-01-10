import os
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

STATUS_FILE = "data/download_transcripts_status.json"


def get_transcript_urls():
    base_url = "https://www.philosophizethis.org/transcripts"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    transcript_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "/transcript/" in href:
            if not href.startswith(("http:", "https:")):
                href = "https://www.philosophizethis.org" + href
            transcript_links.append(href)

    return transcript_links


def get_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_transcript(url, session):
    try:
        time.sleep(1)  # Add a 1-second delay between requests
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        content_div = soup.find("div", class_="sqs-block-content")

        if content_div:
            transcript_text = " ".join(content_div.stripped_strings)
            return url, transcript_text
        else:
            return url, None
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return url, None


def save_transcript(url, text, raw_dir):
    if text:
        filename = os.path.join(raw_dir, url.split("/")[-1] + ".txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    return False


def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)


def ingest_data():
    print("Starting data ingestion...")

    transcript_urls = get_transcript_urls()
    print(f"Found {len(transcript_urls)} transcript URLs.")

    raw_dir = "./data/raw_transcripts"
    os.makedirs(raw_dir, exist_ok=True)

    status = load_status()
    urls_to_fetch = [
        url for url in transcript_urls if url not in status or not status[url]
    ]

    session = get_session()
    successful_saves = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(fetch_transcript, url, session): url
            for url in urls_to_fetch
        }

        for future in tqdm(
            as_completed(future_to_url),
            total=len(urls_to_fetch),
            desc="Fetching and saving transcripts",
        ):
            url = future_to_url[future]
            url, text = future.result()
            if save_transcript(url, text, raw_dir):
                successful_saves += 1
                status[url] = True
            else:
                status[url] = False

            # Save status after each transcript to avoid losing progress
            save_status(status)

    print(
        f"Successfully saved {successful_saves} out of {len(urls_to_fetch)} new transcripts in {raw_dir}"
    )
    print(
        f"Total transcripts: {len(transcript_urls)}, Already downloaded: {len(transcript_urls) - len(urls_to_fetch)}"
    )


if __name__ == "__main__":
    ingest_data()
