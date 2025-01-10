import os
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

STATUS_FILE = "data/download_transcripts_status.json"


def get_transcript_urls(existing_urls):
    base_url = "https://www.acquired.fm/episodes"
    driver = webdriver.Chrome()  # Make sure you have ChromeDriver installed and in PATH
    driver.get(base_url)

    transcript_links = []
    page = 1
    new_episodes_found = True

    while new_episodes_found:
        new_episodes_found = False
        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "collection-list"))
        )

        # Parse the page content
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find all episode links
        episodes = soup.find_all("div", class_="blog-thumbnail")
        for episode in episodes:
            link = episode.find("a")["href"]
            full_link = f"https://www.acquired.fm{link}"
            if full_link not in existing_urls:
                transcript_links.append(full_link)
                new_episodes_found = True

        print(
            f"Collected {len(transcript_links)} new links from page {page}; Samples: {transcript_links[-3:]}"
        )

        if new_episodes_found:
            # Check if there's a next page
            try:
                older_button = driver.find_element(
                    By.XPATH, "//a[contains(@class, 'w-pagination-next')]"
                )
                older_button.click()
                page += 1
                time.sleep(2)  # Wait for the page to load
            except:
                print("No more pages to load")
                break
        else:
            print("No new episodes found on this page. Stopping search.")
            break

    driver.quit()
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

    status = load_status()
    existing_urls = set(status.keys())

    new_transcript_urls = get_transcript_urls(existing_urls)
    print(f"Found {len(new_transcript_urls)} new transcript URLs.")

    raw_dir = "./data/raw_transcripts"
    os.makedirs(raw_dir, exist_ok=True)

    session = get_session()
    successful_saves = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(fetch_transcript, url, session): url
            for url in new_transcript_urls
        }
        for future in tqdm(
            as_completed(future_to_url),
            total=len(new_transcript_urls),
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
        f"Successfully saved {successful_saves} out of {len(new_transcript_urls)} new transcripts in {raw_dir}"
    )
    print(
        f"Total transcripts: {len(status)}, Newly downloaded: {len(new_transcript_urls)}"
    )


if __name__ == "__main__":
    ingest_data()
