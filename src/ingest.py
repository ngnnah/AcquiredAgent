import os
import json
import re
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
RAW_DIR = "./data/raw_transcripts"


def is_failed_transcript(transcript_data):
    return (
        transcript_data.get("heading", "") == ""
        or transcript_data.get("transcript", "") == ""
    )


def revert_failed_transcripts(raw_dir, status_file):
    status = load_status(status_file)
    reverted_count = 0

    for filename in os.listdir(raw_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(raw_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)

            if is_failed_transcript(transcript_data):
                url = transcript_data["url"]
                status[url] = False
                os.remove(file_path)
                reverted_count += 1

    save_status(status, status_file)
    print(f"Reverted {reverted_count} failed transcripts.")
    return reverted_count


def load_status(status_file):
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            return json.load(f)
    return {}


def save_status(status, status_file):
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)


def get_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


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


def fetch_transcript(url, session):
    try:
        time.sleep(1)  # Add a 1-second delay between requests
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # A) Section heading information
        section_heading = soup.find("div", class_="section-heading")

        # Extract title
        title = section_heading.find("h1", class_="blog-post-title")
        title = title.text.strip() if title else "N/A"

        # Extract heading (episode info)
        heading_elements = section_heading.find_all("h2", class_="heading-4")
        heading = "N/A"
        for h in heading_elements:
            if h.text.strip():
                heading = h.text.strip()
                break

        # Extract date
        blog_date = section_heading.find("div", class_="blog-date")
        blog_date = blog_date.text.strip() if blog_date else "N/A"

        # B) Description information
        description = ""
        description_div = soup.find("div", class_="w-richtext")
        if description_div:
            for elem in description_div.find_all(["h2", "p"]):
                if "Sponsors:" in elem.text:
                    break
                description += elem.text.strip() + "\n"
        description = description.strip()

        # C) Transcript information
        transcript_div = soup.find("div", id="transcript")
        if transcript_div:
            # Find the next div with class 'rich-text-block-6 w-richtext'
            transcript_content = transcript_div.find_next(
                "div", class_="rich-text-block-6 w-richtext"
            )
            if transcript_content:
                transcript_text = transcript_content.get_text(
                    separator="\n", strip=True
                )
                # Remove the "Transcript:" prefix and any disclaimer if present
                transcript_text = re.sub(
                    r"^Transcript:\s*\(.*?\)\s*", "", transcript_text, flags=re.DOTALL
                )
            else:
                transcript_text = "N/A"
        else:
            transcript_text = "N/A"

        # Ensure the transcript doesn't include the "More Episodes" section
        more_episodes = soup.find("div", class_="main-section gray")
        if more_episodes and transcript_text != "N/A":
            more_episodes_text = more_episodes.get_text()
            transcript_text = transcript_text.split(more_episodes_text)[0].strip()
        episode_data = {
            "url": url,
            "title": title,
            "heading": heading,
            "date": blog_date,
            "description": description,
            "transcript": transcript_text,
        }

        return url, episode_data

    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return url, None


def save_transcript(url, data, raw_dir):
    if data:
        filename = os.path.join(raw_dir, url.split("/")[-1] + ".json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    return False


def ingest_data():
    print("Starting data ingestion...")

    # First, revert any failed transcripts
    reverted_count = revert_failed_transcripts(RAW_DIR, STATUS_FILE)
    print(f"Found and reverted {reverted_count} failed transcripts")

    status = load_status(STATUS_FILE)
    existing_urls = set(status.keys())

    new_transcript_urls = get_transcript_urls(existing_urls)
    print(f"Found {len(new_transcript_urls)} new transcript URLs.")

    # Add new URLs to status
    for url in new_transcript_urls:
        status[url] = False

    # Get all undownloaded URLs
    urls_to_fetch = [url for url, downloaded in status.items() if not downloaded]
    print(f"Total undownloaded transcripts: {len(urls_to_fetch)}")

    os.makedirs(RAW_DIR, exist_ok=True)

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
            url, episode_data = future.result()
            if save_transcript(url, episode_data, RAW_DIR):
                successful_saves += 1
                status[url] = True
            else:
                status[url] = False

            # Save status after each transcript to avoid losing progress
            save_status(status, STATUS_FILE)

    print(
        f"Successfully saved {successful_saves} out of {len(urls_to_fetch)} transcripts in {RAW_DIR}"
    )
    print(
        f"Total transcripts: {len(status)}, Newly added: {len(new_transcript_urls)}, Newly downloaded: {successful_saves}"
    )


if __name__ == "__main__":
    ingest_data()
