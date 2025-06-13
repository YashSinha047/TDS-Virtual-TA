# Install dependencies
!pip install requests beautifulsoup4 tqdm

# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import json
import time

# ========== CONFIG ==========

# Paste your __forum_session cookie here (get it from browser's dev tools)
COOKIES = {
    "__forum_session": "Paste your cookie here",
    "_t": "Paste your cookie here",

}

# Base category URL for TDS Knowledge Base
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34"

# Headers (mimic a real browser)
HEADERS = {
    "User-Agent": "Mozilla/5.0",
}

# Date range (inclusive) 
# Please change the date range here (year, month, day)
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14)

# Output file
# Chnage the file name to whaatever you want
OUTPUT_FILE = "discourse_filtered_posts.jsonl"

# ========== HELPER FUNCTIONS ==========

def get_topic_urls(base_url):
    topic_urls = set()
    page = 0
    while True:
        url = f"{base_url}.json?page={page}"
        print(f"Fetching topic list page {page}...")
        response = requests.get(url, headers=HEADERS, cookies=COOKIES)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}, status: {response.status_code}")
            break
        data = response.json()
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break
        for topic in topics:
            topic_id = topic["id"]
            slug = topic["slug"]
            topic_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}"
            topic_urls.add(topic_url)
        page += 1
        time.sleep(1)  # avoid rate limiting
    return list(topic_urls)

def parse_posts_from_topic(topic_url):
    posts_data = []
    try:
        topic_id = topic_url.split("/")[-1]
        base_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json"
        response = requests.get(base_url, headers=HEADERS, cookies=COOKIES)
        if response.status_code != 200:
            print(f"Failed to fetch topic: {base_url}")
            return []

        data = response.json()
        all_post_ids = data.get("post_stream", {}).get("stream", [])

        # Break into chunks of 20 to avoid overwhelming the API
        chunk_size = 20
        for i in range(0, len(all_post_ids), chunk_size):
            chunk = all_post_ids[i:i + chunk_size]
            post_ids_param = "&".join([f"post_ids[]={pid}" for pid in chunk])
            chunk_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/posts.json?{post_ids_param}"
            chunk_response = requests.get(chunk_url, headers=HEADERS, cookies=COOKIES)
            if chunk_response.status_code != 200:
                print(f"Failed to fetch posts chunk: {chunk_url}")
                continue
            posts = chunk_response.json().get("post_stream", {}).get("posts", [])
            for post in posts:
                created_at = datetime.strptime(post["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                if START_DATE <= created_at <= END_DATE:
                    posts_data.append({
                        "id": post["id"],
                        "topic_id": post["topic_id"],
                        "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{data['slug']}/{post['post_number']}",
                        "username": post.get("username"),
                        "content": BeautifulSoup(post["cooked"], "html.parser").get_text(),
                        "created_at": post["created_at"],
                    })
            time.sleep(0.3)  # be gentle to server

    except Exception as e:
        print(f"Error parsing {topic_url}: {e}")
    return posts_data


# ========== MAIN EXECUTION ==========

all_posts = []
topic_urls = get_topic_urls(BASE_URL)

print(f"\nFound {len(topic_urls)} topics. Scraping posts...\n")

for topic_url in tqdm(topic_urls):
    topic_posts = parse_posts_from_topic(topic_url)
    all_posts.extend(topic_posts)
    time.sleep(0.5)  # avoid hammering the server

# Write to JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for post in all_posts:
        json.dump(post, f)
        f.write("\n")

print(f"\n✅ Scraping complete. {len(all_posts)} posts saved to: {OUTPUT_FILE}")
