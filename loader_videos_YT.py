import os
import re
import time
import pytube
import bs4
import random
import threading
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def process_url(channel_url):
    channel_url = channel_url.rstrip('/videos') + "/videos"
    options = Options()
    options.add_argument('--headless')
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(channel_url)
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(random.uniform(0.5, 1.5))
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = bs4.BeautifulSoup(driver.page_source, "lxml")
        urls = set(filter(lambda x: x.startswith("/watch"), map(lambda x: x["href"], soup.find_all("a", href=True))))
    finally:
        driver.quit()

    return ["https://www.youtube.com" + x for x in urls]

def download_video(url, output_path):
    try:
        name = re.search("v=(.+)", url).group(1)
        yt = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(adaptive=True, res="144p").first()
        if stream:
            print(f"Downloading {name}")
            stream.download(output_path=output_path, skip_existing=True)
            print("Done!")
        else:
            print(f"Error: {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download():
    if not os.path.exists("data/videos"):
        os.makedirs("data/videos")
    with open("data/channels_urls.txt") as file:
        channels_urls = file.readlines()
        
    for channel_url in channels_urls:
        video_urls = process_url(channel_url.strip())
        threads = []
        for url in video_urls:
            t = threading.Thread(target=download_video, args=(url, "data/videos"))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
            
if __name__=="__main__":
    download()