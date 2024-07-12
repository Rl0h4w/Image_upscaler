import os
import re
import time
import random
import asyncio
import aiohttp
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from pytube import YouTube
from bs4 import BeautifulSoup

MAX_SIZE_MB = 50_000

def get_total_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def process_url(channel_url):
    channel_url = channel_url.rstrip('/videos') + "/videos"
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-notifications')
    options.add_argument('--disable-popup-blocking')
    service = Service(ChromeDriverManager().install())
    with webdriver.Chrome(service=service, options=options) as driver:
        driver.get(channel_url)
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(random.uniform(0.1, 3))  
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "lxml")
        urls = set(filter(lambda x: x.startswith("/watch"), map(lambda x: x["href"], soup.find_all("a", href=True))))

    return list(map(lambda x: "https://www.youtube.com" + x, urls))

async def download_video(session, url, output_path):
    global current_size
    try:
        name = re.search("v=(.+)", url).group(1)
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(adaptive=True, res="144p").first()
        if stream:
            video_size = stream.filesize
            if (current_size + video_size) <= MAX_SIZE_MB * 1024 * 1024:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, stream.download, output_path)
                current_size += video_size
            else:
                print(f"Skipping {url} due to size limit.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

async def process_channel(session, channel_url, output_path):
    video_urls = process_url(channel_url.strip())
    tasks = [download_video(session, url, output_path) for url in video_urls]
    await asyncio.gather(*tasks)

async def download():
    global current_size
    if not os.path.exists("data/videos"):
        os.makedirs("data/videos")

    current_size = get_total_size("data/videos")
    async with aiohttp.ClientSession() as session:
        with open("data/channels_urls.txt") as file:
            channels_urls = file.readlines()
        
        tasks = [process_channel(session, channel_url, "data/videos") for channel_url in channels_urls]
        await asyncio.gather(*tasks)