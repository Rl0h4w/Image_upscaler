import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import re
import time
import pytube
import bs4
import random


def process_url(channel_url):
    
    channel_url = channel_url.rstrip('/videos') + "/videos"  
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(channel_url)

        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(random.uniform(0., 5.))
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = bs4.BeautifulSoup(driver.page_source, "lxml")
        urls = set(filter(lambda x: x.startswith("/watch"), map(lambda x: x["href"], soup.find_all("a", href=True))))

    finally:
        driver.quit()

    return list(map(lambda x: "https://www.youtube.com" + x, urls))

def download():
    if not os.path.exists("data/videos"):
        os.mkdir("data/videos")
        
    with open("data/channels_urls.txt") as file:
        channels_urls = file.readlines()
        
    for channel_url in channels_urls:
        for url in process_url(channel_url):
            name = re.search("v=(.+)", url).group(1)
            yt = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
            stream = yt.streams.filter(adaptive=True, res="720p").first()
            if stream:
                print(f"Downloading {name}")
                stream.download(output_path="data/videos", skip_existing=True)
                print("Done!")
            else:
                print(f"Error: {url}")
        
if __name__=="__main__":
    
    download()