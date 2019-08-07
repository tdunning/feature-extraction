import os
import re
import shutil
import time

import urllib3
from bs4 import BeautifulSoup


base_url = 'https://dumps.wikimedia.org/other/pagecounts-raw/2016/2016-07/'

http = urllib3.PoolManager()

r = http.request('GET', base_url)

soup = BeautifulSoup(r.data, 'html.parser')

page_pattern = re.compile('pagecounts-.*gz')

links = [anchor['href'] for anchor in soup.find_all("a") if re.match(page_pattern, anchor['href'])]

try:
    os.mkdir('./wiki-stats')
except FileExistsError:
    print("wiki-stats already exists")

print(os.getcwd())
for link in links:
    url = base_url + '/' + link
    local_file = os.path.join('wiki-stats', link)
    if os.path.exists(local_file):
        print("%s already exists, skipping" % local_file)
    else:
        with http.request('GET', url, preload_content=False) as resp, open(local_file, "wb") as f:
            shutil.copyfileobj(resp, f)
        print(link)
        time.sleep(0.5)
