import urllib.request
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from bs4 import BeautifulSoup
import requests


def fetch_press_release_content(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.get_text()
    return content


def topic_modeling(data, n_topics=5):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(data)
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    topic_labels = nmf.transform(tfidf).argmax(axis=1)
    return topic_labels


url = 'https://api.hkma.gov.hk/public/press-releases?lang=en&offset=0'

with urllib.request.urlopen(url) as req:
    data = json.loads(req.read())

records = data['result']['records']
df = pd.DataFrame(records)

# Fetch press release content
content_list = []
for link in df['link']:
    content = fetch_press_release_content(link)
    content_list.append(content)

# Perform topic modeling
n_topics = 5
topic_labels = topic_modeling(content_list, n_topics=n_topics)

# Add topic labels to the DataFrame
df['topic'] = topic_labels

print(df)
