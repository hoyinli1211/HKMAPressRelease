import urllib.request
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import requests


def fetch_press_releases():
    url = 'https://api.hkma.gov.hk/public/press-releases?lang=en&offset=0'
    with urllib.request.urlopen(url) as req:
        data = json.loads(req.read())["result"]["records"]
    return data


def topic_modeling(data, n_topics=5):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(data)
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    topic_labels = nmf.transform(tfidf).argmax(axis=1)

    # Get the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    top_words = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_n_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        top_words.append(", ".join(top_n_words))

    return topic_labels, top_words


def main():
    press_releases_data = fetch_press_releases()

    # Sort by date and take the top 10
    press_releases_data = sorted(press_releases_data, key=lambda x: x['date'], reverse=True)[:10]

    data = []
    titles = []
    for item in press_releases_data:
        titles.append(item["title"])
        content_url = item["link"]
        response = requests.get(content_url)
        content = response.text if response.status_code == 200 else ''
        data.append(content)

    if data:
        n_topics = 5
        topic_labels, top_words = topic_modeling(data, n_topics=n_topics)

        # Add topic labels and names to the DataFrame
        df = pd.DataFrame(press_releases_data)
        df['topic'] = topic_labels
        df['topic_name'] = [top_words[label] for label in topic_labels]

        df
    else:
        print("No data available.")


if __name__ == "__main__":
    main()
