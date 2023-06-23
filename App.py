import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import urllib.request
import json


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
    return topic_labels


def main():
    st.title("HKMA Press Releases Topic Categorization")

    st.write("Fetching press releases data from HKMA OpenAPI...")
    press_releases_data = fetch_press_releases()

    # Sort by date and take the top 10
    press_releases_data = sorted(press_releases_data, key=lambda x: x['date'], reverse=True)[:10]
    st.dataframe(press_releases_data)
    
    data = []
    titles = []
    for item in press_releases_data:
        titles.append(item["title"])
        content_url = item["link"]
        response = requests.get(content_url)
        content = response.text if response.status_code == 200 else ''
        data.append(content)

    if data:
        st.write(f"Number of press releases: {len(data)}")
        n_topics = st.slider("Select the number of topics:", 1, 10, 5)
        topic_labels = topic_modeling(data, n_topics=n_topics)

        st.write("Topic labels assigned to press releases:")
        for idx, topic_label in enumerate(topic_labels):
            st.write(f"Press release {idx + 1}: Topic {topic_label + 1}")
    else:
        st.error("No data available.")


if __name__ == "__main__":
    main()
