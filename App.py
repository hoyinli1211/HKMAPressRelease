import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import urllib.request
import json
from bs4 import BeautifulSoup


def fetch_press_releases():
    url = 'https://api.hkma.gov.hk/public/press-releases?lang=en&offset=0'
    with urllib.request.urlopen(url) as req:
        data = json.loads(req.read())["result"]["records"]
    return data


def fetch_press_release_content(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.get_text()
    return content


def topic_modeling(data, n_topics=5, n_top_words=10):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(data)
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics


def main():
    st.title("HKMA Press Releases Topic Categorization")

    st.write("Fetching press releases data from HKMA OpenAPI...")
    press_releases_data = fetch_press_releases()

    data = []
    titles = []
    for item in press_releases_data:
        titles.append(item["title_en"])
        content_url = item["link_en"]
        content = fetch_press_release_content(content_url)
        data.append(content)

    if data:
        st.write(f"Number of press releases: {len(data)}")
        n_topics = st.slider("Select the number of topics:", 1, 10, 5)
        topics = topic_modeling(data, n_topics=n_topics)

        st.write("Topics:")
        for topic in topics:
            st.write(topic)

        selected_topic = st.selectbox("Select a topic to view related press releases:", list(range(1, n_topics + 1)))
        topic_idx = selected_topic - 1
        topic_words = set(topics[topic_idx].replace("Topic", "").split(": ")[1].split(", "))

        st.write(f"Press releases in Topic {selected_topic}:")
        for idx, text in enumerate(data):
            if len(topic_words.intersection(set(text.lower().split()))) > 0:
                st.write(f"- {titles[idx]}")
    else:
        st.error("No data available.")


if __name__ == "__main__":
    main()
