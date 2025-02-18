# app.py
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# Just import the already-loaded environment variables from config
from config import (
    DB_HOST, DB_NAME, DB_USER, DB_PASS,
    REDDIT_CLIENT_ID, REDDIT_SECRET
)

def get_data():
    # Create the SQLAlchemy engine
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    )
    query = """
        SELECT subreddit, sentiment_label, sentiment_compound, created_utc
        FROM reddit_posts
    """
    df = pd.read_sql(query, engine)
    return df

def main():
    st.title("Reddit Sentiment Dashboard")

    # Refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()

    df = get_data()

    # Overall counts
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = df['sentiment_label'].value_counts()
    st.bar_chart(sentiment_counts)

    # Average sentiment by subreddit
    st.subheader("Average Sentiment by Subreddit")
    avg_sentiment = df.groupby('subreddit')['sentiment_compound'].mean().sort_values(ascending=False)
    st.write(avg_sentiment)

    # Show a few recent posts
    st.subheader("Recent Posts (sorted by latest)")
    df_sorted = df.sort_values(by='created_utc', ascending=False).head(10)
    st.write(df_sorted[['subreddit', 'sentiment_label', 'sentiment_compound', 'created_utc']])

if __name__ == "__main__":
    main()
