#analysis.py
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from config import (
    DB_HOST, DB_NAME, DB_USER, DB_PASS,
    REDDIT_CLIENT_ID, REDDIT_SECRET
)
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', etc.
import matplotlib.pyplot as plt

conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

# Pull everything into a DataFrame
query = "SELECT subreddit, sentiment_label, sentiment_compound FROM reddit_posts;"
df = pd.read_sql(query, conn)
conn.close()

# Overall distribution
print(df['sentiment_label'].value_counts())

# Average compound score by subreddit
avg_by_sub = df.groupby('subreddit')['sentiment_compound'].mean().sort_values(ascending=False)
print("\nAverage sentiment compound by subreddit:")
print(avg_by_sub.head(10))



# Bar chart of sentiment label counts
df['sentiment_label'].value_counts().plot(kind='bar', color=['green','red','gray'])
plt.title("Sentiment Label Counts")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Horizontal bar for average sentiment by subreddit
avg_by_sub.head(10).plot(kind='barh', color='blue')
plt.title("Top 10 Subreddits by Avg Sentiment")
plt.xlabel("Avg Compound Score")
plt.ylabel("Subreddit")
plt.show()
