# collect_data.py
import os
import sys
import praw
import psycopg2
import pandas as pd

from config import (
    DB_HOST, DB_NAME, DB_USER, DB_PASS,
    REDDIT_CLIENT_ID, REDDIT_SECRET
)
from logger_setup import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting data collection script...")

    # 1) Connect to DB to fetch the list of subreddits
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()
        # We'll assume the 'subreddits' table has a single column called "name"
        cur.execute("SELECT name FROM subreddits;")
        rows = cur.fetchall()
        sub_list = [row[0] for row in rows]
        logger.info(f"Subreddits to collect: {sub_list}")
        conn.close()
    except Exception as e:
        logger.error(f"Error fetching subreddits: {e}", exc_info=True)
        return

    # 2) Initialize PRAW
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent="MySentimentApp/0.1"
        )
    except Exception as e:
        logger.error(f"Error initializing PRAW: {e}", exc_info=True)
        return

    # 3) Collect data from each subreddit
    limit_per_sub = 60
    data = []

    for sub in sub_list:
        logger.info(f"Collecting hot posts from r/{sub}...")
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.hot(limit=limit_per_sub):
                data.append({
                    "id": post.id,
                    "subreddit": sub,
                    "title": post.title,
                    "selftext": post.selftext or "",  # ensure no None
                    "author": str(post.author),
                    "score": post.score,
                    "created_utc": int(post.created_utc),
                    "num_comments": post.num_comments
                })
        except Exception as e:
            logger.error(f"Error collecting posts from r/{sub}: {e}", exc_info=True)

    # 4) Convert to a DataFrame (optional, for CSV)
    df = pd.DataFrame(data)
    logger.info(f"Total posts collected: {len(df)}")

    # 5) Save to CSV if you want
    csv_filename = "reddit_data.csv"
    try:
        df.to_csv(csv_filename, index=False)
        logger.info(f"Data saved to {csv_filename}")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}", exc_info=True)

    # 6) Insert into PostgreSQL (reddit_posts)
    insert_query = """
        INSERT INTO reddit_posts
            (id, subreddit, title, selftext, author, score, created_utc, num_comments)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING;
    """

    try:
        with psycopg2.connect(
            host=DB_HOST,
            port=5432,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        ) as conn, conn.cursor() as cursor:

            for post_data in data:
                try:
                    cursor.execute(
                        insert_query,
                        (
                            post_data["id"],
                            post_data["subreddit"],
                            post_data["title"],
                            post_data["selftext"],
                            post_data["author"],
                            post_data["score"],
                            post_data["created_utc"],
                            post_data["num_comments"]
                        )
                    )
                except Exception as e:
                    logger.error(f"Error inserting post {post_data['id']}: {e}", exc_info=True)

            conn.commit()
            logger.info("Data inserted into PostgreSQL successfully!")

    except Exception as e:
        logger.error("Error inserting data into PostgreSQL", exc_info=True)

    logger.info("Script completed!")
    logger.info(f"Python executable used: {sys.executable}")


if __name__ == "__main__":
    main()
