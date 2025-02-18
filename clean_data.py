# clean_data.py
import re
import psycopg2
from dotenv import load_dotenv
from nltk.corpus import stopwords
from config import (
    DB_HOST, DB_NAME, DB_USER, DB_PASS
)
from logger_setup import get_logger

logger = get_logger(__name__)


def clean_text(text):
    """
    Lowercases, removes URLs and punctuation, strips stopwords.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def main():
    load_dotenv()

    # Connect to DB
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cursor = conn.cursor()
        logger.info("Connected to database.")
    except Exception as e:
        logger.error(f"Error connecting to DB: {e}", exc_info=True)
        return

    # Step 1) Fetch posts with NULL cleaned_text
    # We'll retrieve both 'title' and 'selftext'
    select_query = """
         SELECT id, title, selftext
         FROM reddit_posts
         WHERE cleaned_text IS NULL
     """
    try:
        cursor.execute(select_query)
        rows = cursor.fetchall()
        logger.info(f"Fetched {len(rows)} rows to clean.")
    except Exception as e:
        logger.error(f"Error fetching rows: {e}", exc_info=True)
        conn.close()
        return

    update_query = """
         UPDATE reddit_posts
         SET cleaned_text = %s
         WHERE id = %s
     """

    # Step 2) For each row, combine title + selftext
    for post_id, title, body in rows:
        if not title:
            title = ""
        if not body:
            body = ""
        # Combine them, e.g. "title + body"
        combined_text = title + " " + body

        cleaned = clean_text(combined_text)
        try:
            cursor.execute(update_query, (cleaned, post_id))
        except Exception as e:
            logger.error(f"Error updating post {post_id}: {e}", exc_info=True)

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Data cleaning complete!")


if __name__ == "__main__":
    main()
