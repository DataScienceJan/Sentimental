import streamlit as st
import psycopg2
import pandas as pd
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

def get_connection():
    """Simple helper to get a psycopg2 DB connection."""
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def main():
    st.header("Tracked Subreddits")

    # 1) Fetch current subreddits
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM subreddits;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    sub_list = [r[0] for r in rows]

    st.write("Currently tracking:")

    # 2) Display each subreddit with a "Remove" button next to it
    for sub in sub_list:
        col1, col2 = st.columns([3, 1])  # e.g. 3:1 width ratio
        col1.write(sub)
        if col2.button("Remove", key=sub):
            # Attempt to delete this subreddit from the table
            try:
                conn = get_connection()
                with conn.cursor() as c:
                    c.execute("DELETE FROM subreddits WHERE name = %s;", (sub,))
                conn.commit()
                conn.close()
                st.success(f"Removed subreddit: {sub}")
                st.experimental_rerun()  # Refresh the page to update the list
            except Exception as e:
                st.error(f"Error removing subreddit '{sub}': {e}")

    # 3) Let the user add a new subreddit
    st.write("---")
    new_sub = st.text_input("Add a new subreddit (no 'r/'):")
    if st.button("Add Subreddit"):
        if new_sub.strip():
            try:
                conn = get_connection()
                with conn.cursor() as c:
                    c.execute(
                        "INSERT INTO subreddits (name) VALUES (%s) ON CONFLICT DO NOTHING;",
                        (new_sub.strip(),)
                    )
                conn.commit()
                conn.close()
                st.success(f"Added subreddit: {new_sub.strip()}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error adding subreddit: {e}")
        else:
            st.warning("Please enter a valid subreddit name.")

    # 4) Show the latest posts from `post_emotions`
    st.write("---")
    st.header("Labels based on the SQL Table")
    st.write("Here you will see labels based on the recent posts that are saved in post_emotions. These are cleaned and given a 'score' and is from every subreddit that we track`:")

    conn = get_connection()
    df = pd.read_sql("""
        SELECT post_id, emotion_label, post_text
        FROM post_emotions
        ORDER BY post_id DESC
        LIMIT 50
    """, conn)
    conn.close()

    st.dataframe(df)

if __name__ == "__main__":
    main()


# .\venv\Scripts\activate.ps1  # Windows PowerShell example
# streamlit run app_goemotions.py