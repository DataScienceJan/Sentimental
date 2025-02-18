"""
goemotions_classification.py

Uses a multi-label GoEmotions model to classify Reddit posts
and store them in the 'post_emotions' table, now including post_text.
"""

import psycopg2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Database connection info
DB_HOST = "46.227.152.16"
DB_NAME = "rededit_db"
DB_USER = "postgres"
DB_PASS = "123456"

# 2) Path to your fine-tuned GoEmotions model
MODEL_PATH = r"C:\Users\47936\OneDrive\Desktop\Portfolio Projects\my_goemotions_model"

# 3) The 28 emotion labels from GoEmotions
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

def connect_db():
    """ Connect to PostgreSQL. """
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    conn.autocommit = True
    print("Connected to DB.")
    return conn

def load_model():
    """ Load your fine-tuned GoEmotions model & tokenizer. """
    print(f"Loading model from folder: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # eval mode
    return tokenizer, model

def classify_text(text, tokenizer, model, threshold=0.5):
    """Multi-label classification on text -> returns a list of predicted emotions."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].numpy()  # shape: (28,)
    predicted_indices = np.where(probs >= threshold)[0]
    predicted_labels = [GOEMOTIONS_LABELS[i] for i in predicted_indices]
    if len(predicted_labels) == 0:
        predicted_labels = ["neutral"]
    return predicted_labels

def main():
    conn = connect_db()
    cur = conn.cursor()

    # Ensure our table has a post_text column
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS post_emotions (
        post_id VARCHAR NOT NULL,
        emotion_label VARCHAR NOT NULL,
        post_text TEXT,
        FOREIGN KEY (post_id) REFERENCES reddit_posts(id)
    );
    """
    cur.execute(create_table_sql)

    # If the column doesn't exist, we'll add it. (In case you haven't run it manually.)
    alter_sql = """
    ALTER TABLE post_emotions
    ADD COLUMN IF NOT EXISTS post_text TEXT;
    """
    cur.execute(alter_sql)

    tokenizer, model = load_model()

    # Select all posts that have some cleaned_text
    select_sql = "SELECT id, cleaned_text FROM reddit_posts WHERE cleaned_text IS NOT NULL;"
    cur.execute(select_sql)
    rows = cur.fetchall()
    print(f"Fetched {len(rows)} rows from reddit_posts.")

    for (post_id, text) in rows:
        if not text:
            continue  # if there's no text, skip

        # Multi-label inference
        predicted_emotions = classify_text(text, tokenizer, model, threshold=0.3)

        # Remove old classification rows for that post_id
        delete_sql = "DELETE FROM post_emotions WHERE post_id = %s;"
        cur.execute(delete_sql, (post_id,))

        # Insert each emotion with the text
        insert_sql = """
            INSERT INTO post_emotions (post_id, emotion_label, post_text)
            VALUES (%s, %s, %s);
        """
        for emotion in predicted_emotions:
            cur.execute(insert_sql, (post_id, emotion, text))

    print("Done! Emotions inserted into 'post_emotions' with text included.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
