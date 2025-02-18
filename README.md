Reddit Multi‐Label Emotion Analysis Project
By Jan F. Dabrowski
1. Project Motivation
Hello! I built this project to explore nuanced emotional signals in Reddit posts using Google’s GoEmotions dataset/model. Traditional sentiment analysis (positive/negative/neutral) can be oversimplified, missing real emotional depth. This pipeline collects real‐time Reddit data, cleans and preprocesses it, then assigns one or more of 28 possible emotions (including “neutral”) to each post.
I aimed to show my data science engineering skills:
Automating data ingestion from an external API (Reddit).
Built a ETL (Extract‐Transform‐Load) flow with PostgreSQL.
Handling multi‐label classification with Transformers from Hugging Face.
Creating a Streamlit dashboard so I can easily manage which subreddits are tracked and view recent labeled posts.

2. Data & Tools
Data Source: Reddit’s “hot” posts, retrieved with PRAW (the official Python Reddit API wrapper).
Database: PostgreSQL for storage (reddit_posts for raw data, post_emotions for multi‐label results).
NLP Model: A multi‐label classification model from the GoEmotions dataset (27 emotions + 1 neutral).
Scheduling: Python’s schedule library automatically re‐runs the pipeline every few minutes, so it’s near real‐time (as long as the script is activated)

3. Architecture & Code Flow
collect_data.py
Reads a list of subreddits from the subreddits table.
Uses PRAW to fetch posts from each subreddit, storing them in the reddit_posts table (title, selftext, author, timestamps, etc.).
clean_data.py
Merges title + selftext into a single string.
Removes URLs, punctuation, and stopwords, saving the result in cleaned_text.
goemotions_classification.py
Loads the GoEmotions multi‐label model from a local folder (fine‐tuned DistilBERT).
For each row with a non‐null cleaned_text, classifies the text. If the probability for an emotion is >= 0.3 or 0.5 (a customizable threshold), that label is assigned. If no label passes, “neutral” is assigned.
Inserts one row per emotion into the post_emotions table.
scheduler.py
Automates the entire pipeline on a schedule (e.g. every 2 minutes).
It runs collect_data.py, then clean_data.py, then goemotions_classification.py in sequence, logging any errors.
app_goemotions.py (Streamlit)
Shows which subreddits are currently tracked (subreddits table).
Allows adding or removing subreddits via a simple UI.
Displays the most recent 50 labeled posts in post_emotions, including the text and assigned emotion labels.

4. Database Schema
subreddits
name (VARCHAR PRIMARY KEY)
This lets me dynamically manage which subreddits are collected, without changing code.
reddit_posts
id (VARCHAR PRIMARY KEY)
subreddit, title, selftext, author, score, created_utc, num_comments
cleaned_text (TEXT) for processed text.
post_emotions
post_id (VARCHAR, references reddit_posts.id)
emotion_label (VARCHAR)
post_text (TEXT, optional convenience copy of the cleaned text)
Each Reddit post can have multiple emotions, which is why we have a separate post_emotions table (a one‐to‐many or many‐to‐many approach).

5. Quick Setup & Usage
Clone or download the repo.
Create a Python virtual environment and install the dependencies (praw, psycopg2, streamlit, transformers, etc.).
Ensure you have a PostgreSQL DB set up with the subreddits, reddit_posts, and post_emotions tables created.
Insert a few subreddits into subreddits.
Start the pipeline scheduler:
python scheduler.py
This runs every 2 minutes by default, collecting data, cleaning text, and classifying emotions.
Launch the Streamlit dashboard:
streamlit run app_goemotions.py (write in terminal)


Visit the link in your browser (usually http://localhost:8501).
Under “Manage Tracked Subreddits,” you can see the current subreddits. Press “Remove” next to any you no longer want.
Add new ones as needed (e.g., “relationships,” “teenagers,” etc.).
Scroll down to see the most recent posts and their assigned emotions.

6. Challenges & Insights
Multi‐label thresholds: Tuning the threshold can drastically change how many emotions get assigned, and how often “neutral” is assigned.
Automating at scale: The scheduler ensures everything updates continuously without manual runs, which is essential for production data pipelines.
Flexible UI: The Streamlit UI fosters an easy, self‐service approach to adding or removing subreddits. No code edits required.
