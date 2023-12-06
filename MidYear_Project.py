from difflib import SequenceMatcher
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import difflib
import nltk
from nltk.tokenize import word_tokenize

# Load DataFrames
file_path = 'himym_data.csv'
df = pd.read_csv(file_path)

transcription_path = 'himym_transcripts.csv'
tf = pd.read_csv(transcription_path)

# Dark theme and fixed top bar styling
st.markdown(
    """
    <style>
    body {
        color: white;
        background-color: #2e2e2e;
        padding-top: 50px; /* Adjust the padding-top to fit your content */
    }
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #333;
        padding: 10px;
        text-align: center;
        font-size: 18px;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a fixed top bar
st.markdown('<div class="top-bar">HIMYM Analysis | DataFrame Analysis | Scrapped Transcription | HIMYM 3.5 AI Bot</div>', unsafe_allow_html=True)

# Define content for each section
section = st.selectbox("Select Section", ["HIMYM Analysis", "DataFrame Analysis", "Scrapped Transcription", "HIMYM 3.5 AI Bot"])

if section == "HIMYM Analysis":
    st.write("This project provides an analysis of How I Met Your Mother TV show data.")

    # Display the Cover Image
    st.image('cover.jpg', use_column_width=True)

    # Display the DataFrame
    st.subheader("Original DataFrame")
    st.dataframe(df.head(5))

    # Modified Data Information
    st.subheader("Modified Data Information")
    st.text("Summary information about the modified dataset:")

    # Number of rows and columns
    st.table(pd.DataFrame({
        'Column': df.columns,
        'Data Type': [df[col].dtype for col in df.columns],
        'Missing Values': df.isnull().sum()
    }))

    # Data Statistics
    st.subheader("Data Statistics")
    st.text("Descriptive statistics of the dataset:")
    st.dataframe(df.describe())

    # Data Types Conversion & Parsing Dates and Handling Outliers
    st.subheader("Data Types Conversion & Parsing Dates / Handling Outliers")

    # First row with two columns
    col1, col2 = st.columns(2)

    # Data Types Conversion & Parsing Dates
    with col1:
        st.text("Converting data types and parsing dates for analysis.")
        df['original_air_date'] = pd.to_datetime(df['original_air_date'], format='%m/%d/%Y')
        df['year'] = df['original_air_date'].dt.year
        df['month'] = df['original_air_date'].dt.month
        st.dataframe(df[['original_air_date', 'year', 'month']].head())

    # Handling Outliers
    with col2:
        st.text("Identifying and removing outliers in 'imdb_rating'.")
        Q1 = df['imdb_rating'].quantile(0.25)
        Q3 = df['imdb_rating'].quantile(0.75)
        IQR = Q3 - Q1
        df_outliers = df.loc[(df['imdb_rating'] >= Q1 - 1.5 * IQR) & (df['imdb_rating'] <= Q3 + 1.5 * IQR)]
        st.dataframe(df_outliers[['imdb_rating']].head())

    # Feature Scaling and New Features Extraction
    st.subheader("Feature Scaling / New Features Extraction")

    # Second row with two columns
    col3, col4 = st.columns(2)

    # Feature Scaling
    with col3:
        st.text("Applying Min-Max scaling for 'us_viewers'.")
        df['us_viewers_scaled'] = ((df['us_viewers'] - df['us_viewers'].min()) / (df['us_viewers'].max() - df['us_viewers'].min())).round(2)
        st.dataframe(df[['us_viewers', 'us_viewers_scaled']].head())

    # New Features Extraction
    with col4:
        st.text("Extracting the 'holding_up_perc' feature.")
        df.loc[:, 'holding_up_perc'] = ((df['votes'] / df['us_viewers'])*1000).round(2)
        st.dataframe(df[['votes', 'us_viewers', 'holding_up_perc']].head())

    # Drop Unnecessary Columns
    st.subheader("Drop Unnecessary Columns")
    st.text("Dropping unnecessary columns.")
    columns_to_drop = ['episode_num_overall', 'votes', 'original_air_date']
    df.drop(columns=columns_to_drop, inplace=True)
    st.dataframe(df.head())

    # Box Plots
    st.subheader("Box Plots")
    st.text("Visualizing numerical columns with box plots.")
    numerical_columns = ['imdb_rating', 'holding_up_perc', 'us_viewers', 'us_viewers_scaled']
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i])
    st.pyplot(fig)

    # Distribution Plots
    st.subheader("Distribution Plots")
    st.text("Visualizing the distribution of numerical columns.")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    sns.distplot(df['us_viewers_scaled'], kde=False, bins=25, ax=axes[0])
    axes[0].set_title('Distribution Plot for us_viewers_scaled')
    sns.distplot(df['imdb_rating'], kde=False, bins=25, ax=axes[1])
    axes[1].set_title('Distribution Plot for imdb_rating')
    sns.distplot(df['holding_up_perc'], kde=False, bins=25, ax=axes[2])
    axes[2].set_title('Distribution Plot for holding_up_perc')
    if 'us_viewers' in df.columns:
        sns.distplot(df['us_viewers'], kde=False, bins=25, ax=axes[3])
        axes[3].set_title('Distribution Plot for us_viewers')
    st.pyplot(fig)

###########
    
    # Logarithmic Transformation
    st.subheader("Logarithmic Transformation")
    st.text("Applying logarithmic transformation to 'holding_up_perc'.")
    
    # Print basic information for debugging
    st.text(f"Before transformation - Number of rows: {len(df)}, Column exists: {'holding_up_perc' in df.columns}")
    
    # Display the first few rows of the DataFrame
    st.text("First few rows of the DataFrame:")
    st.write(df.head())
    
    # Create a copy of the DataFrame for transformation
    df_transformed = df.copy()
    
    # Check for NaN or Inf values before transformation
    st.text("Checking for NaN or Inf values before transformation:")
    st.write(df_transformed[df_transformed['holding_up_perc'].isna() | ~np.isfinite(df_transformed['holding_up_perc'])])
    
    # Original Distribution Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(x='holding_up_perc', data=df_transformed, kde=True, ax=axes[0])
    axes[0].set_title('Original Distribution Plot for holding_up_perc')
    
    # Logarithmic Transformation using np.log1p
    df_transformed['holding_up_perc'] = np.log1p(df_transformed['holding_up_perc'])
    
    # Check for NaN or Inf values after transformation
    st.text("Checking for NaN or Inf values after transformation:")
    st.write(df_transformed[df_transformed['holding_up_perc'].isna() | ~np.isfinite(df_transformed['holding_up_perc'])])
    
    # New Distribution Plot after Logarithmic Transformation
    sns.histplot(x='holding_up_perc', data=df_transformed, kde=True, ax=axes[1])
    axes[1].set_title('Logarithmic Transformation for holding_up_perc')
    
    # Show the plot
    st.pyplot(fig)

###########
    
    # KDE Plots
    st.subheader("KDE Plots")
    st.text("Visualizing KDE plots for selected columns.")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))

    kde_us_viewers = sns.kdeplot(df['us_viewers_scaled'], fill=False, ax=axes[0])
    sns.lineplot(x=kde_us_viewers.get_lines()[0].get_xdata(), y=kde_us_viewers.get_lines()[0].get_ydata(), ax=axes[0])
    axes[0].set_title('Line Plot for us_viewers_scaled')

    kde_imdb_rating = sns.kdeplot(df['imdb_rating'], fill=False, ax=axes[1])
    sns.lineplot(x=kde_imdb_rating.get_lines()[0].get_xdata(), y=kde_imdb_rating.get_lines()[0].get_ydata(), ax=axes[1])
    axes[1].set_title('Line Plot for imdb_rating')

    kde_holding_up_perc = sns.kdeplot(df['holding_up_perc'], fill=False, ax=axes[2])
    sns.lineplot(x=kde_holding_up_perc.get_lines()[0].get_xdata(), y=kde_holding_up_perc.get_lines()[0].get_ydata(), ax=axes[2])
    axes[2].set_title('Line Plot for holding_up_perc')

    if 'us_viewers' in df.columns:
        kde_us_viewers = sns.kdeplot(df['us_viewers'], fill=False, ax=axes[3])
        sns.lineplot(x=kde_us_viewers.get_lines()[0].get_xdata(), y=kde_us_viewers.get_lines()[0].get_ydata(), ax=axes[3])
        axes[3].set_title('Line Plot for us_viewers')

    st.pyplot(fig)

    # Final Box Plots
    st.subheader("Final Box Plots")
    st.text("Final visualizations of numerical columns with box plots.")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))  # Change nrows to 1
    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=df[column], ax=axes[i])
    st.pyplot(fig)

    # Final DataFrame
    st.subheader("Final DataFrame")
    st.text("The modified dataset after all transformations.")
    st.dataframe(df)

    # Modified Data Information
    st.subheader("Modified Data Information")
    st.text("Summary information about the modified dataset:")

    # Number of rows and columns
    st.table(pd.DataFrame({
        'Column': df.columns,
        'Data Type': [df[col].dtype for col in df.columns],
        'Missing Values': df.isnull().sum()
    }))

elif section == "DataFrame Analysis":

    # Download the nltk data (if not already downloaded)
    nltk.download('punkt')

    # Create a new Streamlit page for DataFrame Analysis
    st.title("DataFrame Analysis")

    # Data Types Conversion & Parsing Dates and Handling Outliers
    df['original_air_date'] = pd.to_datetime(df['original_air_date'], format='%m/%d/%Y')
    df['year'] = df['original_air_date'].dt.year
    df['month'] = df['original_air_date'].dt.month

    # Handling Outliers
    Q1 = df['imdb_rating'].quantile(0.25)
    Q3 = df['imdb_rating'].quantile(0.75)
    IQR = Q3 - Q1
    df_outliers = df.loc[(df['imdb_rating'] >= Q1 - 1.5 * IQR) & (df['imdb_rating'] <= Q3 + 1.5 * IQR)]

    # Feature Scaling
    df['us_viewers_scaled'] = ((df['us_viewers'] - df['us_viewers'].min()) / (df['us_viewers'].max() - df['us_viewers'].min())).round(2)

    # New Features Extraction
    df['holding_up_perc'] = ((df['votes'] / df['us_viewers'])*1000).round(2)

    # Drop Unnecessary Columns
    columns_to_drop = ['votes', 'original_air_date']
    df.drop(columns=columns_to_drop, inplace=True)

    # Business Question: What is the average IMDb rating for each season?
    st.subheader("Average IMDb Rating per Season")
    avg_imdb_rating_per_season = df.groupby('season')['imdb_rating'].mean().round(1)
    st.table(pd.DataFrame({'Season': avg_imdb_rating_per_season.index, 'Average IMDb Rating': avg_imdb_rating_per_season.values}))

    # Business Question: What is the average viewership (US viewers) for each season?
    st.subheader("Average Viewership per Season")
    avg_viewership_per_season = (df.groupby('season')['us_viewers'].mean() / 1000000).round()
    st.table(pd.DataFrame({'Season': avg_viewership_per_season.index, 'Average Viewership (Millions)': avg_viewership_per_season.values}))

    # Business Question: Overall average IMDb rating and viewership across all seasons
    st.subheader("Overall Average IMDb Rating and Viewership")
    overall_avg_imdb_rating = round(df['imdb_rating'].mean(), 2)
    overall_avg_viewership = round(df['us_viewers'].mean() / 100000, 0) if 'us_viewers' in df.columns else None

    # Create a table for the overall average IMDb rating and viewership
    overall_avg_table = pd.DataFrame({
        'Metric': ['IMDb Rating', 'Viewership (Millions)'],
        'Average': [overall_avg_imdb_rating, overall_avg_viewership]
    })
    st.table(overall_avg_table)

    # Business Question: Highest and lowest-rated episodes
    st.subheader("Highest and Lowest-Rated Episodes")
    highest_rated_episode = df.loc[df['imdb_rating'].idxmax()]
    lowest_rated_episode = df.loc[df['imdb_rating'].idxmin()]

    # Create a table for the highest and lowest-rated episodes
    episode_table = pd.DataFrame({
        'Metric': ['Highest Rated Episode', 'Lowest Rated Episode'],
        'Title': [highest_rated_episode['title'], lowest_rated_episode['title']],
        'IMDb Rating': [highest_rated_episode['imdb_rating'], lowest_rated_episode['imdb_rating']]
    })
    st.table(episode_table)

    # Business Question: Top 3 directors with the highest average IMDb rating
    st.subheader("Top 3 Directors with Highest Average IMDb Rating")
    director_avg_rating = df.groupby('directed_by')['imdb_rating'].mean().nlargest(3).round(2)
    st.table(pd.DataFrame({'Director': director_avg_rating.index, 'Average IMDb Rating': director_avg_rating.values}))

    # Business Question: Top 3 writers with the highest average IMDb rating
    st.subheader("Top 3 Writers with Highest Average IMDb Rating")
    writer_avg_rating = df.groupby('written_by')['imdb_rating'].mean().nlargest(3).round(2)
    st.table(pd.DataFrame({'Writer': writer_avg_rating.index, 'Average IMDb Rating': writer_avg_rating.values}))

    # Business Question: Correlation between "holding_up_perc" and "imdb_rating"
    st.subheader("Correlation between Holding Up Percentage and IMDb Rating")
    correlation = df['holding_up_perc'].corr(df['imdb_rating'])
    st.latex(f"= {correlation:.2f}")

    # Business Question: Top-rated episodes based on IMDb ratings
    st.subheader("Top 5 Episodes Based on IMDb Ratings")
    top_rated_episodes = df.nlargest(5, 'imdb_rating')
    st.table(top_rated_episodes[['title', 'imdb_rating']])

    # Business Question: Top-rated episodes based on viewership
    st.subheader("Top 5 Episodes Based on Viewership")
    most_viewed_episodes = df.nlargest(5, 'us_viewers')
    st.table(most_viewed_episodes[['title', 'us_viewers']])

    # Business Question: Viewership trends over different episodes
    st.subheader("Viewership Trends Over Different Seasons")
    fig, ax = plt.subplots(figsize=(15, 3))  # Adjusted figsize here
    viewership_trend = df.groupby('episode_num_overall')['us_viewers'].sum().plot(ax=ax)
    plt.xlabel('episode_num_overall')
    plt.ylabel('Total Viewership')
    plt.title('Viewership Trends Over Different Episodes')
    st.pyplot(fig)

    # Scatter plot with linear regression line
    fig, ax = plt.subplots(figsize=(15, 3))  # Adjusted figsize here
    sns.regplot(x='holding_up_perc', y='imdb_rating', data=df, scatter_kws={'alpha': 0.5}, ax=ax)

    # Add labels and title
    plt.xlabel('Holding Up Percentage')
    plt.ylabel('IMDb Rating')
    plt.title('Relationship between Holding Up Percentage and IMDb Rating')

    # Fit a linear regression model
    X = df['holding_up_perc'].values.reshape(-1, 1)
    y = df['imdb_rating'].values
    model = LinearRegression().fit(X, y)

    # Get the slope (m) and intercept (b) of the regression line
    m = model.coef_[0]
    b = model.intercept_

    # Print the equation of the regression line
    equation = f"IMDb Rating = {m:.3f} * Holding Up Percentage + {b:.3f}"
    st.subheader("Mathematical equation for the relationship between Holding Up Percentage & IMDb Rating")
    st.text(f"\n{equation}\n\n")

    # Show the plot
    st.pyplot(fig)

elif section == "Scrapped Transcription":

    # Create new columns for season_number, episode_number, and episode_name
    tf[['season_number', 'episode_number', 'episode_name']] = tf['episode'].str.extract(r'(\d+)x(\d+) - (.+)')
    tf['episode_name'] = tf['episode_name'].str.strip()

    # Create a new column named 'script'
    tf['script'] = tf['episode_name'] + ' - ' + tf['name'] + ': ' + tf['line'].str.strip()

    # Change the order of the 'name' column and rename it to 'character'
    tf.insert(5, 'character', tf.pop('name'))

    # Drop the 'line' column
    tf = tf.drop(columns=['line', 'episode'])

    # Sort the DataFrame by season_number and episode_number
    tf = tf.sort_values(by=['season_number', 'episode_number']).reset_index(drop=True)

    # Streamlit app
    st.title('HIMYM Scrapped Transcription Data')

    # Display the DataFrame
    st.dataframe(tf.head(3))

    # Initialize a SessionState for tracking user clicks
    class SessionState:
        def __init__(self):
            self.clicked_yes = False
            self.clicked_no = False

    # Display the headline question
    st.title("Do producers needed to make an alternative ending?")

    # Initialize the session state
    session_state = SessionState()

    # Display the Yes button
    if not session_state.clicked_yes:
        yes_button = st.button("Yes, Because the Data Say So ..")
        if yes_button:
            session_state.clicked_yes = True
            session_state.clicked_no = False

    # Display the No button
    if not session_state.clicked_no:
        no_button = st.button("No, didn't study data verywell")
        if no_button:
            session_state.clicked_no = True
            session_state.clicked_yes = False

    # Change button colors based on user clicks
    if session_state.clicked_yes:
        st.success("C'est vraiment impressionnant")
    elif session_state.clicked_no:
        st.error("Come On! And you call yourself a data scientist?")

elif section == "HIMYM 3.5 AI Bot":
    st.title("HIMYM 3.5 AI ChatBot")

    # Create new columns for season_number, episode_number, and episode_name
    tf[['season_number', 'episode_number', 'episode_name']] = tf['episode'].str.extract(r'(\d+)x(\d+) - (.+)')
    tf['episode_name'] = tf['episode_name'].str.strip()

    # Create a new column named 'script'
    tf['script'] = tf['episode_name'] + ' - ' + tf['name'] + ': ' + tf['line'].str.strip()

    # Change the order of the 'name' column and rename it to 'character'
    tf.insert(5, 'character', tf.pop('name'))

    # Drop the 'line' column
    tf = tf.drop(columns=['line', 'episode'])

    # Sort the DataFrame by season_number and episode_number
    tf = tf.sort_values(by=['season_number', 'episode_number']).reset_index(drop=True)

    # Convert 'season_number' and 'episode_number' to integers
    tf['season_number'] = tf['season_number'].astype(int)
    tf['episode_number'] = tf['episode_number'].astype(int)

    # Add the 'link' column
    tf['link'] = tf.apply(lambda row: f"https://watchhowimetyourmother.co/episodes/how-i-met-your-mother-{row['season_number']}x{row['episode_number']}", axis=1)

    # Download the nltk data (if not already downloaded)
    nltk.download('punkt')

    # Function to calculate the similarity ratio between two strings using SequenceMatcher
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Function to find lines by text
    def find_lines_by_text(text):
        results = []
        for index, row in tf.iterrows():
            if text.lower() in row['script'].lower() or similar(text, row['script']) > 0.5:
                results.append((row['character'], row['season_number'], row['episode_number']))
                if len(results) == 10:
                    break
        return results

    # Function to get a random line by character
    def random_line_by_character():
        character_names = ['Ted', 'Tracy', 'Robin', 'Barney', 'Marshall', 'Lily']

        # Display buttons for character selection
        selected_character = st.radio("Choose a character:", character_names)

        character_lines = tf[tf['character'] == selected_character]

        if character_lines.empty:
            return None

        random_row = character_lines.sample(1).iloc[0]
        season = random_row['season_number']
        episode = random_row['episode_number']
        script = random_row['script']

        return season, episode, script

    # Function to extract the first image URL from HTML content
    def extract_first_image_url(html_content):
        try:
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the first 'a' tag with an 'href' attribute
            a_tag = soup.find('a', {'href': True})

            # Extract the 'href' attribute value
            href_value = a_tag['href']

            return href_value
        except Exception as e:
            print(f"Error extracting 'href' value: {e}")
            return None

    # Function for TF-IDF recommendations
    def tfidf_recommendations(search_word):
        tf['tokenized_script'] = tf['script'].apply(lambda x: word_tokenize(x.lower()))
        most_related_episodes_tf = tf[tf['tokenized_script'].apply(lambda tokens: any(difflib.get_close_matches(search_word, tokens, n=1, cutoff=0.7)))].copy()

        # Drop duplicates based on season_number and episode_number
        most_related_episodes_tf = most_related_episodes_tf.drop_duplicates(subset=['season_number', 'episode_number'])

        # Calculate matching percentages and sort by descending order
        most_related_episodes_tf['matching_percentage'] = most_related_episodes_tf['tokenized_script'].apply(lambda tokens: max(similar(search_word, token) for token in tokens))
        most_related_episodes_tf = most_related_episodes_tf.sort_values(by='matching_percentage', ascending=False)

        # Add the 'link' column
        most_related_episodes_tf['link'] = most_related_episodes_tf.apply(lambda row: f"https://watchhowimetyourmother.co/episodes/how-i-met-your-mother-{row['season_number']}x{row['episode_number']}", axis=1)

        return most_related_episodes_tf.head(10)[['season_number', 'episode_number', 'episode_name', 'matching_percentage', 'link']]

    # User chooses a mode
    mode = st.sidebar.radio("Choose a mode:", ["Find Lines by Text", "Random Line by Character", "TF-IDF Recommendations"])

    if mode == "Find Lines by Text":
        user_input = st.text_input("You: ", key="user_input")
        if st.button("Send", key="send_button"):
            results_task1 = find_lines_by_text(user_input)
            if results_task1:
                st.text("Characters who said the line or something similar:")
                for result in results_task1:
                    st.text(f"Character: {result[0]}, Season: {result[1]}, Episode: {result[2]}")
            else:
                st.text("No matching lines found.")

    elif mode == "Random Line by Character":
        random_line_result = random_line_by_character()
        if random_line_result:
            season, episode, script = random_line_result
            selected_character = script.split(' - ')[1].split(': ')[0]
            st.text(f"Random line by {selected_character} - Season: {season}, Episode: {episode}")
            st.text(f"Line: {' '.join(script.split(': ')[1:])}")
        else:
            st.text("No lines found for the selected character.")

    elif mode == "TF-IDF Recommendations":
        search_word = st.text_input("You: ", key="search_word_tfidf")
        if st.button("Send", key="send_button_tfidf"):
            most_related_episodes_tf = tfidf_recommendations(search_word)
            if not most_related_episodes_tf.empty:
                st.text("Top 5 Episodes with the Highest Matching Rates:")
                for index, row in most_related_episodes_tf.head(4).iterrows():
                    # Convert season_number and episode_number to integers
                    season_number = int(row['season_number'])
                    episode_number = int(row['episode_number'])
                    st.text(f"Season: {season_number}, Episode: {episode_number}, Name: {row['episode_name']} with {round(row['matching_percentage'] * 100, 2)}% Similarity Score")

                    # Display clickable link
                    st.markdown(f"A Free Watch Via: [{row['link']}]({row['link']})", unsafe_allow_html=True)
            else:
                st.text("There are no episodes most related to the search word.")
