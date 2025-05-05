import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import re
import string
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup
import time
import json
import nltk
import os
from fuzzywuzzy import process, fuzz
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# pip install --upgrade tensorflow

folder_path = "/Users/hotpotato/Desktop/Final Project/IMDb Review Dataset - ebD"

# Import the IMDb review dataset
file_list = [f"part-0{i}.json" for i in range(1, 7)]
full_paths = [os.path.join(folder_path, file) for file in file_list]

dataframes = []
for path in full_paths:
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f]
        df = pd.DataFrame(lines)
        dataframes.append(df)

# Combined all data
df_raw = pd.concat(dataframes, ignore_index=True)

# Show the part of result
df_raw.head()


# Flatten the dict in each column
flattened_reviews = df_raw.transpose().reset_index(drop=True)
flattened_reviews = pd.json_normalize(flattened_reviews[0])
print(flattened_reviews.columns)
flattened_reviews.head()
flattened_reviews.info()
flattened_reviews.isnull().sum()
flattened_reviews.groupby('rating').describe()


def clean_reviews_data(df):
    df = df.dropna(subset=["review_id"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df = df[df["rating"].between(1, 10, inclusive="both")]

    text_cols = ["reviewer", "movie", "review_summary", "review_detail"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        df = df.dropna(subset=["review_date"])

    if "spoiler_tag" in df.columns:
        valid_tags = ["NO", "YES"]
        df.loc[~df["spoiler_tag"].isin(valid_tags), "spoiler_tag"] = np.nan

    return df
df_cleaned = clean_reviews_data(flattened_reviews)
df_cleaned.head()


# let's modify the data to add more useful information for later
df_cleaned[["helpful_yes", "helpful_plus_not_helpful"]] = df_cleaned['helpful'].apply(pd.Series)

# casting from original strings to ints, but before that removing the "," as there are some big numbers like "1,264" causing issues
df_cleaned["helpful_yes"] = df_cleaned["helpful_yes"].str.replace(',', '').astype("int16")
df_cleaned["helpful_plus_not_helpful"] = df_cleaned["helpful_plus_not_helpful"].str.replace(',', '').astype("int16")
df_cleaned["helpful_ratio"] = df_cleaned["helpful_yes"] / df_cleaned["helpful_plus_not_helpful"]

# and let's check the data
display(df_cleaned.head())
display(df_cleaned.describe(include="all"))
print(df_cleaned.info(memory_usage="deep"))
print(df_cleaned.shape)

df_cleaned['review_date'] = pd.to_datetime(df_cleaned['review_date'], errors='coerce')
df_cleaned['review_year'] = df_cleaned['review_date'].dt.year
year_counts = df_cleaned['review_year'].value_counts().sort_index()
print(year_counts)

year_counts.plot(kind='bar', figsize=(10,5), color='skyblue')
plt.title("Distribution of Review Years")
plt.xlabel("Year")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

df_cleaned['publish_year'] = df_cleaned['movie'].str.extract(r"\((\d{4})\)").astype('Int64')
movie_year_counts = df_cleaned['publish_year'].value_counts().sort_index()
print(movie_year_counts)




def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return int(match.group(1))
    return None

df_cleaned["publish_year"] = df_cleaned["movie"].apply(extract_year)

df_years = df_cleaned.dropna(subset=["publish_year"])
df_years["publish_year"] = df_years["publish_year"].astype(int)

year_counts = df_years["publish_year"].value_counts().sort_index()

plt.figure(figsize=(16, 6))
plt.bar(year_counts.index, year_counts.values, color="#4C72B0")

plt.xticks(
    ticks=range(min(year_counts.index), max(year_counts.index)+1, 5),
    rotation=45
)
plt.title("Number of Movies per Year (Based on Extracted Year)", fontsize=16, weight='bold')
plt.xlabel("Publish Year", fontsize=14)
plt.ylabel("Number of Movies", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()




# Get the movie name
movie_set = set(df_cleaned['movie'].dropna().unique())

# Sort alphabetically
movie_list_sorted = sorted(movie_set)

print(f"A total of {len(movie_list_sorted)} movies were extracted")
print(movie_list_sorted[:120]) 




import re

# Step 1: Remove duplicated titles
unique_titles = pd.Series(movie_list_sorted)
unique_titles = unique_titles[~unique_titles.duplicated(keep=False)].tolist()

# Step 2: Define what qualifies as a movie
def is_movie_title(title):
    """
    Determine whether the title is a movie (not a TV series, video, episode, etc.)
    """
    match = re.search(r"\((\d{4})\)", title)
    is_clean = not re.search(r"(TV|Video|Season|Episode|Making|Behind|Featurette|Deleted Scenes|Up Close|Capturing|Crafting|Building|Miniature|Interview|Bonus|Part)", title, flags=re.IGNORECASE)
    return bool(match and is_clean)

# Step 3: Keep only the valid movies
movie_titles_only = [title for title in unique_titles if is_movie_title(title)]

print(movie_titles_only[:610]) 


def clean_years_only(title):
    """
    Clean only the year portion (inside parentheses) and remove specific punctuation marks (., #, ‘).
    Keep the rest of the title structure and punctuation as-is.
    """
    # Remove the unwanted characters
    for char in [".", "#", "‘"]:
        title = title.replace(char, "")

    # Split out and clean the year part
    parts = re.split(r'(\(.*?\))', title)
    cleaned_parts = []

    for part in parts:
        if part.startswith('(') and part.endswith(')'):
            inner = part[1:-1]
            inner_clean = re.sub(r'[^\d]', '', inner)
            if len(inner_clean) == 4:
                cleaned_parts.append(f"({inner_clean})")
            else:
                continue
        else:
            cleaned_parts.append(part)

    return "".join(cleaned_parts).strip()

movie_titles_only_cleaned = [
    clean_years_only(title)
    for title in movie_titles_only
]

# Preview
for t in movie_titles_only_cleaned[:300]:
    print(t)



labels = ["Movie" if t in movie_titles_only else "Other" for t in movie_list_cleaned]
df_classified = pd.DataFrame({"title": movie_list_cleaned, "type": labels})
type_counts = df_classified["type"].value_counts()
total = type_counts.sum()

# --- plotting ---
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# draw the pie, pushing labels & pct‐texts a bit outward
wedges, texts, autotexts = ax.pie(
    type_counts,
    labels=type_counts.index,
    startangle=140,
    colors = ["#6BAED6", "#FDAE6B"],
    autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})",
    pctdistance=0.75,
    labeldistance=1.05,
    textprops={'fontsize': 12},
    wedgeprops={'linewidth': 0}            # no edge lines
)

ax.set_title("Proportion of Movies vs Others in IMDb Review Dataset", fontsize=14, pad=20)
ax.axis("equal")  # keep it a circle

# move legend below the pie, centered
ax.legend(
    [w for w in wedges],
    [f"{k}: {v:,} items" for k, v in type_counts.items()],
    title="Counts",
    title_fontsize=12,
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

plt.show()



# Extraction Year
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return int(match.group(1))
    return None

years = [extract_year(title) for title in movie_titles_only]
df_years = pd.DataFrame(years, columns=["year"])
df_years = df_years.dropna()
df_years["year"] = df_years["year"].astype(int)

# Statistics by year
year_counts = df_years["year"].value_counts().sort_index()

# Plot
plt.figure(figsize=(16, 6))
plt.bar(year_counts.index, year_counts.values, color="#4C72B0")  # Removed edgecolor

# Ticks, title, labels
plt.xticks(ticks=range(min(year_counts.), max(year_counts.index), 5), rotation=45)
plt.title("Number of Movies per Year (Filtered Titles)", fontsize=16, weight='bold')
plt.xlabel("Release Year", fontsize=14)
plt.ylabel("Number of Movies", fontsize=14)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Layout
plt.tight_layout()
plt.show()


df_year_counts = year_counts.reset_index()
df_year_counts.columns = ["Year", "Movie Count"]

df_year_counts = df_year_counts.sort_values("Year")

print(df_year_counts.to_string(index=False))  




# Line chart
plt.figure(figsize=(16, 6))
plt.plot(year_counts.index, year_counts.values, marker='o', linestyle='-', color="#4C72B0")

plt.title("Number of Movies per Year (Line Chart)", fontsize=16, weight='bold')
plt.xlabel("Release Year", fontsize=14)
plt.ylabel("Number of Movies", fontsize=14)

plt.xticks(ticks=range(min(year_counts.index), max(year_counts.index)+1, 5), rotation=45)

plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



# Load the flattened_reviews DataFrame (assuming it's already loaded in your environment)
# Display review_date distribution by year
review_date_distribution = flattened_reviews['review_date'].dropna()

# Convert to datetime if not already
review_date_distribution = pd.to_datetime(review_date_distribution, errors='coerce')
review_date_distribution = review_date_distribution.dropna()

# Group by year and count
review_year_counts = review_date_distribution.dt.year.value_counts().sort_index()


plt.figure(figsize=(12, 6))
review_year_counts.plot(kind='bar', color="#6BAED6")
plt.title("Review Count by Year", fontsize=16, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Reviews", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




# Filter movies from 2010 to 2020
df_selected = df_movies[df_movies["year"] == 2020].copy()

# Optionally sample a subset if needed
df_sampled = df_selected.sample(n=1000, random_state=42)

print(f"Selected {len(df_sampled)} movies 2020")





# Step 1: Extract release year from each title
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return int(match.group(1))
    return None

# Create DataFrame and extract year
df_movies = pd.DataFrame({'movie': movie_titles_only_cleaned})
df_movies['year'] = df_movies['movie'].apply(extract_year)

# Step 2: Select movies released only 2020 (up to 1000)
df_range = df_movies[df_movies['year'] == 2020].copy()
df_sampled = df_range.sample(n=min(3000, len(df_range)), random_state=42)

# Step 3: Count number of reviews per movie from df_cleaned
review_counts = df_cleaned.groupby("movie")["review_detail"].count().reset_index()
review_counts.columns = ["movie", "review_count"]

df_sampled = df_sampled.merge(review_counts, on="movie", how="left") # Merge actual review counts into df_sampled

df_sampled["review_count"] = df_sampled["review_count"].fillna(0).astype(int) # Fill any missing counts with 0 (in case a sampled movie had no matched reviews)

# Step 4: Remove movies with fewer than 30 reviews
df_filtered = df_sampled[df_sampled['review_count'] >= 30].copy()

# Step 5: Remove movies with Chinese characters in their titles
def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

df_filtered = df_filtered[~df_filtered['movie'].apply(contains_chinese)].copy()

# Final result
df_filtered = df_filtered.sort_values(by="review_count", ascending=False).reset_index(drop=True)# sort by review count descending
print(f"Final movie count: {len(df_filtered)}")
display(df_filtered.head())




print(df_filtered）

df_filtered.to_excel("filtered_movies.xlsx", index=False)


nm = pd.read_excel("/Users/hotpotato/Desktop/Final Project/filtered_movies.xlsx")
movies = nm["movie"].tolist()

# Unify the film titles and remove the year
def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()

# Grab the Country field in the infobox
def extract_country(td):
    for sup in td.find_all("sup"):
        sup.decompose()  
    if td.find_all("a"):
        return ", ".join([a.get_text(strip=True) for a in td.find_all("a")])
    else:
        return td.get_text(separator=", ").strip()

# Use possible formats to get Wikipedia page
def get_country(movie_title):
    base = movie_title.replace(" ", "_")
    candidates = [
        f"https://en.wikipedia.org/wiki/{base}_(2020_film)",
        f"https://en.wikipedia.org/wiki/{base}_(film)",
        f"https://en.wikipedia.org/wiki/{base}"
    ]

    for url in candidates:
        try:
            res = requests.get(url, timeout=10)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.content, "html.parser")
            infobox = soup.find("table", class_=lambda x: x and "infobox" in x)
            if infobox:
                for row in infobox.find_all("tr"):
                    if row.th and any(k in row.th.text for k in ["Country", "Countries", "Country of origin"]):
                        country = extract_country(row.td)
                        return country, url
        except Exception as e:
            print(f"Error for {url}: {e}")
            continue

    return "Not Found", "No Valid Link"

# Execute the main logic
results = []
for m in movies:
    clean = clean_title(m)
    country, link = get_country(clean)
    print(f"{m} → {country} ({link})")
    results.append({"movie": m, "country": country, "source_link": link})
    time.sleep(1.5)  

# Transform into DataFrame
df = pd.DataFrame(results)
df.to_csv("wiki_movie_country_final.csv", index=False)

# pip install fuzzywuzzy python-Levenshtein

# Step 1: Read filtered movie list

df_filtered = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies.xlsx")
target_years = set(df_filtered["year"].astype(int).tolist())

# Step 2: Read the IMDb raw data in chunks, keeping only the movies and years you need

imdb_cols = ["tconst", "titleType", "primaryTitle", "startYear"]
reader = pd.read_csv( 
"/Users/hotpotato/Desktop/Final Project/Dataset/title.basics.tsv.gz", 
sep="\t", 
dtype=str, 
usecols=imdb_cols, chunksize=200000
)

chunks = []
for chunk in reader: 
mask = (chunk["titleType"] == "movie") & (chunk["startYear"].isin(map(str, target_years))) 
filtered = chunk.loc[mask, ["tconst", "primaryTitle", "startYear"]] 
chunks.append(filtered)
df_titles = pd.concat(chunks, ignore_index=True)
df_titles["startYear"] = df_titles["startYear"].astype(int)

# Step 3: Build a fast index grouped by year

year_to_titles = {}
year_to_id = {}
for year, grp in df_titles.groupby("startYear"): 
titles = grp["primaryTitle"].tolist() 
year_to_titles[year] = titles year_to_id[year] = dict(zip(grp["primaryTitle"], grp["tconst"]))

# Step 4: Clean up movie names (remove the year in brackets)

def clean_title(title):
return re.sub(r"\s*\(\d{4}\)", "", title).strip()

df_filtered["clean_movie"] = df_filtered["movie"].apply(clean_title)

# Step 5: Matching function, first precise then fuzzy (using fuzzywuzzy)

def match_imdb(row):
title = row["clean_movie"]
year = int(row["year"])
candidates = year_to_titles.get(year, [])
# Do case-insensitive exact match first
for cand in candidates:
if cand.lower() == title.lower(): 
imdb_id = year_to_id[year][cand] 
return imdb_id, f"https://www.imdb.com/title/{imdb_id}/" 

# Do fuzzy matching again 
best = process.extractOne( 
title, 
candidates, 
scorer=fuzz.WRatio, 
score_cutoff=90 
) 
if best: 
matched = best[0] 
imdb_id = year_to_id[year][matched] 
return imdb_id, f"https://www.imdb.com/title/{imdb_id}/" 
return "Not found", "N/A"

# -----------------------
# Step 6: Apply in batches and export
# -----------------------
df_filtered[["imdb_id", "imdb_link"]] = df_filtered.apply( 
match_imdb, axis=1, result_type="expand"
)

df_filtered.to_excel( "/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_with_imdb_links_fast.xlsx", 
index=False
)
print("Done in: filtered_movies_with_imdb_links_fast.xlsx")


# pip install pandas openpyxl requests beautifulsoup4



# —— Configuration ——
INPUT_XLSX = "filtered_movies_with_imdb_links_fast.xlsx" # with imdb_link
OUTPUT_XLSX = "filtered_movies_with_country_only.xlsx" # only with country
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_SEC = 1.0

# —— Read and filter ——
df = pd.read_excel(INPUT_XLSX)
df = df[(df.imdb_id != "Not found") & (df.imdb_link != "N/A")].reset_index(drop=True)
df["country_of_origin"] = "" # Add a new column

# —— Extract Country Function --
def extract_country(imdb_url):
try:
resp = requests.get(imdb_url, headers=HEADERS, timeout=10)
resp.raise_for_status()
soup = BeautifulSoup(resp.content, "html.parser")

# Locate data-testid="title-details-origin"
li = soup.find("li", {"data-testid": "title-details-origin"})
if not li:
return "Not listed"

# Delete possible footnotes sup
for sup in li.find_all("sup"):
sup.decompose()

# Extract all text in <a> tags
links = li.select("div.ipc-metadata-list-item__content-container a")
if links:
return ", ".join(a.text.strip() for a in links)
else:
# If not <a>, then directly get the container text
div = li.find("div", class_="ipc-metadata-list-item__content-container")
return div.get_text(", ", strip=True) if div else "Not listed"
except:
return "Error"

# —— Loop to grab the country ——
for idx, row in df.iterrows():
country = extract_country(row["imdb_link"])
df.at[idx, "country_of_origin"] = country
print(f"[{idx+1}/{len(df)}] {row['movie']} → {country}")
time.sleep(SLEEP_SEC)

# —— Save the result ——
df.to_excel(OUTPUT_XLSX, index=False)
print("Done in:", OUTPUT_XLSX)


# —— Configuration ——
INPUT_XLSX  = "/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_with_country_only.xlsx" 
SLEEP_SEC   = 1.0
HEADERS     = {"User-Agent": "Mozilla/5.0"}

# —— Reload the intermediate result ——
df = pd.read_excel(INPUT_XLSX)

# —— The same extract_country function ——
def extract_country(imdb_url):
    try:
        resp = requests.get(imdb_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        li = soup.find("li", {"data-testid": "title-details-origin"})
        if not li:
            return "Not listed"

        for sup in li.find_all("sup"):  # remove footnote markers
            sup.decompose()

        links = li.select("div.ipc-metadata-list-item__content-container a")
        if links:
            return ", ".join(a.text.strip() for a in links)
        else:
            div = li.find("div", class_="ipc-metadata-list-item__content-container")
            return div.get_text(", ", strip=True) if div else "Not listed"
    except:
        return "Error"

# —— Identify rows that previously errored —— 
error_mask = df["country_of_origin"] == "Error"
error_count = error_mask.sum()
print(f"Found {error_count} rows with 'Error' country; retrying those...")

# —— Retry extraction for those rows —— 
for idx in df[error_mask].index:
    imdb_link = df.at[idx, "imdb_link"]
    # Retry
    country = extract_country(imdb_link)
    df.at[idx, "country_of_origin"] = country
    print(f"  Retried [{idx}] {df.at[idx,'movie']} → {country}")
    time.sleep(SLEEP_SEC)

# —— Save updated results —— 
df.to_excel(INPUT_XLSX, index=False)
print(" Retried all errors and updated:", INPUT_XLSX)




df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_with_country_only.xlsx")

# Split the multi-country situation and count each possible multiple countries of each movie separately
countries_series = (
    df['country_of_origin']
      .dropna()
      .apply(lambda s: [c.strip() for c in s.split(',')])
)

all_countries = [c for sublist in countries_series for c in sublist]
country_counts = pd.Series(all_countries).value_counts()


plt.figure(figsize=(10, 6))
country_counts.plot(kind='bar')
plt.xlabel('Country of Origin')
plt.ylabel('Number of Movies')
plt.title('Distribution of Movies by Country of Origin')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




df = pd.read_excel("filtered_movies_with_country_only.xlsx")

# Split multiple countries and count each possible country for each movie
countries_series = (
df['country_of_origin']
.dropna()
.apply(lambda s: [c.strip() for c in s.split(',')])
)

# Flatten the list and count
all_countries = [c for sublist in countries_series for c in sublist]
country_counts = pd.Series(all_countries).value_counts()

# Calculate percentages and classify <1% as "Others"
total = country_counts.sum()
pct = country_counts / total * 100
large = country_counts[pct >= 1]
small_sum = country_counts[pct < 1].sum()
new_counts = large.copy()
if small_sum > 0:
new_counts['Others'] = small_sum

# Draw a pie chart using the desaturated Pastel1 palette
plt.figure(figsize=(8, 8))
cmap = plt.get_cmap('Pastel1')
colors = cmap(range(len(new_counts)))
new_counts.plot(
kind='pie',
autopct='%1.1f%%',
startangle=140,
counterclock=False,
colors=colors
)
plt.ylabel('')
plt.title('Movies by Country of Origin (Others <1%)')
plt.tight_layout()
plt.show()




# Load the Excel file with country information
df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_with_country_only.xlsx")

# Filter rows where Country of origin contains 'United States'
df_us = df[df["country_of_origin"].str.contains("United States", case=False, na=False)].reset_index(drop=True)

# Save the filtered results to a new Excel file
output_path = "/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_us_only.xlsx"
df_us.to_excel(output_path, index=False)

output_path

# Configuration
INPUT_XLSX = "/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_us_only.xlsx"
OUTPUT_XLSX = "/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_us_with_release.xlsx"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_SEC = 1.5

# Read US-only movie list
df = pd.read_excel(INPUT_XLSX)
df["release_date"] = ""  # Add new column

def extract_release_date(imdb_url):
    """
    Extract only the Release date from the IMDb page (using the data-testid tag)
    """
    try:
        resp = requests.get(imdb_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        rd_li = soup.find("li", {"data-testid": "title-details-releasedate"})
        if not rd_li:
            return "Not listed"

        first_li = rd_li.select_one(
            "div.ipc-metadata-list-item__content-container ul li"
        )
        if not first_li:
            return "Not listed"

        a_tag = first_li.find("a")
        return a_tag.text.strip() if a_tag else first_li.get_text(strip=True)
    except Exception:
        return "Error"

# Loop to capture the Release date of all US movies
for idx, row in df.iterrows():
    imdb_link = row.get("imdb_link", "")
    if imdb_link and imdb_link != "N/A":
        df.at[idx, "release_date"] = extract_release_date(imdb_link)
        print(f"[{idx+1}/{len(df)}] {row['movie']} → {df.at[idx, 'release_date']}")
    else:
        df.at[idx, "release_date"] = "No link"
    time.sleep(SLEEP_SEC)

# Save the new table with Release date
df.to_excel(OUTPUT_XLSX, index=False)
print(f"Saved to: {OUTPUT_XLSX}")


# ── Configuration ─────────────────────────────────────
INPUT_XLSX  = "/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_us_with_release.xlsx"
OUTPUT_CSV  = "/Users/hotpotato/Desktop/Final Project/code_output/ "
FAILED_CSV  = "/Users/hotpotato/Desktop/Final Project/code_output/failed_movies.csv"
HEADERS     = {"User-Agent": "Mozilla/5.0"}
SLEEP_SEC   = (1, 3)  # Sleep random between 1-3 seconds
MAX_WEEKS   = 6

# ── Read movie list ────────────────────────────────────
df = pd.read_excel(INPUT_XLSX)
movies = df["movie"].dropna().astype(str).tolist()

# ── Prepare HTTP session with retry ─────────────────────
session = requests.Session()
session.headers.update(HEADERS)
retry_strategy = Retry(
    total=2,
    backoff_factor=0.3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ── Helper functions ───────────────────────────────────
def extract_year(title: str):
    m = re.search(r"\((\d{4})\)", title)
    return m.group(1) if m else None

def generate_slugs(title: str):
    year = extract_year(title)
    if not year:
        return
    base = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    norm = re.sub(r"[^0-9A-Za-z -]", "", base)
    slug = norm.replace(" ", "-")
    yield f"{slug}-({year})"
    yield slug
    if base.lower().startswith("the "):
        rest = norm[4:].strip().replace(" ", "-")
        yield f"{rest}-The-({year})"

def fetch_weekly_box_office(slug: str):
    url = f"https://www.the-numbers.com/movie/{slug}#tab=box-office"
    resp = session.get(url, timeout=(3, 5))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table_tag = soup.find("table", id="box_office_weekly")
    if not table_tag:
        for df_tbl in pd.read_html(StringIO(resp.text)):
            if "Week" in df_tbl.columns:
                table_tag = df_tbl
                break
        else:
            return None

    if hasattr(table_tag, "get_text"):
        df_tbl = pd.read_html(StringIO(str(table_tag)))[0]
    else:
        df_tbl = table_tag

    return df_tbl.iloc[:MAX_WEEKS].reset_index(drop=True)

# ── Main crawl loop ────────────────────────────────────
no_slug = []
fetch_fail = []
all_results = []

for idx, title in enumerate(movies, 1):
    year = extract_year(title)
    if not year:
        no_slug.append(title)
        print(f"[{idx}/{len(movies)}] {title} → ❗ No year found, skipped")
        continue

    df_weeks = None
    used_slug = None

    for slug in generate_slugs(title):
        print(f"[{idx}/{len(movies)}] Trying slug: {slug} for movie: {title}")
        try:
            df_weeks = fetch_weekly_box_office(slug)
            if isinstance(df_weeks, pd.DataFrame):
                used_slug = slug
                break
        except Exception as e:
            print(f"    ✖ Failed slug {slug}: {str(e)[:100]}")  # Only show first 100 chars
            df_weeks = None
        time.sleep(0.5)  # between slugs

    status = "no slug" if not used_slug else ("fetched" if df_weeks is not None else "failed")
    print(f"→ {title} | Status: {status} | Used slug: {used_slug}")

    if df_weeks is None:
        fetch_fail.append(title)
    else:
        df_weeks.insert(0, "Movie", title)
        df_weeks.insert(1, "Slug", used_slug)
        all_results.append(df_weeks)

    time.sleep(random.uniform(*SLEEP_SEC))  # between movies

# ── Reporting and Save ─────────────────────────────────
print(" Summary:")
print(f"- Movies without year: {len(no_slug)}")
print(f"- Movies failed fetching: {len(fetch_fail)}")
print(f"- Movies successfully fetched: {len(all_results)}")

if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f" Saved box office data to: {OUTPUT_CSV}")
else:
    print("No box office data was fetched.")

if fetch_fail:
    pd.DataFrame({"Failed Movies": fetch_fail}).to_csv(FAILED_CSV, index=False)
    print(f" Saved failed movies list to: {FAILED_CSV}")





from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Read Excel
path = "/Users/hotpotato/Desktop/Final Project/filtered_movies_us_with_release.xlsx"
df = pd.read_excel(path)

movie_column_name = "movie"
movies = df[movie_column_name].dropna().astype(str).unique().tolist()

# 2. Configure requests Session
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/113.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.the-numbers.com/"
}
session = requests.Session()
session.headers.update(HEADERS)
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# 3. Helper functions

def extract_year(title):
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None

def make_slug_variants(title):
    """
    Generate multiple slug variants for better match rates.
    """
    year = extract_year(title)
    name = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    name_clean = re.sub(r"[^0-9A-Za-z\s:.'&,-]", "", name).strip()
    words = name_clean.split()
    slugs = []

    # Basic slug
    slugs.append("-".join(words) + f"-({year})")

    # Article last variant (The Movie → Movie-The)
    if words and words[0].lower() in ["the", "a", "an"]:
        alt_words = words[1:] + [words[0]]
        slugs.append("-".join(alt_words) + f"-({year})")

    # No year variant (e.g., https://www.the-numbers.com/movie/Scoob)
    slugs.append("-".join(words))
    if words and words[0].lower() in ["the", "a", "an"]:
        slugs.append("-".join(alt_words))

    return slugs

# 4. Crawl function

def fetch_weekly_box_office(slug, max_weeks=6, timeout=(5,15)):
    url = f"https://www.the-numbers.com/movie/{slug}#tab=box-office"
    print("Fetching:", url)
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", id="box_office_weekly")
    if not table:
        heading = soup.find(lambda t: t.name in ["h2","h3"] and "Weekly Box Office Performance" in t.text)
        if heading:
            table = heading.find_next("table")
    if not table:
        raise ValueError(f"No weekly table for slug '{slug}'")

    html_tbl = str(table)
    df_tbl = pd.read_html(StringIO(html_tbl))[0]
    return df_tbl.iloc[:max_weeks].reset_index(drop=True)

# 5. Main process
all_results = []
for title in movies:
    slug_variants = make_slug_variants(title)
    success = False
    for slug in slug_variants:
        try:
            df6 = fetch_weekly_box_office(slug)
            df6.insert(0, "Movie", title)
            all_results.append(df6)
            print("✔ OK:", title, "→", slug)
            success = True
            break
        except Exception as e:
            print("✖ Failed:", slug)
    if not success:
        print("✖ Totally Failed:", title)
    time.sleep(random.uniform(1.5, 2.5))

# 6. Save results
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    print(final_df.head())
    final_df.to_csv("weekly1-6test2.csv", index=False)
    print(" Saved to weekly1-6test2.csv")
else:
    print("No data was obtained. Check if URLs work manually.")





# Load the scraped weekly box office file
df_weekly = pd.read_csv("/Users/hotpotato/Desktop/Final Project/code_output/the_numbers_box_office_weeks1-6.csv")

# Drop actor rows or any rows missing 'Week' and 'Gross'
df_weekly_cleaned = df_weekly[df_weekly['Week'].notna() & df_weekly['Gross'].notna()].copy()

# Normalize movie names for grouping
df_weekly_cleaned['Movie'] = df_weekly_cleaned['Movie'].str.strip()

# Summarize weekly total gross for Weeks 1-6 per movie
weekly_summary = df_weekly_cleaned.groupby('Movie')['Gross'].apply(list).reset_index()

# Create a mapping of all attempted movies
attempted_movies = df_weekly['Movie'].dropna().unique()

# Mark success/fail based on whether the movie had any valid weekly gross data
successful_movies = df_weekly_cleaned['Movie'].unique()
summary_table = pd.DataFrame({'Movie': attempted_movies})
summary_table['Status'] = summary_table['Movie'].apply(lambda x: 'Success' if x in successful_movies else 'No Data')

# Merge weekly gross data (Weeks 1–6)
summary_table = pd.merge(summary_table, weekly_summary, on='Movie', how='left')
summary_table.rename(columns={'Gross': 'Weekly Gross (1–6)'}, inplace=True)

summary_table







# Step 1: Read the release_date file
release_df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_movies_us_with_release.xlsx")
release_df = release_df[['movie', 'release_date']].dropna()

# Step 2: Clean the release_date string (extract only the date)
release_df['release_date'] = release_df['release_date'].str.extract(r'([A-Za-z]+\s\d{1,2},\s\d{4})')[0]
release_df['release_date'] = pd.to_datetime(release_df['release_date'], errors='coerce')

# Step 3: Normalize the movie name for matching
release_df['movie_norm'] = release_df['movie'].str.strip().str.lower()
summary_table['movie_norm'] = summary_table['Movie'].str.strip().str.lower()

# Step 4: Merge
summary_table_with_date = summary_table.merge( 
release_df[['movie_norm', 'release_date']], 
on='movie_norm', 
how='left'
)

# Step 5: Save results
summary_table_with_date.to_excel( 
"/Users/hotpotato/Desktop/Final Project/code_output/summary_table_with_release_date.xlsx", 
index=False
)

print(f"Successful match! The final table contains {len(summary_table_with_date)} movies.")



import pandas as pd

# Step 1: Load files
summary_df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/summary_table_with_release_date.xlsx")
reviews_df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_reviews_only.xlsx")

# Step 2: Normalize movie names (strip/lower)
summary_df['movie_norm'] = summary_df['Movie'].str.strip().str.lower()
reviews_df['movie_norm'] = reviews_df['movie'].str.strip().str.lower()

# Step 3: Merge release_date into reviews
reviews_df = reviews_df.merge(summary_df[['movie_norm', 'release_date']], on='movie_norm', how='left')

# Step 4: Convert to datetime safely
reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'], errors='coerce')
reviews_df['release_date'] = pd.to_datetime(reviews_df['release_date'], errors='coerce')

# Step 5: Calculate days and weeks since release
reviews_df['days_since_release'] = (reviews_df['review_date'] - reviews_df['release_date']).dt.days
reviews_df['week_since_release'] = (reviews_df['days_since_release'] // 7) + 1

# Step 6: Filter: 1-6 weeks, and days_since_release >= 0
filtered_reviews = reviews_df[ 
(reviews_df['release_date'].notna()) & 
(reviews_df['week_since_release'] >= 1) & 
(reviews_df['week_since_release'] <= 6) & 
(reviews_df['days_since_release'] >= 0)
]

# Step 7: Save result
filtered_reviews.to_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_reviews_weeks1-6_final.xlsx", index=False)

print(f"Successfully filtered out {len(filtered_reviews)} reviews, saved to filtered_reviews_weeks1-6_final.xlsx")



filtered_reviews.head()



# Count number of reviews per week
review_week_counts = df_reviews_filtered['review_week'].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 5))
plt.bar(review_week_counts.index, review_week_counts.values, color="#4C72B0")

# Labels and title
plt.title("Review Distribution by Week Since Release", fontsize=14, weight='bold')
plt.xlabel("Week Since Release", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)
plt.xticks(review_week_counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()








# !pip install wordcloud
# !pip install spacy
# !pip install textblob



#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



#importing the training data
imdb_data=pd.read_csv('/Users/hotpotato/Desktop/Final Project/IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)



#Summary of the dataset
imdb_data.describe()



#sentiment count
imdb_data['sentiment'].value_counts()



#split the dataset  
#train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)



import nltk
nltk.download('stopwords')



#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')



# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(str(text), "html.parser")
    return soup.get_text()

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# Apply the cleaning function
imdb_data['review'] = imdb_data['review'].apply(denoise_text)




#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_special_characters)



#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(simple_stemmer)



#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_stopwords)



#normalized train reviews
norm_train_reviews=imdb_data.review[:40000]
norm_train_reviews[0]
#convert dataframe to string
#norm_train_string=norm_train_reviews.to_string()
#Spelling correction using Textblob
#norm_train_spelling=TextBlob(norm_train_string)
#norm_train_spelling.correct()
#Tokenization using Textblob
#norm_train_words=norm_train_spelling.words
#norm_train_words



#Normalized test reviews
norm_test_reviews=imdb_data.review[40000:]
norm_test_reviews[45005]
##convert dataframe to string
#norm_test_string=norm_test_reviews.to_string()
#spelling correction using Textblob
#norm_test_spelling=TextBlob(norm_test_string)
#print(norm_test_spelling.correct())
#Tokenization using Textblob
#norm_test_words=norm_test_spelling.words
#norm_test_words



#word cloud for positive review words
plt.figure(figsize=(10,10))
positive_text=norm_train_reviews[1]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show



#Word cloud for negative review words
plt.figure(figsize=(10,10))
negative_text=norm_train_reviews[8]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
negative_words=WC.generate(negative_text)
plt.imshow(negative_words,interpolation='bilinear')
plt.show



#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0.0,max_df=1.0,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(norm_train_reviews)
#transformed test reviews
cv_test_reviews=cv.transform(norm_test_reviews)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)
#vocab=cv.get_feature_names()-toget feature names



#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0.0,max_df=1.0,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(norm_train_reviews)
#transformed test reviews
tv_test_reviews=tv.transform(norm_test_reviews)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)



#labeling the sentiment data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data=lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)



#Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)



#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments.ravel())
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments.ravel())
print(lr_tfidf)



#Predicting the model for bag of words
lr_bow_predict=lr.predict(cv_test_reviews)
print(lr_bow_predict)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
print(lr_tfidf_predict)



#Accuracy score for bag of words
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)



#Classification report for bag of words 
lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])
print(lr_bow_report)

#Classification report for tfidf features
lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# --- Confusion Matrix for Bag of Words ---
cm_bow = confusion_matrix(test_sentiments, lr_bow_predict, labels=[1, 0])
disp_bow = ConfusionMatrixDisplay(confusion_matrix=cm_bow, display_labels=['Positive', 'Negative'])
disp_bow.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Logistic Regression with BoW")
plt.show()

# --- Confusion Matrix for TF-IDF ---
cm_tfidf = confusion_matrix(test_sentiments, lr_tfidf_predict, labels=[1, 0])
disp_tfidf = ConfusionMatrixDisplay(confusion_matrix=cm_tfidf, display_labels=['Positive', 'Negative'])
disp_tfidf.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Logistic Regression with TF-IDF")
plt.show()




import joblib
# Save the Bag-of-Words model
joblib.dump(lr_bow, 'logistic_bow_model.pkl')
# Save the TF-IDF model
joblib.dump(lr_tfidf, 'logistic_tfidf_model.pkl')




#training the linear svm
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)
#fitting the svm for bag of words
svm_bow=svm.fit(cv_train_reviews,train_sentiments.ravel())
print(svm_bow)
#fitting the svm for tfidf features
svm_tfidf=svm.fit(tv_train_reviews,train_sentiments.ravel())
print(svm_tfidf)



#Predicting the model for bag of words
svm_bow_predict=svm.predict(cv_test_reviews)
print(svm_bow_predict)
#Predicting the model for tfidf features
svm_tfidf_predict=svm.predict(tv_test_reviews)
print(svm_tfidf_predict)



#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)
print("svm_bow_score :",svm_bow_score)
#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(test_sentiments,svm_tfidf_predict)
print("svm_tfidf_score :",svm_tfidf_score)



#Classification report for bag of words 
svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])
print(svm_bow_report)
#Classification report for tfidf features
svm_tfidf_report=classification_report(test_sentiments,svm_tfidf_predict,target_names=['Positive','Negative'])
print(svm_tfidf_report)




# Bag of Words Confusion Matrix for SVM
cm_svm_bow = confusion_matrix(test_sentiments, svm_bow_predict)
print("SVM BoW Accuracy:", accuracy_score(test_sentiments, svm_bow_predict))

disp_bow = ConfusionMatrixDisplay(confusion_matrix=cm_svm_bow, display_labels=['Negative', 'Positive'])
disp_bow.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: SVM with BoW")
plt.show()

# TF-IDF Confusion Matrix for SVM
cm_svm_tfidf = confusion_matrix(test_sentiments, svm_tfidf_predict)
print("SVM TF-IDF Accuracy:", accuracy_score(test_sentiments, svm_tfidf_predict))

disp_tfidf = ConfusionMatrixDisplay(confusion_matrix=cm_svm_tfidf, display_labels=['Negative', 'Positive'])
disp_tfidf.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: SVM with TF-IDF")
plt.show()




joblib.dump(cm_svm_bow, 'svm_bow_model.pkl')
joblib.dump(cm_svm_tfidf, 'svm_tfidf_model.pkl')




!pip install xgboost



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Initialize and train XGBoost
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(tv_train_reviews, train_sentiments.ravel())

# Predict and evaluate
xgb_preds = xgb_model.predict(tv_test_reviews)
print("XGBoost Accuracy:", accuracy_score(test_sentiments, xgb_preds))



joblib.dump(xgb_model, 'xgb_model.pkl')



# !pip install tensorflow
# pip install --upgrade tensorflow
# conda install python=3.10
# !pip uninstall -y numpy
# !pip install numpy==1.24.3



import numpy as np
print(np.__version__)



import nltk

nltk.download('punkt')              
nltk.download('punkt_tab')          
nltk.download('popular')           

from nltk.tokenize import word_tokenize
print(word_tokenize("Hello, world!"))



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("/Users/hotpotato/Desktop/Final Project/IMDB Dataset.csv")

# Clean text
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word not in stop_words])

df['clean_review'] = df['review'].apply(clean_text)

# Encode labels
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)



# !pip install tensorflow==2.16.2



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

X = df['clean_review']
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
maxlen = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)



embedding_index = {}
with open("/Users/hotpotato/Desktop/Final Project/glove.840B.300d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        try:
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
        except ValueError:
            continue

print("GloVe embeddings loaded:", len(embedding_index))



embedding_dim = 300
word_index = tokenizer.word_index
num_words = min(10000, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words and word in embedding_index:
        embedding_matrix[i] = embedding_index[word]



from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import History

cnn_glove = Sequential()
cnn_glove.add(Embedding(input_dim=num_words,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))
cnn_glove.add(Conv1D(128, 5, activation='relu'))
cnn_glove.add(GlobalMaxPooling1D())
cnn_glove.add(Dense(64, activation='relu'))
cnn_glove.add(Dropout(0.5))
cnn_glove.add(Dense(1, activation='sigmoid'))

cnn_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = cnn_glove.fit(X_train_pad, y_train,
                        epochs=5,
                        batch_size=128,
                        validation_split=0.2)


joblib.dump(cnn_glove, 'cnn_glove.pkl')


# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("CNN Accuracy (GloVe)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("CNN Loss (GloVe)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

cnn_preds = (cnn_glove.predict(X_test_pad) > 0.5).astype("int32")
print("GloVe CNN Accuracy:", accuracy_score(y_test, cnn_preds))

cm = confusion_matrix(y_test, cnn_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: CNN with GloVe")
plt.show()



# Read movie review data with sentiment scores
df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_reviews_with_sentiment.xlsx")

# Make sure the date format is correct
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df = df.dropna(subset=['review_date', 'release_date'])

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Calculate the number of weeks after the release of each review (1-6 weeks)
df['week'] = ((df['review_date'] - df['release_date']).dt.days // 7 + 1)
df = df[(df['week'] >= 1) & (df['week'] <= 6)]

# Group by movie and week, count the number of reviews in each group
weekly_counts = df.groupby(['movie', 'week']).size().reset_index(name='review_count')

weekly_counts



# Load the dataset
df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_reviews_with_sentiment.xlsx")

# Convert review_date to datetime
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

# Ensure sentiment_score and week are numeric
df["sentiment_score"] = pd.to_numeric(df["sentiment_cnn"], errors="coerce")
df["week"] = pd.to_numeric(df["week_since_release"], errors="coerce")

# Create a binary column for positive sentiment (threshold 0.5)
df["is_positive"] = df["sentiment_score"] >= 0.5

# Filter for weeks 1 to 6 only
df_filtered = df[df["week"].between(1, 6)]

# Group by movie and week, and compute positive ratio
summary = (
    df_filtered.groupby(["movie", "week"])
    .agg(
        total_reviews=("review_detail", "count"),
        positive_reviews=("is_positive", "sum")
    )
    .reset_index()
)
summary["positive_ratio"] = summary["positive_reviews"] / summary["total_reviews"]

# Pivot for clearer week-wise view
summary_pivot = summary.pivot(index="movie", columns="week", values="positive_ratio")

# Optional: rename columns to "Week X Positive %"
summary_pivot.columns = [f"Week {int(c)} Pos %" for c in summary_pivot.columns]

summary_pivot


# Load sentiment and review counts per movie-week
reviews_df = pd.read_excel("/Users/hotpotato/Desktop/Final Project/code_output/filtered_reviews_with_sentiment.xlsx")

# Load box office data per movie-week
box_office_df = pd.read_csv("/Users/hotpotato/Desktop/Final Project/code_output/the_numbers_box_office_weeks1-6.csv")

# Preprocess reviews_df: count total and positive reviews per movie-week
reviews_df['week'] = reviews_df['week_since_release'].astype(int)
reviews_df['is_positive'] = reviews_df['sentiment_cnn'] > 0.5

weekly_review_stats = reviews_df.groupby(['movie', 'week']).agg(
    review_count=('sentiment_cnn', 'count'),
    positive_count=('is_positive', 'sum')
).reset_index()

weekly_review_stats['positive_ratio'] = weekly_review_stats['positive_count'] / weekly_review_stats['review_count']

# Merge with box office data
merged_df = pd.merge(
    box_office_df,
    weekly_review_stats,
    how='left',
    left_on=['Movie', 'Week'],
    right_on=['movie', 'week']
)

# Clean up and display final dataset
final_df = merged_df[['Movie', 'Week', 'Gross', 'review_count', 'positive_count', 'positive_ratio']]
final_df = final_df.dropna(subset=['review_count']).reset_index(drop=True)
final_df



print(final_df.columns.tolist())

# Make sure week is a string
final_df['week'] = final_df['week'].astype(str)

# Set style
sns.set(style="whitegrid")

# Draw
plt.figure(figsize=(14, 6))
sns.lineplot(data=final_df, x="week", y="gross", hue="movie", legend=False)
plt.title("Weekly Box Office by Movie")

# Hide y-axis label scale
plt.gca().set_yticklabels([]) # Key statements
plt.ylabel("")

plt.xlabel("Week")
plt.tight_layout()
plt.show()



sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=final_df,
    x="positive_ratio",
    y="gross",
    hue="week",
    palette="viridis",
    alpha=0.8
)

plt.yscale("log")  # Optional: log scale makes lower-grossing movies easier to compare
plt.title("Sentiment vs Box Office")
plt.xlabel("Positive Review Ratio")
plt.ylabel("Weekly Gross ($)")
plt.tight_layout()
plt.show()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the clean final_df if not already
final_df['week'] = final_df['week'].astype(int)

# Drop rows with missing values in either review_count or gross
df_reg = final_df[['review_count', 'gross']].dropna()

# Prepare variables
X = df_reg['review_count']
y = df_reg['gross']

# Add constant to X for statsmodels
X_sm = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X_sm).fit()
print(model.summary())

# Plot regression line with scatterplot
plt.figure(figsize=(10, 6))
sns.regplot(x='review_count', y='gross', data=df_reg, scatter_kws={'alpha':0.6})
plt.title("Linear Regression: Review Count vs Weekly Gross")
plt.xlabel("Review Count (per week)")
plt.ylabel("Weekly Gross ($)")
plt.tight_layout()
plt.grid(True)
plt.show()





import statsmodels.api as sm

X = df_clean[['review_count', 'positive_ratio']]
X = sm.add_constant(X)
y = df_clean['gross']

model = sm.OLS(y, X).fit()
print(model.summary())




# Clean up empty values
df_clean = final_df.dropna(subset=['gross', 'review_count', 'positive_ratio'])

# Convert 'gross' to float by removing currency symbols and commas
df_clean['gross'] = df_clean['gross'].replace('[\$,]', '', regex=True).astype(float)

# Create the design matrix X and target y
X = df_clean[["review_count", "positive_ratio", "week"]]
X = sm.add_constant(X)
y = df_clean["gross"]

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Return the summary
print(model.summary())




import statsmodels.formula.api as smf

# Create week dummies
final_df['week'] = final_df['week'].astype(str)  # make sure week is categorical
model = smf.ols(formula="gross ~ review_count + positive_ratio + C(week)", data=final_df).fit()
print(model.summary())







