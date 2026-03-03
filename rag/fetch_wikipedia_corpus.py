import wikipedia
import os
import re

# Define output directory and file path
CORPUS_DIR = "../data/corpus"
# 1. Unify the file name to a master knowledge file
CORPUS_FILE = os.path.join(CORPUS_DIR, "master_knowledge.txt")

TARGET_PAGES = [
    "Culture of China", "Education in China", "Sport in China", "Chinese cuisine",
    "Culture of Iran", "Education in Iran", "Sport in Iran", "Iranian cuisine",
    "Culture of the United Kingdom", "Education in the United Kingdom", "Sport in the United Kingdom", "British cuisine",
    "Culture of the United States", "Education in the United States", "Sports in the United States", "American cuisine"
]

def clean_text(text):
    """
    Clean the extracted Wikipedia text.
    Removes citation brackets like [1], [23] and strips extra whitespace.
    """
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

def fetch_wikipedia_data():
    """
    Fetch content from the strictly defined Wikipedia pages, clean it,
    chunk it by paragraphs, and append it to the master corpus file.
    """
    os.makedirs(CORPUS_DIR, exist_ok=True)
    wikipedia.set_lang("en")

    total_chunks = 0

    # 2. CRITICAL CHANGE: Use "a" (append) instead of "w" (write/overwrite)
    with open(CORPUS_FILE, "a", encoding="utf-8") as f:
        print("\n--- Starting Wikipedia Data Fetching ---")
        for page_title in TARGET_PAGES:
            print(f"Fetching data for: {page_title}...")
            try:
                page = wikipedia.page(page_title, auto_suggest=False)
                content = page.content
                paragraphs = content.split('\n\n')

                for para in paragraphs:
                    cleaned_para = clean_text(para)
                    if len(cleaned_para) > 100 and "==" not in cleaned_para:
                        single_line_chunk = cleaned_para.replace('\n', ' ')
                        f.write(single_line_chunk + "\n")
                        total_chunks += 1

            except wikipedia.exceptions.DisambiguationError as e:
                print(f"⚠️ Disambiguation error for {page_title}. Skipping.")
            except wikipedia.exceptions.PageError:
                print(f"❌ Page not found: {page_title}. Skipping.")
            except Exception as e:
                print(f"❌ An error occurred while fetching {page_title}: {e}")

    print(f"\n✅ Wikipedia facts appended to: {CORPUS_FILE}")
    print(f"📊 Total Wikipedia chunks added: {total_chunks}")

if __name__ == "__main__":
    fetch_wikipedia_data()