import os
import urllib.request

file_path = "dataset/the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

os.makedirs("dataset", exist_ok=True)

# Download and read
with urllib.request.urlopen(url) as response:
    raw_text = response.read().decode("utf-8")

# Save explicitly
with open(file_path, "w", encoding="utf-8") as f:
    f.write(raw_text)

print(raw_text[:500])