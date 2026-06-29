import os

path = r"c:\Users\motiv\OneDrive\Desktop\ann hack\templates\index.html"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Define mapping for CP1252/latin1 double-encoded characters to their UTF-8 equivalents
replacements = {
    "â€¢": "•",
    "â†’": "→",
    "â”€": "─",
    "Ã—": "×",
    "Â©": "©",
    "â†“": "↓"
}

for bad, good in replacements.items():
    text = text.replace(bad, good)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Encoding correction applied successfully.")
