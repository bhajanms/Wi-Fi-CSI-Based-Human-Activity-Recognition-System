import pandas as pd
import os
import re

INPUT_FOLDER = r"C:\esp32-csi-tool\Activity"     # ← change
OUTPUT_FOLDER = r"C:\esp32-csi-tool\csv_dataset2"    # ← change

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def smart_read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.readline()

    # detect separator
    if "," in sample:
        sep = ","
    elif "\t" in sample:
        sep = "\t"
    else:
        sep = r"\s+"   # whitespace regex

    return pd.read_csv(path, sep=sep, engine="python", header=None)


for file in os.listdir(INPUT_FOLDER):

    if not file.lower().endswith(".txt"):
        continue

    txt_path = os.path.join(INPUT_FOLDER, file)
    csv_name = file.replace(".txt", ".csv")
    csv_path = os.path.join(OUTPUT_FOLDER, csv_name)

    try:
        df = smart_read_txt(txt_path)
        df.to_csv(csv_path, index=False)
        print("✅ Converted:", file)

    except Exception as e:
        print("❌ Failed:", file, e)

print("\n🎯 Done — all txt files converted")
