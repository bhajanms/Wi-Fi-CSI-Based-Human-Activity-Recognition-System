import os
import csv

INPUT_ROOT = r"C:\esp32-csi-tool\Activity"   # ← change this
OUTPUT_ROOT = r"C:\esp32-csi-tool\activity2"  # ← change this

def convert_file(txt_path, csv_path):

    rows = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 20:
                continue

            meta = parts[:5]
            csi = parts[5:]

            csi_string = "[" + " ".join(csi) + "]"

            rows.append({
                "timestamp": meta[0],
                "rssi": meta[2],
                "CSI_DATA": csi_string
            })

    if not rows:
        print("⚠️ empty:", txt_path)
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp","rssi","CSI_DATA"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("✅", csv_path)


# =========================
# WALK ALL FOLDERS
# =========================

for root, dirs, files in os.walk(INPUT_ROOT):

    for file in files:
        if not file.lower().endswith(".txt"):
            continue

        txt_full = os.path.join(root, file)

        rel = os.path.relpath(txt_full, INPUT_ROOT)
        csv_rel = rel.replace(".txt", ".csv")

        csv_full = os.path.join(OUTPUT_ROOT, csv_rel)

        convert_file(txt_full, csv_full)

print("\n🎯 ALL DONE")
