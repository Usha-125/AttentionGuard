import csv

rows = [
    ["Method", "Accuracy (%)", "Attention Drift MSE", "Robustness Rank"],
    ["FaceNet Baseline", 98.0, "-", 2],
    ["Attention CNN", 95.0, "5.8128e-05", 3],
    ["Proposed Consistency Model", 96.5, "5.6341e-05", 1],
]

for row in rows:
    print(row)

with open("outputs/results/final_results_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("\nSaved to outputs/results/final_results_table.csv")