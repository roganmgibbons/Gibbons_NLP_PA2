import matplotlib.pyplot as plt
from collections import Counter

# load predictions from predict.py
with open("predictions.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# extract labels (the number at the end of each line)
labels = [int(line.strip().split("\t")[-1]) for line in lines]

# count how many 0's and 1's or logits
counts = Counter(labels)
print("Prediction distribution:")
for label, count in counts.items():
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"{sentiment}: {count}")

# check - ratio
total = sum(counts.values())
pos_ratio = counts[1] / total
neg_ratio = counts[0] / total
print(f"\nPositive ratio: {pos_ratio:.2%}")
print(f"Negative ratio: {neg_ratio:.2%}")

# plot using matplotlib to visualize the data better
plt.bar(["Negative (0)", "Positive (1)"], [counts[0], counts[1]], color=["red", "green"])
plt.title("Sentiment Distribution in Unlabeled Data Predictions")
plt.ylabel("Number of Reviews")
plt.xlabel("Predicted Sentiment")
plt.show()
