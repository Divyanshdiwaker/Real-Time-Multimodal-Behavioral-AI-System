from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(values, title):

    x = np.arange(len(values))
    values = np.array(values)

    plt.figure(figsize=(7, 3))

    # ---------------- MAIN LINE ----------------
    plt.plot(x, values, linewidth=2, marker='o', markersize=3)

    # ---------------- FILL AREA ----------------
    plt.fill_between(x, values, alpha=0.2)

    # ---------------- RISK ZONES ----------------
    # ---------------- RISK ZONES ----------------
    plt.axhspan(0, 0.3, alpha=0.15, color="green", label="Low")
    plt.axhspan(0.3, 0.6, alpha=0.15, color="orange", label="Medium")
    plt.axhspan(0.6, 1, alpha=0.15, color="red", label="High")
    plt.legend(loc="upper right", fontsize=7)

    # ---------------- PEAK DETECTION ----------------
    peaks = np.where(values > 0.75)[0]
    if len(peaks) > 0:
        plt.scatter(peaks, values[peaks], s=30)
        plt.text(peaks[0], values[peaks[0]],
                 "Peak", fontsize=8)

    # ---------------- DROP DETECTION (important for confidence) ----------------
    drops = np.where(values < 0.3)[0]
    if len(drops) > 0:
        plt.scatter(drops, values[drops], s=30)
        plt.text(drops[0], values[drops[0]],
                 "Drop", fontsize=8)

    # ---------------- TREND LINE ----------------
    if len(values) > 5:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), linestyle="--")

    # ---------------- STATS ANNOTATION ----------------
    mean_val = np.mean(values)
    std_val = np.std(values)

    plt.text(
        0.02 * len(values),
        0.9,
        f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
        fontsize=8
    )

    # ---------------- STYLE ----------------
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    buffer.seek(0)
    img_bytes = buffer.read()
    buffer.close()
    return BytesIO(img_bytes)