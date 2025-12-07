import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column, title=None):
    """
    Plots a histogram for a numerical column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    if title:
        plt.title(title)
    plt.show()
