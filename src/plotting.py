import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(path):
    csv_file = f"{path}/metrics.csv"
    df = pd.read_csv(csv_file)
    plt.plot(df['epoch'], df['train_loss_epoch'])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f"{path}/train_loss.png")