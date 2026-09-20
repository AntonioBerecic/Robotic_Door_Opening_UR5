import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_rewards(npy_file, window_size=100):
    """
    Plots training rewards from a .npy file.
    Args:
        npy_file (str): Path to the .npy file.
        window_size (int): Size of the moving average window.
    """
    if not os.path.exists(npy_file):
        print(f"Error: File {npy_file} not found.")
        return

    try:
        rewards = np.load(npy_file)
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")
        return

    print(f"Loaded {len(rewards)} episodes from {npy_file}")

    plt.figure(figsize=(10, 5))

    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, label='Raw Rewards', color='blue')

    # Plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, label=f'Moving Avg ({window_size})', color='red', linewidth=2)

    plt.title(f'Training Rewards: {os.path.basename(npy_file)}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_png = npy_file.replace('.npy', '.png')
    plt.savefig(output_png)
    print(f"Graph saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training rewards from MuJoCo logs.')
    parser.add_argument('--file', type=str, help='Path to the .npy reward file', required=True)
    parser.add_argument('--window', type=int, default=100, help='Moving average window size')

    args = parser.parse_args()
    plot_rewards(args.file, args.window)
