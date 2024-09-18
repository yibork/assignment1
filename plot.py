import json
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_subplot(ax, x, y, title, xlabel, ylabel):
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

def create_combined_plot(x, y, title, xlabel, ylabel, legend, output_path):
    plt.figure(figsize=(12, 6))
    for key, values in y.items():
        plt.plot(x, values, label=key, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_and_save_training_data(data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    epochs = range(1, len(data['train_loss']) + 1)

    create_combined_plot(
        epochs,
        {
            'Training Loss': data['train_loss'],
            'Test Loss': data['test_loss'],
            'Training Error': data['train_error'],
            'Test Error': data['test_error']
        },
        'Training and Test Loss/Error Over Epochs',
        'Epochs',
        'Loss/Error',
        ['Training Loss', 'Test Loss', 'Training Error', 'Test Error'],
        os.path.join(output_folder, 'loss_error_plot.png')
    )

    # Plot performance metrics (separate subplots)
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Performance Metrics', fontsize=16)

    create_subplot(axs[0, 0], epochs, data['accuracy'], 'Accuracy', 'Epochs', 'Score')
    create_subplot(axs[0, 1], epochs, data['precision'], 'Precision', 'Epochs', 'Score')
    create_subplot(axs[1, 0], epochs, data['recall'], 'Recall', 'Epochs', 'Score')
    create_subplot(axs[1, 1], epochs, data['f1_score'], 'F1 Score', 'Epochs', 'Score')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'performance_metrics.png'), dpi=300)
    plt.close()

    print(f"Plots have been saved in the '{output_folder}' directory.")

if __name__ == "__main__":
    json_file_path = 'test/model_1_training_data.json'    
    output_folder = 'test/training_plots_1'
    # Load the data
    training_data = load_data(json_file_path)
    plot_and_save_training_data(training_data, output_folder)
