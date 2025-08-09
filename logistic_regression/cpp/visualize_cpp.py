import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("results.csv")
loss_df = pd.read_csv("loss_history.csv")
boundary_df = pd.read_csv("boundary.csv")

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('C++ Logistic Regression Analysis (Corrected)', fontsize=16)

axes[0].plot(loss_df['loss'], color='dodgerblue', linewidth=2)
axes[0].set_title('BCE Loss over Epochs', fontsize=14)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Binary Cross-Entropy Loss')

class_0 = results_df[results_df['y_true'] == 0]
class_1 = results_df[results_df['y_true'] == 1]
axes[1].scatter(class_0['X'], class_0['y_true'], alpha=0.7, edgecolors='k', label='Class 0 (True Value)')
axes[1].scatter(class_1['X'], class_1['y_true'], alpha=0.7, edgecolors='k', label='Class 1 (True Value)')

axes[1].plot(boundary_df['x_boundary'], boundary_df['y_boundary'], color='crimson', linewidth=3, label='Model Prediction (S-Curve)')

axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary (Threshold = 0.5)')

axes[1].set_title('Logistic Regression Fit', fontsize=14)
axes[1].set_xlabel('X (Feature)')
axes[1].set_ylabel('Probability (y_pred)')
axes[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("cpp_results_plot.png", dpi=150)
plt.show()
print("Plot saved to cpp_results_plot.png")