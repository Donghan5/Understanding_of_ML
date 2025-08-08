import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("c_results.csv")
loss_df = pd.read_csv("c_loss_history.csv")

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('C Linear Regression Analysis', fontsize=16)

axes[0].plot(loss_df['loss'], color='dodgerblue', linewidth=2)
axes[0].set_title('Model Loss over Epochs', fontsize=14)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Mean Squared Error')

axes[1].scatter(results_df['X'], results_df['y_true'], alpha=0.6, edgecolors='k', label='Data Points')
axes[1].plot(results_df['X'], results_df['y_pred'], color='crimson', linewidth=3, label='Regression Line')
axes[1].set_title('Regression Fit on Data', fontsize=14)
axes[1].set_xlabel('X (Feature)')
axes[1].set_ylabel('y (Target)')
axes[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("c_results_plot.png", dpi=150)
plt.show()
print("Plot saved to c_results_plot.png")