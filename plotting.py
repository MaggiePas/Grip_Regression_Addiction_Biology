# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def correlation_plot(all_predictions, cohort, save_path=None):
    """
    Create a correlation plot between actual and predicted grip strength.
    
    Args:
    all_predictions (pd.DataFrame): DataFrame containing 'Actual', 'Predicted', and 'Diagnosis' columns
    save_path (str, optional): Path to save the plot. If None, the plot is displayed but not saved.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    if cohort == 'diseased':
        # Plot the controls on the background with high transparency
        ax = sns.regplot(data=all_predictions[all_predictions["Diagnosis"]==0], x="Actual", y="Predicted",color="tab:blue", scatter_kws={'alpha':0.25},line_kws={'alpha':0.25})
        # Plot for AUD/HIV subjects
        sns.regplot(data=all_predictions[all_predictions["Diagnosis"]>0], 
                x="Actual", y="Predicted", color="tab:orange", ax=ax)
    elif cohort == 'control':
        # Plot the controls on the background with high transparency
        ax = sns.regplot(data=all_predictions[all_predictions["Diagnosis"]>0], x="Actual", y="Predicted",color="tab:orange", scatter_kws={'alpha':0.25},line_kws={'alpha':0.25})
        # Plot for control subjects
        sns.regplot(data=all_predictions[all_predictions["Diagnosis"]==0], 
                x="Actual", y="Predicted", color="tab:blue", ax=ax)
    elif cohort == 'none':
        # Plot the controls on the background with high transparency
        ax = sns.regplot(data=all_predictions[all_predictions["Diagnosis"]>0], x="Actual", y="Predicted",color="tab:orange")
        # Plot for control subjects
        sns.regplot(data=all_predictions[all_predictions["Diagnosis"]==0], 
                x="Actual", y="Predicted", color="tab:blue", ax=ax)
    
    # Annotations for AUD/HIV
    audhiv = all_predictions[all_predictions["Diagnosis"]>0]
    stat, p = pearsonr(audhiv['Actual'], audhiv['Predicted'])
    annotate_correlation(ax, stat, p, (0.59, 0.056), color='tab:orange')
    
    # Annotations for controls
    controls = all_predictions[all_predictions["Diagnosis"]==0]
    stat, p = pearsonr(controls['Actual'], controls['Predicted'])
    annotate_correlation(ax, stat, p, (0.59, 0.13), color='tab:blue')
    
    # Set plot limits and labels
    ax.set(ylim=(1, 37), xlim=(0, 42))
    ax.set(xlabel='Measured Grip Strength', ylabel='Predicted Grip Strength')
    plt.yticks([10,20,30])
    
    # Adjust font sizes
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    sns.despine()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def annotate_correlation(ax, stat, p, position, color):
    """Helper function to annotate correlation statistics."""
    if p >= 0.001:
        ax.annotate(f'r = {stat:.2f}, p = {p:.3f}', xy=position, xycoords='axes fraction',
                    ha='left', va='center', fontsize=18, 
                    bbox={'boxstyle': 'square', 'fc': 'white', 'ec': 'white', 'alpha': 1.0},
                    color=color)
    else:
        ax.annotate(f'r = {stat:.2f}, p<0.001', xy=position, xycoords='axes fraction',
                    ha='left', va='center', fontsize=18, 
                    bbox={'boxstyle': 'square', 'fc': 'white', 'ec': 'white', 'alpha': 1.0},
                    color=color)