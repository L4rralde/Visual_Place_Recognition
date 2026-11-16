import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from adjustText import adjust_text


def plot_model_comparison(models_data, save_path=None, max_area=2500, ylim=None, xlim=None):
    """
    Plots a bubble chart comparing AI models with strictly proportional bubble sizes.
    
    Args:
        models_data (list of dict): List containing model specifications.
        save_path (str, optional): File path to save the plot.
        max_area (float): Visual area for the absolute largest bubble.
        ylim (tuple, optional): A tuple of (ymin, ymax) to set the y-axis limits.
        xlim (tuple, optional): A tuple of (xmin, xmax) to set the x-axis limits.
    """
    
    sns.set_theme(style="ticks", palette="bright", context="paper", font_scale=1.8)
    df = pd.DataFrame(models_data)
    
    # --- Strict Proportional Bubble Sizing ---
    # We find a single scaling factor so area is mathematically proportional to size.
    raw_max = df['size'].max()
    scale_factor = max_area / raw_max if raw_max > 0 else 1
    
    df['norm_size'] = df['size'] * scale_factor
    
    # Set up the figure (made slightly wider to accommodate external legend)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    
    # Create the main bubble chart
    scatter = ax.scatter(
        x=df['inference_time'],
        y=df['performance'],
        s=df['norm_size'],
        c=df['color'],
        alpha=0.75,
        edgecolors='white',
        linewidth=2,
        zorder=3
    )
    
    # Add smart model labels using adjustText
    texts = []
    for i, row in df.iterrows():
        texts.append(ax.text(
            row['inference_time'], 
            row['performance'], 
            row['label'], 
            fontsize=16, 
            fontweight='bold',
            color='#333333'
        ))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.8), ax=ax)
        
    # --- Create Legend with Powers of 10 ---
    # Find the maximum power of 10 that fits in our max size
    if raw_max >= 1:
        max_power = int(np.floor(np.log10(raw_max)))
        scale_raw_sizes = [10**i for i in range(max_power + 1)] # e.g., 1, 10, 100
    else:
        scale_raw_sizes = [raw_max]

    # Calculate proportional areas for the legend
    scale_norm_areas = [s * scale_factor for s in scale_raw_sizes]

    legend_handles = []
    for raw_size, norm_area in zip(scale_raw_sizes, scale_norm_areas):
        handle = ax.scatter(
            [], [], 
            s=norm_area, 
            c='gray',
            alpha=0.3,
            edgecolors='gray',
            linewidth=1.5,
            label=f"{raw_size:,} MB"
        )
        legend_handles.append(handle)
        
    # Add Legend OUTSIDE the plot to prevent occlusion
    ax.legend(
        handles=legend_handles, 
        title="Size",
        loc="upper left",
        #bbox_to_anchor=(0.05, 0.93),
        frameon=True,
        framealpha=0.95, 
        fontsize=13, 
        title_fontsize=18,
        edgecolor='silver',
        labelspacing=2.0,
        borderpad=2.2,
        handletextpad=2.0
    )
    
    # Customize axes, titles, and limits
    ax.set_xlabel("Extra Inference Time (ms per batch of 16 images)", fontweight='bold', labelpad=12)
    ax.set_ylabel("VPR Performance (Recall@1)", fontweight='bold', labelpad=12)
    ax.set_title("Time vs. Performance (size = #Extra parameters)", fontweight='black', pad=20, fontsize=18)
    
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_xlim(xlim)
    
    # Refine the grid and borders
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0, color='slategray')
    sns.despine(trim=False, offset=10) 
    
    # Tight layout rect helps ensure the external legend doesn't get cut off when saved
    plt.tight_layout()
    
    if save_path:
        # bbox_inches='tight' is crucial here to save the external legend
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")



models = [
    {
        "label": "MegaLoc",
        "performance": 89.2,
        "inference_time": 65,
        "size": 228.6,
        "color": "#F1795B",
    },
    {
        "label": "DINOv2 SALAD",
        "performance": 86.1,
        "inference_time": 60,
        "size": 88,
        "color": "#F1795B",
    },
    {
        "label": "BoQ",
        "performance": 88.2,
        "inference_time": 63,
        "size": 95.2,
        "color": "#F1795B",
    },
    #{
    #    "label": "MixVPR",
    #    "performance": 80.8,
    #    "inference_time": 20,
    #    "size": 10.9,
    #    "color": "#B9CBE9",
    #},
    {
        "label": "EDTformer",
        "performance": 88.8,
        "inference_time": 68,
        "size": 97,
        "color": "#F1795B",
    },
    {
        "label": "ImAge",
        "performance": 86.4,
        "inference_time": 58,
        "size": 86.6,
        "color": "#F1795B",
    },
    {
        "label": "VeGGieT SALAD+",
        "performance": 88.4,
        "inference_time": 36,
        "size": 10.2,
        "color": "#2DDD77",
    },
    {
        "label": "VeGGieT SALAD-$\Omega$",
        "performance": 87.5,
        "inference_time": 36,
        "size": 10.2,
        "color": "#2DDD77",
    },
]

if __name__ == '__main__':
    plot_model_comparison(
        models_data=models, 
        save_path="model_comparison.pdf", 
        max_area=10000,     
        xlim=(5, 75),     
        ylim=(85.9, 89.5)    
    )