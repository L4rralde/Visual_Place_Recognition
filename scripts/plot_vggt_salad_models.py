import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_model_comparison(models_data, save_path=None, min_area=0, max_area=2500, ylim=None):
    """
    Plots a bubble chart comparing AI models with a normalized bubble size scale.
    
    Args:
        models_data (list of dict): List containing model specifications.
            Expected keys: 'label', 'performance', 'inference_time', 'color', 'size'
        save_path (str, optional): File path to save the plot (e.g., 'plot.pdf').
        min_area (float): Minimum visual area for the smallest bubble.
        max_area (float): Maximum visual area for the largest bubble.
        ylim (tuple, optional): A tuple of (ymin, ymax) to set the y-axis limits.
    """
    
    # 1. Apply Seaborn professional styling
    sns.set_theme(style="ticks",palette="pastel", context="paper", font_scale=1.5)
    df = pd.DataFrame(models_data)
    
    # --- Normalize Bubble Sizes ---
    raw_min = 0
    raw_max = df['size'].max()
    
    if raw_max > raw_min:
        df['norm_size'] = ((df['size'] - raw_min) / (raw_max - raw_min)) * (max_area - min_area) + min_area
    else:
        df['norm_size'] = (max_area + min_area) / 2
    
    df['norm_size']  *= 2
    
    # 2. Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 3. Create the main bubble chart
    scatter = ax.scatter(
        x=df['inference_time'],
        y=df['performance'],
        s=df['norm_size'],
        c=df['color'],
        alpha=0.7,
        edgecolors='white',
        linewidth=1.5
    )
    
    # 4. Add model labels to each bubble
    for i, row in df.iterrows():
        ax.annotate(
            row['label'],
            (row['inference_time'], row['performance']),
            xytext=(0, 12),
            textcoords='offset points',
            ha='left',
            va='bottom',
            fontsize=16,
            fontweight='medium'
        )
        
    # --- Create Gray Circles for Size Scale Legend (Logarithmic 10s) ---
    
    # Find the maximum power of 10 that is less than or equal to raw_max
    if raw_max >= 1:
        max_power = int(np.log10(raw_max))
        scale_raw_sizes = [10**i for i in range(max_power + 1)]
    else:
        # Fallback in case the max size is less than 1
        scale_raw_sizes = [raw_max]

    
    # Calculate what their corresponding normalized visual areas would be
    if raw_max > raw_min:
        scale_norm_areas = [
            # max(0, ...) prevents matplotlib errors if the power of 10 is lower than raw_min
            max(0, ((s - raw_min) / (raw_max - raw_min)) * (max_area - min_area) + min_area) 
            for s in scale_raw_sizes
        ]
    else:
        scale_norm_areas = [(max_area + min_area) / 2] * len(scale_raw_sizes)

    for i in range(len(scale_norm_areas)):
        scale_norm_areas[i] *= 2
    print(scale_norm_areas)
    
    legend_handles = []
    for raw_size, norm_area in zip(scale_raw_sizes, scale_norm_areas):
        handle = ax.scatter(
            [], [], 
            s=norm_area, 
            c='gray',
            alpha=0.4,
            edgecolors='gray', 
            label=f"{raw_size:,} MB" # Added thousands separator for readability (e.g., 10,000)
        )
        legend_handles.append(handle)
        
    ax.legend(
        handles=legend_handles, 
        title="Model Size",
        loc="upper left",
        frameon=True,
        framealpha=0.9, 
        fontsize=14, 
        edgecolor='lightgray',
        labelspacing=1.8,
        borderpad=1.5,
        handletextpad=1.5
    )
    
    # 5. Customize axes, titles, and limits
    ax.set_xlabel("Extra inference Time (ms per batch of 16 images)", fontweight='bold', labelpad=10)
    ax.set_ylabel("Pitts250k-test R1", fontweight='bold', labelpad=10)
    ax.set_title("Time vs. Performance (Size = # Extra parameters)", fontweight='bold', pad=15)
    
    # Apply Y-Axis Limits
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # 6. Refine the grid and borders
    ax.grid(True, linestyle='--', alpha=0.5)
    sns.despine(trim=False, offset=5) 
    
    plt.tight_layout()
    
    # 7. Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")
    
    plt.show()


models = [
    {
        "label": "MegaLoc",
        "performance": 96.4,
        "inference_time": 65,
        "size": 228.6,
        "color": "#4C72B0",
    },
    {
        "label": "DINOv2+SALAD",
        "performance": 95.1,
        "inference_time": 60,
        "size": 88,
        "color": "#B04C9A",
    },
    {
        "label": "VGGT-S",
        "performance": 95.2,
        "inference_time": 10.5,
        "size": 1.8,
        "color": "#4CB0A8",
    },
    {
        "label": "VGGT-S+",
        "performance": 95.7,
        "inference_time": 36.2,
        "size": 27,
        "color": "#4CB0A8",
    },
    {
        "label": "VGGT-S++",
        "performance": 96.2,
        "inference_time": 67.7,
        "size": 52.2,
        "color": "#4CB0A8",
    },
    {
        "label": "VGGT-S+(LoRA)",
        "performance": 95.6,
        "inference_time": 42,
        "size":  1.90,
        "color": "#3c8c70"
    },
    {
        "label": "VGGT-S++(LoRA)",
        "performance": 95.8,
        "inference_time": 78.8,
        "size":  2.0,
        "color": "#3c8c70"
    }
]


if __name__ == '__main__':
    save_path = "model_comparison.pdf"
    plot_model_comparison(models, save_path, ylim=(95, 96.6))
