import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import SimpleITK as sitk


def get_array(path):
    """Safely load arrays from .npz, images, or .nii.gz formats."""
    if not os.path.exists(path):
        return None
    try:
        if path.endswith('.npz'):
            data = np.load(path)
            return data[data.files[0]]
        elif path.endswith(('.png', '.jpg', '.jpeg')):
            if 'imgs' in path or 'images' in path:
                return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            else:
                return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif path.endswith(('.nii.gz', '.nii')):
            return sitk.GetArrayFromImage(sitk.ReadImage(path))
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None


def plot_comparisons(file_list, category_name, args, metrics_dict=None):
    """Generate and save the visual comparison plots."""
    if not file_list:
        print(f"No files found for {category_name}.")
        return

    # Random sampling if no metrics are provided
    if metrics_dict is None and args.num_samples and len(file_list) > args.num_samples:
        print(f"Sampling {args.num_samples} files out of {len(file_list)}...")
        file_list = random.sample(file_list, args.num_samples)

    fig, axes = plt.subplots(len(file_list), 4, figsize=(22, 5 * len(file_list)))

    if len(file_list) == 1:
        axes = [axes]

    tp_patch = mpatches.Patch(color='yellow', label='Match (True Positive)')
    fn_patch = mpatches.Patch(color='lime', label='Missed (False Negative)')
    fp_patch = mpatches.Patch(color='red', label='Over-predicted (False Positive)')
    fig.legend(handles=[tp_patch, fn_patch, fp_patch], loc='upper center', ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, 0.98))

    title_mode = args.mode.capitalize() if args.mode != 'random' else 'Random'
    fig.suptitle(f"{args.dataset} - {category_name} Modalities - {title_mode} Results Analysis", fontsize=20,
                 fontweight='bold', y=1.02)

    for i, filename in enumerate(file_list):
        img_path = os.path.join(args.data_root, args.dataset, 'MedSAM_test', args.subset, 'imgs', filename)
        gt_path = os.path.join(args.data_root, args.dataset, 'MedSAM_test', args.subset, 'gts', filename)
        pred_path = os.path.join(args.data_root, args.dataset, 'MedSAM_test', args.subset, args.pred_folder, filename)

        img = get_array(img_path)
        gt = get_array(gt_path)
        pred = get_array(pred_path)

        if img is None or gt is None or pred is None:
            print(f"Skipping {filename}: Missing modalities.")
            continue

        # Handle 3D slicing
        if img.ndim == 3 and img.shape[-1] != 3:
            mid = img.shape[0] // 2
            img, gt, pred = img[mid], gt[mid], pred[mid]

        gt_bin = (gt > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)

        diff_map = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
        diff_map[(gt_bin == 1) & (pred_bin == 1)] = [255, 255, 0]
        diff_map[(gt_bin == 1) & (pred_bin == 0)] = [0, 255, 0]
        diff_map[(gt_bin == 0) & (pred_bin == 1)] = [255, 0, 0]

        clean_title = filename.replace('2DBox_', '').replace('3DBox_', '').replace('_demo.npz', '').replace('.nii.gz',
                                                                                                            '').replace(
            '.npz', '')

        row_axes = axes[i]

        row_axes[0].imshow(img, cmap='gray')
        row_axes[0].set_title(f"{clean_title}\nOriginal Image", fontsize=14)

        row_axes[1].imshow(img, cmap='gray')
        row_axes[1].imshow(np.ma.masked_where(gt_bin == 0, gt_bin), cmap='autumn', alpha=0.6)
        row_axes[1].set_title("Ground Truth", fontsize=14)

        # Handle metrics formatting if present
        if metrics_dict and filename in metrics_dict:
            scores = metrics_dict[filename]
            score_text = f"DSC: {scores['dsc']:.4f} | NSD: {scores['nsd']:.4f}"
            title_color = 'darkred' if scores['dsc'] < 0.5 else 'black'
            pred_title = f"Model Prediction\n{score_text}"
        else:
            title_color = 'black'
            pred_title = "Model Prediction"

        row_axes[2].imshow(img, cmap='gray')
        row_axes[2].imshow(np.ma.masked_where(pred_bin == 0, pred_bin), cmap='cool', alpha=0.6)
        row_axes[2].set_title(pred_title, fontsize=14, color=title_color)

        row_axes[3].imshow(img, cmap='gray')
        diff_mask = np.any(diff_map != [0, 0, 0], axis=-1)
        row_axes[3].imshow(np.ma.masked_where(~diff_mask, diff_map[..., 0]), cmap='gray')
        row_axes[3].imshow(diff_map, alpha=0.5)
        row_axes[3].set_title("Difference Map", fontsize=14)

        for ax in row_axes:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    os.makedirs(args.out_dir, exist_ok=True)
    final_out_path = os.path.join(args.out_dir, args.out_file)
    plt.savefig(final_out_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to: {final_out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MedSAM Evaluation Visualizer")
    parser.add_argument('--data_root', type=str, required=True, help="Base path to data folder")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (e.g., dental_caries_segmentation_processed)")
    parser.add_argument('--subset', type=str, required=True, help="Subset folder (e.g., RGB_Teeth)")
    parser.add_argument('--pred_folder', type=str, required=True, help="Predictions folder name (e.g., segs_og_ckpt)")
    parser.add_argument('--out_dir', type=str, required=True, help="Output directory for plots")
    parser.add_argument('--out_file', type=str, required=True, help="Output filename (e.g., plot.png)")
    parser.add_argument('--mode', type=str, choices=['random', 'best', 'worst'], required=True, help="Selection mode")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to visualize")
    parser.add_argument('--csv_name', type=str, default='', help="Metrics CSV filename (Required for best/worst modes)")

    args = parser.parse_args()

    pred_dir = os.path.join(args.data_root, args.dataset, 'MedSAM_test', args.subset, args.pred_folder)
    if not os.path.exists(pred_dir):
        print(f"Prediction directory not found: {pred_dir}")
        return

    target_files = []
    metrics_map = None

    if args.mode == 'random':
        target_files = sorted(os.listdir(pred_dir))
    else:
        if not args.csv_name:
            print("Error: --csv_name is required when mode is 'best' or 'worst'.")
            return

        csv_path = os.path.join(args.data_root, args.dataset, 'MedSAM_test', args.subset, args.csv_name)
        if not os.path.exists(csv_path):
            print(f"Metrics CSV not found at: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        ascending = True if args.mode == 'worst' else False
        df_sorted = df.sort_values(by='dsc', ascending=ascending)
        selected_cases = df_sorted.head(args.num_samples)

        available_preds = os.listdir(pred_dir)
        metrics_map = {}

        for _, row in selected_cases.iterrows():
            case_id = str(row['case'])
            matching_files = [f for f in available_preds if case_id in f]

            if matching_files:
                target_file = matching_files[0]
                target_files.append(target_file)
                metrics_map[target_file] = {'dsc': row['dsc'], 'nsd': row['nsd']}
            else:
                print(f"Warning: Could not find prediction file for case {case_id}")

    # Categorize files
    files_2d = [f for f in target_files if
                '2D' in f or (f.endswith(('.png', '.jpg', '.jpeg', '.npz')) and '3D' not in f)]
    files_3d = [f for f in target_files if '3D' in f or f.endswith('.nii.gz')]

    if files_2d:
        print(f"Rendering {args.mode.capitalize()} 2D Modalities...")
        plot_comparisons(files_2d, "2D", args, metrics_map)

    if files_3d:
        print(f"Rendering {args.mode.capitalize()} 3D Modalities...")
        plot_comparisons(files_3d, "3D", args, metrics_map)


if __name__ == "__main__":
    main()