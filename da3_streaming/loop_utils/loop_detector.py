# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import os
from pathlib import Path
import faiss
import torch
import numpy as np


class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""

    def __init__(self, image_dir, output="loop_closures.txt", config=None):
        """Initialize the loop detector

        Args:
            image_dir: Directory path containing images
            batch_size: Batch size for processing
            similarity_threshold: Similarity threshold for loop closure
            top_k: Number of nearest neighbors to check for each image
            use_nms: Whether to use Non-Maximum Suppression (NMS) filtering
            nms_threshold: NMS threshold for minimum frame difference between loop pairs
            output: Output file path
        """
        self.config = config
        self.image_dir = image_dir
        self.similarity_threshold = self.config["Loop"]["SALAD"]["similarity_threshold"]
        self.top_k = self.config["Loop"]["SALAD"]["top_k"]
        self.use_nms = self.config["Loop"]["SALAD"]["use_nms"]
        self.nms_threshold = self.config["Loop"]["SALAD"]["nms_threshold"]
        self.output = output

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_paths = None
        self.descriptors = None
        self.loop_closures = None

    def get_image_paths(self):
        """Get paths of all image files in directory"""
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))

        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        return image_paths

    def extract_descriptors(self):
        """Extract image feature descriptors"""
        if self.image_paths is None:
            self.get_image_paths()

        descriptors_dir = os.path.join(os.path.dirname(self.output), "descriptors")
        if not os.path.isdir(descriptors_dir):
            raise FileNotFoundError(descriptors_dir)

        descriptors = []

        for path in self.image_paths:
            fname = os.path.basename(path)
            desc_path = os.path.join(descriptors_dir, f"{fname}.npz")

            if not os.path.exists(desc_path):
                raise FileNotFoundError(desc_path)

            data = np.load(desc_path)
            descriptors.append(data["arr_0"])  # or named key if you used one

        descriptors = np.vstack(descriptors)
        descriptors = np.ascontiguousarray(descriptors)

        self.descriptors = torch.from_numpy(descriptors).float()
        return self.descriptors


    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue

            filtered_loops.append((idx1, idx2, sim))

            suppress_range = set()

            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))

            suppressed.update(suppress_range)

        return filtered_loops

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def find_loop_closures(self):
        """Find loop closures"""
        if self.descriptors is None:
            self.extract_descriptors()

        embed_size = self.descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)

        normalized_descriptors = self.descriptors.numpy()
        faiss_index.add(normalized_descriptors)

        similarities, indices = faiss_index.search(
            normalized_descriptors, self.top_k + 1
        )  # +1 because self is most similar

        loop_closures = []
        for i in range(len(self.descriptors)):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]

                if similarity > self.similarity_threshold and abs(i - neighbor_idx) > 10:
                    if i < neighbor_idx:
                        loop_closures.append((i, neighbor_idx, similarity))
                    else:
                        loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)

        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)

        self.loop_closures = self._ensure_decending_order(loop_closures)
        return self.loop_closures

    def save_results(self):
        """Save loop detection results to file"""
        if self.loop_closures is None:
            self.find_loop_closures()

        with open(self.output, "w") as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")

        print(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            print(f"NMS filtering applied, threshold: {self.nms_threshold}")

        if self.loop_closures:
            print("\nTop 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(self.loop_closures[:10]):
                print(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break

    def get_loop_list(self):
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        """Run complete loop detection pipeline"""

        self.get_image_paths()
        if not self.image_paths:
            print(f"No image files found in {self.image_dir}")
            return

        print(f"Found {len(self.image_paths)} image files")

        self.extract_descriptors()

        self.find_loop_closures()

        self.save_results()

        return self.loop_closures


def main():
    parser = argparse.ArgumentParser(description="Loop detection using SALAD model")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/media/deng/Data/KITTIdataset/data_odometry_color/dataset/sequences/00/image_2",
        help="Directory path containing images",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="./weights/dino_salad.ckpt", help="Model checkpoint path"
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=[336, 336],
        help="Image resize dimensions [height width]",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for loop closure",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of nearest neighbors to check for each image"
    )
    parser.add_argument("--output", type=str, default="loop_closures.txt", help="Output file path")
    parser.add_argument(
        "--use_nms",
        action="store_true",
        default=True,
        help="Whether to use Non-Maximum Suppression (NMS) filtering",
    )
    parser.add_argument(
        "--nms_threshold",
        type=int,
        default=25,
        help="NMS threshold for minimum frame difference between loop pairs",
    )

    args = parser.parse_args()

    detector = LoopDetector(
        image_dir=args.image_dir,
        ckpt_path=args.ckpt_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        use_nms=args.use_nms,
        nms_threshold=args.nms_threshold,
        output=args.output,
    )

    detector.run()


if __name__ == "__main__":
    main()
