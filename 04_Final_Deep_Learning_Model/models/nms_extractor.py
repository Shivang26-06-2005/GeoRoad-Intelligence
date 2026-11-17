import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class NMSExtractor:
    """
    Extracts 2D discrete vertex coordinates from road and intersection probability heatmaps.
    Operates using 2D Max-Pooling / Non-Maximum Suppression to find local confidence peaks.
    """
    def __init__(
        self,
        road_threshold: float = 0.5,
        inter_threshold: float = 0.4,
        window_size: int = 15,
        min_distance: int = 16,
    ):
        self.road_threshold = road_threshold
        self.inter_threshold = inter_threshold
        self.window_size = window_size
        self.min_distance = min_distance

    def _find_peaks(self, heatmap: torch.Tensor, threshold: float) -> List[Tuple[float, float, float]]:
        """
        heatmap: 2D tensor [H, W] with values in [0, 1]
        Returns list of (x, y, score)
        """
        H, W = heatmap.shape
        pad = self.window_size // 2

        # 2D Max pooling NMS
        h_tensor = heatmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        max_pooled = F.max_pool2d(h_tensor, kernel_size=self.window_size, stride=1, padding=pad)
        is_peak = (h_tensor == max_pooled) & (h_tensor > threshold)

        indices = torch.nonzero(is_peak.squeeze(0).squeeze(0), as_tuple=False)
        peaks = []
        for idx in indices:
            y, x = idx[0].item(), idx[1].item()
            score = heatmap[y, x].item()
            peaks.append((float(x), float(y), score))
        return peaks

    def extract_vertices(
        self,
        road_prob: torch.Tensor,
        inter_prob: torch.Tensor,
    ) -> np.ndarray:
        """
        road_prob: [H, W] float tensor
        inter_prob: [H, W] float tensor
        Returns:
            vertices: ndarray of shape [N, 2] containing (x, y) coordinates
        """
        inter_peaks = self._find_peaks(inter_prob, self.inter_threshold)
        road_peaks = self._find_peaks(road_prob, self.road_threshold)

        # Merge peaks: prioritize intersection peaks (higher boost)
        candidates = []
        for x, y, score in inter_peaks:
            candidates.append((x, y, score + 1.0))  # Intersection priority boost
        for x, y, score in road_peaks:
            candidates.append((x, y, score))

        # Sort candidate peaks by score descending
        candidates.sort(key=lambda item: item[2], reverse=True)

        selected_vertices = []
        min_dist_sq = self.min_distance ** 2

        for x, y, score in candidates:
            # Check minimum spatial suppression distance
            keep = True
            for sx, sy in selected_vertices:
                if (x - sx) ** 2 + (y - sy) ** 2 < min_dist_sq:
                    keep = False
                    break
            if keep:
                selected_vertices.append((x, y))

        if len(selected_vertices) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return np.array(selected_vertices, dtype=np.float32)

