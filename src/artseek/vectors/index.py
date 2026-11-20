"""In-memory float32 vector matrices backed by atomic .npy persistence."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

Channel = Literal["visual", "descriptor"]


def _load_npy_contiguous(path: Path) -> np.ndarray:
    """Load a .npy matrix via mmap and close the OS handle immediately (Windows-safe)."""
    mm = np.load(path, mmap_mode="r")
    try:
        arr = np.array(mm, dtype=np.float32, copy=True)
    finally:
        del mm
    if arr.ndim != 2:
        raise ValueError(f"向量文件维度异常：{path} shape={arr.shape}")
    return arr


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Write a float32 matrix atomically without NumPy appending .npy to .tmp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array, dtype=np.float32))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


class VectorIndex:
    """Dual-channel (visual + descriptor) vector matrix aligned 1:1 with SQLite images.id."""

    def __init__(
        self,
        visual_path: Path,
        descriptor_path: Path,
        *,
        active_rows: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        self.visual_path = Path(visual_path)
        self.descriptor_path = Path(descriptor_path)
        self._lock = threading.RLock()
        self._visual: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._descriptor: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._count: int = 0
        self._dimension: int = 0
        self._active: np.ndarray = np.zeros(0, dtype=bool)
        self._cuda_visual: Any = None
        self._cuda_descriptor: Any = None
        self._cuda_device: Any = None
        self._dirty: bool = False
        self.reload(active_rows=active_rows)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def active_count(self) -> int:
        with self._lock:
            return int(np.count_nonzero(self._active[: self._count]))

    @property
    def dimension(self) -> int:
        with self._lock:
            return self._dimension

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def reopen(
        self,
        visual_path: Path,
        descriptor_path: Path,
        *,
        active_rows: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        """Switch backing `.npy` file paths and reload into memory."""
        with self._lock:
            self.visual_path = Path(visual_path)
            self.descriptor_path = Path(descriptor_path)
            self.reload(active_rows=active_rows)

    def reload(self, *, active_rows: np.ndarray | Sequence[int] | None = None) -> None:
        with self._lock:
            self.release_cuda()
            if self.visual_path.is_file() and self.descriptor_path.is_file():
                vis = _load_npy_contiguous(self.visual_path)
                desc = _load_npy_contiguous(self.descriptor_path)
                if vis.shape != desc.shape:
                    raise ValueError(
                        f"双路向量矩阵形状不一致：visual={vis.shape} descriptor={desc.shape}"
                    )
                self._visual = vis
                self._descriptor = desc
                self._count = int(vis.shape[0])
                self._dimension = int(vis.shape[1]) if vis.ndim == 2 else 0
            else:
                self._visual = np.zeros((0, 0), dtype=np.float32)
                self._descriptor = np.zeros((0, 0), dtype=np.float32)
                self._count = 0
                self._dimension = 0
            self.sync_active_mask(active_rows)
            self._dirty = False

    def sync_active_mask(self, active_rows: np.ndarray | Sequence[int] | None) -> None:
        """Align the boolean active mask with live SQLite images.id rows."""
        with self._lock:
            cap = int(self._visual.shape[0]) if self._visual.ndim == 2 else 0
            if active_rows is None:
                if cap > 0:
                    norms = np.linalg.norm(self._visual[: self._count], axis=1)
                    mask = np.zeros(cap, dtype=bool)
                    mask[: self._count] = norms > 1e-8
                    self._active = mask
                else:
                    self._active = np.zeros(0, dtype=bool)
                return
            mask = np.zeros(cap, dtype=bool)
            idx = np.asarray(active_rows, dtype=np.int64)
            valid = idx[(idx >= 0) & (idx < self._count)]
            if valid.size:
                mask[valid] = True
            self._active = mask

    def _ensure_capacity(self, min_rows: int, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"向量维度必须为正数，得到 {dimension}")
        if self._dimension == 0:
            self._dimension = dimension
        elif self._dimension != dimension:
            raise ValueError(
                f"向量维度发生变化（索引 {self._dimension}，当前 {dimension}）。"
                "请执行 artseek rebuild <图片目录> 重建索引。"
            )
        current_cap = (
            int(self._visual.shape[0])
            if self._visual.ndim == 2 and self._visual.shape[1] == dimension
            else 0
        )
        if current_cap >= min_rows and self._visual.shape[1:] == (dimension,):
            return
        new_cap = max(min_rows, int(current_cap * 1.25) + 64)
        new_vis = np.zeros((new_cap, dimension), dtype=np.float32)
        new_desc = np.zeros((new_cap, dimension), dtype=np.float32)
        new_active = np.zeros(new_cap, dtype=bool)
        if self._count > 0 and self._visual.shape[1:] == (dimension,):
            new_vis[: self._count] = self._visual[: self._count]
            new_desc[: self._count] = self._descriptor[: self._count]
            new_active[: self._count] = self._active[: self._count]
        self._visual = new_vis
        self._descriptor = new_desc
        self._active = new_active

    def set_rows(
        self,
        row_ids: Sequence[int] | np.ndarray,
        visual: np.ndarray,
        descriptor: np.ndarray,
    ) -> None:
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size == 0:
            return
        vis = _l2_normalize_rows(visual)
        desc = _l2_normalize_rows(descriptor)
        if vis.shape[0] != ids.size or desc.shape != vis.shape:
            raise ValueError("写入向量行数与 row_ids 数量或形状不匹配")
        with self._lock:
            self.release_cuda()
            max_id = int(ids.max())
            self._ensure_capacity(max_id + 1, int(vis.shape[1]))
            self._visual[ids] = vis
            self._descriptor[ids] = desc
            self._active[ids] = True
            self._count = max(self._count, max_id + 1)
            self._dirty = True

    def set_descriptor_rows(
        self,
        row_ids: Sequence[int] | np.ndarray,
        descriptor: np.ndarray,
    ) -> None:
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size == 0:
            return
        desc = _l2_normalize_rows(descriptor)
        if desc.shape[0] != ids.size:
            raise ValueError("更新 descriptor 向量行数与 row_ids 不匹配")
        with self._lock:
            self.release_cuda()
            max_id = int(ids.max())
            self._ensure_capacity(max_id + 1, int(desc.shape[1]))
            self._descriptor[ids] = desc
            self._dirty = True

    def zero_rows(self, row_ids: Sequence[int] | np.ndarray) -> None:
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size == 0:
            return
        with self._lock:
            self.release_cuda()
            valid = ids[(ids >= 0) & (ids < self._count)]
            if valid.size == 0:
                return
            self._visual[valid] = 0.0
            self._descriptor[valid] = 0.0
            self._active[valid] = False
            self._dirty = True

    def clear(self) -> None:
        with self._lock:
            self.release_cuda()
            self._visual = np.zeros((0, 0), dtype=np.float32)
            self._descriptor = np.zeros((0, 0), dtype=np.float32)
            self._active = np.zeros(0, dtype=bool)
            self._count = 0
            self._dimension = 0
            self._dirty = False
            self.visual_path.unlink(missing_ok=True)
            self.descriptor_path.unlink(missing_ok=True)

    def flush(self) -> int:
        """Atomically persist active slice [0:count] to disk."""
        with self._lock:
            n = self._count
            dim = self._dimension
            if n == 0 or dim == 0:
                vis_out = np.zeros((0, dim), dtype=np.float32)
                desc_out = np.zeros((0, dim), dtype=np.float32)
            else:
                vis_out = self._visual[:n]
                desc_out = self._descriptor[:n]
            _save_npy_atomic(self.visual_path, vis_out)
            _save_npy_atomic(self.descriptor_path, desc_out)
            self._dirty = False
            return n

    def compact(self, live_row_ids_in_order: Sequence[int] | np.ndarray) -> int:
        """Pack active rows densely into [0..M-1] matching the caller's new 0-based IDs."""
        order = np.asarray(live_row_ids_in_order, dtype=np.int64)
        with self._lock:
            self.release_cuda()
            dim = self._dimension
            m = int(order.size)
            if m == 0 or dim == 0:
                self._visual = np.zeros((0, dim), dtype=np.float32)
                self._descriptor = np.zeros((0, dim), dtype=np.float32)
                self._active = np.zeros(0, dtype=bool)
                self._count = 0
            else:
                self._visual = np.ascontiguousarray(self._visual[order], dtype=np.float32)
                self._descriptor = np.ascontiguousarray(self._descriptor[order], dtype=np.float32)
                self._active = np.ones(m, dtype=bool)
                self._count = m
            self._dirty = True
            return self.flush()

    def upload_cuda(self, device: Any | None = None) -> bool:
        """Optionally mirror active matrices onto CUDA as float16 tensors."""
        import torch

        if not torch.cuda.is_available():
            return False
        with self._lock:
            if self._count == 0 or self._dimension == 0:
                return False
            target = device if device is not None else torch.device("cuda")
            self._cuda_visual = torch.from_numpy(self._visual[: self._count]).to(
                device=target, dtype=torch.float16
            )
            self._cuda_descriptor = torch.from_numpy(self._descriptor[: self._count]).to(
                device=target, dtype=torch.float16
            )
            self._cuda_device = target
            return True

    def release_cuda(self) -> None:
        with self._lock:
            had_cuda = self._cuda_visual is not None or self._cuda_descriptor is not None
            self._cuda_visual = None
            self._cuda_descriptor = None
            self._cuda_device = None
        if had_cuda:
            from ..compute import release_accelerator

            release_accelerator()

    def visual_vectors(self, row_ids: Sequence[int] | np.ndarray) -> dict[int, np.ndarray]:
        ids = np.asarray(row_ids, dtype=np.int64)
        with self._lock:
            out: dict[int, np.ndarray] = {}
            for row in ids.tolist():
                if 0 <= row < self._count and self._active[row]:
                    out[row] = self._visual[row].copy()
            return out

    def search(
        self,
        vector: Sequence[float] | np.ndarray,
        channel: Channel,
        k: int,
        rows: np.ndarray | Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(row_ids, cosine_scores)` descending for top-k active rows."""
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        k = max(1, int(k))

        with self._lock:
            n = self._count
            if n == 0 or self._dimension == 0 or q.shape[0] != self._dimension:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

            if rows is not None:
                cand = np.asarray(rows, dtype=np.int64)
                if cand.size == 0:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
                cand = cand[(cand >= 0) & (cand < n)]
                if cand.size == 0:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
                cand = cand[self._active[:n][cand]]
                if cand.size == 0:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            else:
                cand = None

            cuda_mat = self._cuda_visual if channel == "visual" else self._cuda_descriptor
            if cuda_mat is not None and (cand is None or cand.size >= 1024):
                import torch

                with torch.inference_mode():
                    qt = torch.from_numpy(q).to(device=self._cuda_device, dtype=torch.float16)
                    if cand is None:
                        scores_t = cuda_mat @ qt
                        active_t = torch.from_numpy(self._active[:n]).to(
                            device=self._cuda_device
                        )
                        scores_t = torch.where(active_t, scores_t, float("-inf"))
                        take = min(k, n)
                        vals, idx = torch.topk(scores_t, k=take, largest=True, sorted=True)
                        top_rows = idx.cpu().numpy().astype(np.int64)
                        top_scores = vals.float().cpu().numpy().astype(np.float32)
                    else:
                        cand_t = torch.from_numpy(cand).to(device=self._cuda_device)
                        scores_t = cuda_mat[cand_t] @ qt
                        take = min(k, int(cand.size))
                        vals, local_idx = torch.topk(scores_t, k=take, largest=True, sorted=True)
                        top_rows = cand[local_idx.cpu().numpy().astype(np.int64)]
                        top_scores = vals.float().cpu().numpy().astype(np.float32)
                finite = np.isfinite(top_scores)
                return top_rows[finite], top_scores[finite]

            mat = (self._visual if channel == "visual" else self._descriptor)[:n]
            if cand is None:
                scores = mat @ q
                inactive = ~self._active[:n]
                if np.any(inactive):
                    scores = scores.copy()
                    scores[inactive] = -np.inf
                total = n
                if k >= total:
                    order = np.argsort(-scores, kind="stable")
                    top_rows = order.astype(np.int64)
                    top_scores = scores[top_rows].astype(np.float32)
                else:
                    idx = np.argpartition(-scores, k - 1)[:k]
                    idx = idx[np.argsort(-scores[idx], kind="stable")]
                    top_rows = idx.astype(np.int64)
                    top_scores = scores[top_rows].astype(np.float32)
                finite = np.isfinite(top_scores)
                return top_rows[finite], top_scores[finite]

            scores = mat[cand] @ q
            total = int(cand.size)
            if k >= total:
                order = np.argsort(-scores, kind="stable")
            else:
                part = np.argpartition(-scores, k - 1)[:k]
                order = part[np.argsort(-scores[part], kind="stable")]
            return cand[order].astype(np.int64), scores[order].astype(np.float32)

    def neighbor_tag_votes(
        self,
        target_rows: Sequence[int] | np.ndarray,
        row_tags: dict[int, Sequence[str]],
        *,
        limit: int = 24,
        top_tags: int = 5,
        chunk_size: int = 128,
    ) -> dict[int, tuple[list[str], dict[str, float]]]:
        """Vectorized visual-neighbor tag voting directly on the in-memory matrix."""
        with self._lock:
            n = self._count
            if n <= 1 or self._dimension == 0:
                return {}
            active_idx = np.flatnonzero(self._active[:n])
            if active_idx.size <= 1:
                return {}
            valid_targets = [
                int(r) for r in target_rows if 0 <= int(r) < n and self._active[int(r)]
            ]
            if not valid_targets:
                return {}

            mat = self._visual[:n]
            active_mat = mat[active_idx] if active_idx.size < n else mat
            global_to_active: dict[int, int] | None = (
                {int(g): pos for pos, g in enumerate(active_idx.tolist())}
                if active_idx.size < n
                else None
            )
            k = min(max(1, int(limit)), int(active_idx.size) - 1)
            result: dict[int, tuple[list[str], dict[str, float]]] = {}
            for start in range(0, len(valid_targets), chunk_size):
                batch = valid_targets[start : start + chunk_size]
                query = mat[batch]
                scores = query @ active_mat.T
                for local, row in enumerate(batch):
                    self_pos = row if global_to_active is None else global_to_active[row]
                    scores[local, self_pos] = -np.inf
                    best = np.argpartition(-scores[local], k - 1)[:k]
                    votes: dict[str, float] = {}
                    for pos in best.tolist():
                        score = float(scores[local, pos])
                        if not np.isfinite(score):
                            continue
                        neighbor_row = int(active_idx[pos]) if active_idx.size < n else pos
                        for tag in row_tags.get(neighbor_row, ()):
                            votes[tag] = votes.get(tag, 0.0) + score
                    if not votes:
                        result[row] = ([], {})
                        continue
                    ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
                    top = ranked[:top_tags]
                    denom = max(ranked[0][1], 1e-6)
                    result[row] = (
                        [tag for tag, _ in top],
                        {tag: round(score / denom, 4) for tag, score in top},
                    )
            return result
