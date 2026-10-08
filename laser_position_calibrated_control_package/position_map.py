#!/usr/bin/env python3
"""Piecewise-linear image-pixel <-> actuator-position calibration map."""

import json
import math
from pathlib import Path


class PositionMapError(RuntimeError):
    pass


def _barycentric(p, a, b, c):
    """Return barycentric weights for p in triangle abc, or None if degenerate."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    px, py = p

    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-9:
        return None

    wa = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
    wb = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
    wc = 1.0 - wa - wb
    return wa, wb, wc


def _inside(weights, tol=1e-7):
    return min(weights) >= -tol


class PositionMap:
    """A regular row/column calibration grid with triangular interpolation."""

    def __init__(self, rows, cols, samples, camera_width, camera_height):
        self.rows = int(rows)
        self.cols = int(cols)
        self.samples = list(samples)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)

        expected = self.rows * self.cols
        if len(self.samples) != expected:
            raise PositionMapError(
                f"Expected {expected} calibration samples, got {len(self.samples)}"
            )

        grid = {}
        for sample in self.samples:
            r = int(sample["row"])
            c = int(sample["col"])
            grid[(r, c)] = {
                "x": float(sample["x"]),
                "y": float(sample["y"]),
                "servo": float(sample["servo"]),
                "actuator": float(sample["actuator"]),
            }

        if len(grid) != expected:
            raise PositionMapError("Calibration samples do not contain a complete grid")

        self.grid = grid
        self.triangles = []

        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                p00 = grid[(r, c)]
                p01 = grid[(r, c + 1)]
                p10 = grid[(r + 1, c)]
                p11 = grid[(r + 1, c + 1)]

                # Two triangles per cell. Both pixel-space and position-space
                # interpolation use the same local triangle correspondence.
                self.triangles.append((p00, p01, p10))
                self.triangles.append((p11, p10, p01))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise PositionMapError(
                f"Calibration map not found: {path}. Run calibrate_position_map.py first."
            )

        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PositionMapError(f"Could not read {path}: {exc}") from exc

        return cls(
            rows=data["rows"],
            cols=data["cols"],
            samples=data["samples"],
            camera_width=data["camera_width"],
            camera_height=data["camera_height"],
        )

    def save(self, path):
        data = {
            "version": 1,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "rows": self.rows,
            "cols": self.cols,
            "samples": self.samples,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def _interp_triangle(p, triangle, value_key):
        a, b, c = triangle
        pa = (a["x"], a["y"])
        pb = (b["x"], b["y"])
        pc = (c["x"], c["y"])
        weights = _barycentric(p, pa, pb, pc)
        if weights is None:
            return None
        if not _inside(weights):
            return None
        wa, wb, wc = weights
        return (
            wa * a[value_key]
            + wb * b[value_key]
            + wc * c[value_key]
        )

    def pixel_to_position(self, x, y):
        """Convert image pixel -> (servo_position, actuator_position)."""
        p = (float(x), float(y))

        best = None
        best_distance = float("inf")

        for triangle in self.triangles:
            servo = self._interp_triangle(p, triangle, "servo")
            if servo is not None:
                actuator = self._interp_triangle(p, triangle, "actuator")
                if actuator is not None:
                    return float(servo), float(actuator), "GRID"

            # Outside the mesh: keep the requested point from causing an
            # uncontrolled extrapolation. Fall back to the nearest sample.
            for sample in triangle:
                dx = sample["x"] - p[0]
                dy = sample["y"] - p[1]
                d2 = dx * dx + dy * dy
                if d2 < best_distance:
                    best_distance = d2
                    best = sample

        if best is None:
            raise PositionMapError("Calibration map contains no usable samples")

        return float(best["servo"]), float(best["actuator"]), "EDGE"

    def positions_to_pixel(self, servo_position, actuator_position):
        """Approximate inverse lookup, useful for diagnostics/settling."""
        p = (float(servo_position), float(actuator_position))

        for triangle in self.triangles:
            a, b, c = triangle
            pa = (a["servo"], a["actuator"])
            pb = (b["servo"], b["actuator"])
            pc = (c["servo"], c["actuator"])
            weights = _barycentric(p, pa, pb, pc)
            if weights is None or not _inside(weights):
                continue

            wa, wb, wc = weights
            x = wa * a["x"] + wb * b["x"] + wc * c["x"]
            y = wa * a["y"] + wb * b["y"] + wc * c["y"]
            return float(x), float(y), "GRID"

        # Nearest calibrated operating point as a safe diagnostic fallback.
        best = None
        best_d2 = float("inf")
        for sample in self.samples:
            ds = sample["servo"] - p[0]
            da = sample["actuator"] - p[1]
            d2 = ds * ds + da * da
            if d2 < best_d2:
                best_d2 = d2
                best = sample

        if best is None:
            return None
        return float(best["x"]), float(best["y"]), "EDGE"
