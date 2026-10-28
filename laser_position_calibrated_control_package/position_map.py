#!/usr/bin/env python3
"""Position mapping for calibrated servo + linear-actuator aiming.

Supports two map formats:
- Legacy REGULAR_GRID maps, retaining the original grid interpolation and
  grid-coordinate homography/quadratic extrapolation.
- New SCATTERED maps, built from calibration samples collected at arbitrary
  positions by moving the A4 sheet around the same target plane. These use
  Delaunay triangle interpolation inside the convex hull of measured samples
  and a global quadratic pixel-to-motor model only outside that hull.

The scattered map is intended to let a small number of points per sheet
placement cover a much larger camera workspace. Its extrapolation remains a
model estimate; the goal is to spread calibration points so normal operation
stays inside the measured convex hull.
"""

import json
from pathlib import Path

import cv2
import numpy as np


class PositionMapError(RuntimeError):
    pass


def _barycentric(p, a, b, c):
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
    """Maps an image target pixel to absolute servo and actuator positions."""

    def __init__(self, rows, cols, samples, camera_width, camera_height):
        """Load/construct a legacy regular-grid map (kept for compatibility)."""
        self.map_type = "REGULAR_GRID"
        self.rows = int(rows)
        self.cols = int(cols)
        self.samples = list(samples)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.sample_count = len(self.samples)

        expected = self.rows * self.cols
        if len(self.samples) != expected:
            raise PositionMapError(
                f"Expected {expected} grid samples, got {len(self.samples)}"
            )

        grid = {}
        for sample in self.samples:
            r = int(sample["row"])
            c = int(sample["col"])
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise PositionMapError(f"Invalid grid index row={r}, col={c}")
            grid[(r, c)] = {
                "row": r, "col": c,
                "x": float(sample["x"]), "y": float(sample["y"]),
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
                self.triangles.append((p00, p01, p10))
                self.triangles.append((p11, p10, p01))

        image_points = np.asarray(
            [[s["x"], s["y"]] for s in self.samples], dtype=np.float64
        ).reshape(-1, 1, 2)
        grid_points = np.asarray(
            [[int(s["col"]), int(s["row"])] for s in self.samples],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        pixel_to_grid, _ = cv2.findHomography(image_points, grid_points, method=0)
        if pixel_to_grid is None or not np.all(np.isfinite(pixel_to_grid)):
            raise PositionMapError("Could not fit image-to-grid homography")
        try:
            grid_to_pixel = np.linalg.inv(pixel_to_grid)
        except np.linalg.LinAlgError as exc:
            raise PositionMapError("Image-to-grid homography is singular") from exc
        if not np.all(np.isfinite(grid_to_pixel)):
            raise PositionMapError("Image-to-grid homography is numerically unstable")
        self._pixel_to_grid_h = pixel_to_grid
        projected = cv2.perspectiveTransform(grid_points.astype(np.float64), grid_to_pixel)
        residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        self.homography_rmse_px = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))

        design = np.asarray(
            [self._grid_poly_features(s["col"], s["row"]) for s in self.samples],
            dtype=np.float64,
        )
        motors = np.asarray(
            [[s["servo"], s["actuator"]] for s in self.samples], dtype=np.float64
        )
        self._motor_coefficients, _, rank, _ = np.linalg.lstsq(design, motors, rcond=None)
        if rank < design.shape[1] or not np.all(np.isfinite(self._motor_coefficients)):
            raise PositionMapError("Could not fit global grid motor-position model")
        fitted = design @ self._motor_coefficients
        residuals = fitted - motors
        self.model_rmse_servo = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
        self.model_rmse_actuator = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
        self._scattered_triangles = []

    @classmethod
    def from_scattered_samples(cls, samples, camera_width, camera_height):
        """Build a map from arbitrary pixel/motor samples gathered at many sheet placements."""
        obj = cls.__new__(cls)
        obj.map_type = "SCATTERED"
        obj.rows = 0
        obj.cols = 0
        obj.samples = [
            {
                **({k: s[k] for k in ("id", "placement", "corner") if k in s}),
                "x": float(s["x"]),
                "y": float(s["y"]),
                "servo": float(s["servo"]),
                "actuator": float(s["actuator"]),
            }
            for s in samples
        ]
        obj.camera_width = int(camera_width)
        obj.camera_height = int(camera_height)
        obj.sample_count = len(obj.samples)
        obj.grid = {}
        obj.triangles = []
        obj._scattered_triangles = []
        obj.homography_rmse_px = float("nan")
        obj._pixel_to_grid_h = None

        if obj.camera_width < 2 or obj.camera_height < 2:
            raise PositionMapError("Camera dimensions are invalid")
        if obj.sample_count < 6:
            raise PositionMapError(
                f"At least 6 samples are required for a quadratic model; got {obj.sample_count}. "
                "Use at least 12, preferably 16-20 spread across the workspace."
            )

        pixel_coords = np.asarray([[s["x"], s["y"]] for s in obj.samples], dtype=np.float64)
        if not np.all(np.isfinite(pixel_coords)):
            raise PositionMapError("Calibration contains non-finite pixel coordinates")
        if np.any(pixel_coords[:, 0] < 0) or np.any(pixel_coords[:, 0] >= obj.camera_width):
            raise PositionMapError("A calibration sample has an X coordinate outside the camera frame")
        if np.any(pixel_coords[:, 1] < 0) or np.any(pixel_coords[:, 1] >= obj.camera_height):
            raise PositionMapError("A calibration sample has a Y coordinate outside the camera frame")

        # Fit direct pixel (x,y) -> [servo,actuator], not abstract A4 row/column.
        design = np.asarray([obj._pixel_poly_features(s["x"], s["y"]) for s in obj.samples])
        motors = np.asarray([[s["servo"], s["actuator"]] for s in obj.samples], dtype=np.float64)
        obj._motor_coefficients, _, rank, _ = np.linalg.lstsq(design, motors, rcond=None)
        if rank < 6 or not np.all(np.isfinite(obj._motor_coefficients)):
            raise PositionMapError(
                "Calibration points do not span a 2-D workspace well enough for a quadratic model. "
                "Spread the A4 placements across the camera view and record more corners."
            )
        fitted = design @ obj._motor_coefficients
        residuals = fitted - motors
        obj.model_rmse_servo = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
        obj.model_rmse_actuator = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))

        # Delaunay triangulation of the actual camera pixels gives a local map
        # inside the measured convex hull, even when all samples are irregularly placed.
        # Points need to be distinct for Subdiv2D. Reject near-duplicates rather than
        # silently making a degenerate triangle map.
        for i in range(len(pixel_coords)):
            for j in range(i):
                if float(np.linalg.norm(pixel_coords[i] - pixel_coords[j])) < 2.0:
                    raise PositionMapError(
                        f"Samples {j + 1} and {i + 1} are within 2 pixels of each other. "
                        "Choose distinct calibration points or remove a duplicate."
                    )

        try:
            subdiv = cv2.Subdiv2D((0, 0, obj.camera_width, obj.camera_height))
            for x, y in pixel_coords:
                subdiv.insert((float(x), float(y)))
            triangle_list = subdiv.getTriangleList()
        except cv2.error as exc:
            raise PositionMapError(f"Could not triangulate calibration samples: {exc}") from exc

        seen = set()
        for row in triangle_list:
            vertices = np.asarray(row, dtype=np.float64).reshape(3, 2)
            indices = []
            valid_triangle = True
            for vertex in vertices:
                d2 = np.sum((pixel_coords - vertex) ** 2, axis=1)
                index = int(np.argmin(d2))
                if float(d2[index]) > 2.25:  # Subdiv vertices should equal inserted samples.
                    valid_triangle = False
                    break
                indices.append(index)
            if not valid_triangle or len(set(indices)) != 3:
                continue
            key = tuple(sorted(indices))
            if key in seen:
                continue
            seen.add(key)
            tri = tuple(obj.samples[i] for i in indices)
            centroid = (
                float(np.mean([p["x"] for p in tri])),
                float(np.mean([p["y"] for p in tri])),
            )
            weights = _barycentric(
                centroid,
                (tri[0]["x"], tri[0]["y"]),
                (tri[1]["x"], tri[1]["y"]),
                (tri[2]["x"], tri[2]["y"]),
            )
            if weights is not None:
                obj._scattered_triangles.append(tri)
        if not obj._scattered_triangles:
            raise PositionMapError("Could not create usable interpolation triangles")
        return obj

    def _pixel_poly_features(self, x, y):
        # Normalize to roughly [-1,1] so the least-squares model is well-conditioned.
        u = (float(x) - (self.camera_width - 1) / 2.0) / max(1.0, (self.camera_width - 1) / 2.0)
        v = (float(y) - (self.camera_height - 1) / 2.0) / max(1.0, (self.camera_height - 1) / 2.0)
        return np.asarray([1.0, u, v, u * u, u * v, v * v], dtype=np.float64)

    def _grid_poly_features(self, col, row):
        u = (float(col) - (self.cols - 1) / 2.0) / max(1.0, (self.cols - 1) / 2.0)
        v = (float(row) - (self.rows - 1) / 2.0) / max(1.0, (self.rows - 1) / 2.0)
        return np.asarray([1.0, u, v, u * u, u * v, v * v], dtype=np.float64)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise PositionMapError(f"Calibration map not found: {path}. Run the calibration program first.")
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PositionMapError(f"Could not read {path}: {exc}") from exc
        if data.get("map_type") == "SCATTERED":
            obj = cls.from_scattered_samples(
                data["samples"], data["camera_width"], data["camera_height"]
            )
            return obj
        return cls(
            rows=data["rows"], cols=data["cols"], samples=data["samples"],
            camera_width=data["camera_width"], camera_height=data["camera_height"],
        )

    def save(self, path):
        if self.map_type == "SCATTERED":
            data = {
                "version": 2,
                "map_type": "SCATTERED",
                "camera_width": self.camera_width,
                "camera_height": self.camera_height,
                "samples": self.samples,
            }
        else:
            data = {
                "version": 1,
                "map_type": "REGULAR_GRID",
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
        weights = _barycentric(
            p,
            (a["x"], a["y"]),
            (b["x"], b["y"]),
            (c["x"], c["y"]),
        )
        if weights is None or not _inside(weights):
            return None
        wa, wb, wc = weights
        return wa * a[value_key] + wb * b[value_key] + wc * c[value_key]

    def pixel_to_position(self, x, y):
        p = (float(x), float(y))
        if not (0.0 <= p[0] < self.camera_width and 0.0 <= p[1] < self.camera_height):
            raise PositionMapError(f"Pixel ({p[0]:.1f},{p[1]:.1f}) is outside the camera frame")

        if self.map_type == "SCATTERED":
            for triangle in self._scattered_triangles:
                servo = self._interp_triangle(p, triangle, "servo")
                if servo is not None:
                    actuator = self._interp_triangle(p, triangle, "actuator")
                    if actuator is not None:
                        return float(servo), float(actuator), "INTERPOLATED"
            predicted = self._pixel_poly_features(p[0], p[1]) @ self._motor_coefficients
            if not np.all(np.isfinite(predicted)):
                raise PositionMapError("Global sparse position model produced invalid motor targets")
            return float(predicted[0]), float(predicted[1]), "EXTRAPOLATED"

        for triangle in self.triangles:
            servo = self._interp_triangle(p, triangle, "servo")
            if servo is not None:
                actuator = self._interp_triangle(p, triangle, "actuator")
                if actuator is not None:
                    return float(servo), float(actuator), "GRID"

        pixel = np.asarray([[[p[0], p[1]]]], dtype=np.float64)
        grid_point = cv2.perspectiveTransform(pixel, self._pixel_to_grid_h)[0, 0]
        if not np.all(np.isfinite(grid_point)):
            raise PositionMapError("Projective mapping produced an invalid grid coordinate")
        col, row = float(grid_point[0]), float(grid_point[1])
        predicted = self._grid_poly_features(col, row) @ self._motor_coefficients
        if not np.all(np.isfinite(predicted)):
            raise PositionMapError("Global position model produced invalid motor targets")
        return float(predicted[0]), float(predicted[1]), "EXTRAPOLATED"

    def positions_to_pixel(self, servo_position, actuator_position):
        p = (float(servo_position), float(actuator_position))
        all_triangles = self._scattered_triangles if self.map_type == "SCATTERED" else self.triangles
        for triangle in all_triangles:
            a, b, c = triangle
            weights = _barycentric(
                p,
                (a["servo"], a["actuator"]),
                (b["servo"], b["actuator"]),
                (c["servo"], c["actuator"]),
            )
            if weights is None or not _inside(weights):
                continue
            wa, wb, wc = weights
            return (
                wa * a["x"] + wb * b["x"] + wc * c["x"],
                wa * a["y"] + wb * b["y"] + wc * c["y"],
                "INTERPOLATED" if self.map_type == "SCATTERED" else "GRID",
            )

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
