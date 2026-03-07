"""
QuantumDrive — CARLA Synthetic Data Generator
===============================================
Generates synthetic driving data from CARLA simulator with domain randomization.
Falls back to procedural generation when CARLA is not available.

Captures:
    - RGB camera images
    - Semantic segmentation maps
    - Lane annotations
    - Traffic sign labels
    - Vehicle telemetry metadata

Usage:
    python -m synthetic_data.carla_generator --num_frames 5000 --output ./dataset
"""

import json
import importlib
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    raise ImportError("Pillow required: pip install Pillow")


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    """Configuration for data generation."""
    output_dir: str = "./dataset"
    num_frames: int = 1000
    image_width: int = 640
    image_height: int = 480
    camera_fov: float = 90.0
    fps: int = 10
    use_carla: bool = False
    carla_host: str = "localhost"
    carla_port: int = 2000
    seed: int = 42


# ─── Weather Presets ──────────────────────────────────────────────────────────

WEATHER_PRESETS = {
    "clear_noon": {"cloudiness": 10, "precipitation": 0, "sun_altitude": 70, "fog": 0},
    "cloudy": {"cloudiness": 70, "precipitation": 0, "sun_altitude": 50, "fog": 5},
    "rainy": {"cloudiness": 90, "precipitation": 60, "sun_altitude": 30, "fog": 10},
    "heavy_rain": {"cloudiness": 100, "precipitation": 90, "sun_altitude": 20, "fog": 20},
    "foggy": {"cloudiness": 50, "precipitation": 0, "sun_altitude": 40, "fog": 70},
    "sunset": {"cloudiness": 30, "precipitation": 0, "sun_altitude": 5, "fog": 5},
    "night": {"cloudiness": 20, "precipitation": 0, "sun_altitude": -30, "fog": 2},
    "wet_cloudy": {"cloudiness": 80, "precipitation": 30, "sun_altitude": 45, "fog": 15},
}


# ─── Traffic Sign Classes ────────────────────────────────────────────────────

TRAFFIC_SIGNS = [
    "speed_limit_30", "speed_limit_60", "speed_limit_90",
    "stop", "yield", "no_entry",
    "traffic_light_red", "traffic_light_green", "traffic_light_yellow",
    "pedestrian_crossing", "turn_left", "turn_right",
    "straight_ahead", "roundabout", "no_parking",
]


# ─── CARLA Generator ─────────────────────────────────────────────────────────

class CARLADataGenerator:
    """
    Generates synthetic driving data.
    Uses CARLA simulator when available, otherwise uses procedural generation.
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.output_dir = Path(config.output_dir)
        self.frame_count = 0
        self.metadata: List[Dict[str, Any]] = []

        # Try to connect to CARLA
        self.carla_client = None
        if config.use_carla:
            try:
                carla = importlib.import_module("carla")
                self.carla_client = carla.Client(config.carla_host, config.carla_port)
                self.carla_client.set_timeout(10.0)
                print(f"[QuantumDrive] Connected to CARLA at {config.carla_host}:{config.carla_port}")
            except Exception as e:
                print(f"[QuantumDrive] CARLA not available ({e}). Using procedural generation.")
                self.carla_client = None

    def setup_directories(self):
        """Create output directory structure."""
        for subdir in ["images", "segmentation", "lane_labels", "traffic_signs"]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def generate(self):
        """Generate the full dataset."""
        self.setup_directories()
        print(f"[QuantumDrive] Generating {self.config.num_frames} frames...")
        print(f"[QuantumDrive] Output: {self.output_dir}")

        start_time = time.time()

        for i in range(self.config.num_frames):
            if self.carla_client:
                frame_data = self._generate_carla_frame(i)
            else:
                frame_data = self._generate_procedural_frame(i)

            self._save_frame(i, frame_data)
            self.metadata.append(frame_data["metadata"])

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                print(f"  [{i+1}/{self.config.num_frames}] {fps:.1f} frames/sec")

        # Save metadata
        self._save_metadata()

        elapsed = time.time() - start_time
        print(f"[QuantumDrive] Generated {self.config.num_frames} frames in {elapsed:.1f}s")
        print(f"[QuantumDrive] Dataset saved to: {self.output_dir}")

    def _generate_procedural_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Generate a single frame procedurally (no CARLA needed)."""
        W, H = self.config.image_width, self.config.image_height

        # Select random weather
        weather_name = self.rng.choice(list(WEATHER_PRESETS.keys()))
        weather = WEATHER_PRESETS[weather_name]

        # Generate scene parameters
        lane_offset = self.rng.normal(0, 0.3)
        curvature = self.rng.uniform(-0.02, 0.02)
        speed = self.rng.uniform(0, 120)
        has_sign = self.rng.random() < 0.4
        sign_class = self.rng.choice(TRAFFIC_SIGNS) if has_sign else None

        # ─── Generate RGB image ───
        rgb_image = self._render_driving_scene(
            W, H, weather, lane_offset, curvature, sign_class
        )

        # ─── Generate segmentation map ───
        seg_image = self._render_segmentation(W, H, lane_offset, curvature, sign_class)

        # ─── Generate lane labels ───
        lane_points = self._generate_lane_points(W, H, lane_offset, curvature)

        # ─── Metadata ───
        metadata = {
            "frame_id": frame_idx,
            "filename": f"{frame_idx:06d}.png",
            "weather": weather_name,
            "weather_params": weather,
            "lane_offset": float(lane_offset),
            "curvature": float(curvature),
            "speed_kmh": float(speed),
            "traffic_sign": sign_class,
            "sign_class_id": TRAFFIC_SIGNS.index(sign_class) if sign_class else -1,
            "lane_points": lane_points,
            "timestamp": datetime.now().isoformat(),
            "class": TRAFFIC_SIGNS.index(sign_class) % 10 if sign_class else 0,
        }

        return {
            "rgb": rgb_image,
            "segmentation": seg_image,
            "metadata": metadata,
        }

    def _render_driving_scene(
        self, W: int, H: int, weather: dict, lane_offset: float,
        curvature: float, sign_class: Optional[str]
    ) -> Image.Image:
        """Render a synthetic driving scene."""
        img = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(img)

        sun_alt = weather["sun_altitude"]
        cloudiness = weather["cloudiness"]

        # Sky gradient
        sky_base = max(20, min(200, int(100 + sun_alt)))
        for y in range(H // 2):
            ratio = y / (H // 2)
            r = int(sky_base * 0.4 * (1 - ratio * 0.3))
            g = int(sky_base * 0.6 * (1 - ratio * 0.2))
            b = int(min(255, sky_base + 50) * (1 - ratio * 0.1))
            # Add cloudiness
            cloud_offset = int(cloudiness * 0.3 * self.rng.random())
            draw.line([(0, y), (W, y)], fill=(
                min(255, r + cloud_offset),
                min(255, g + cloud_offset),
                min(255, b + cloud_offset),
            ))

        # Ground / road
        road_color = (50 + int(sun_alt * 0.3), 50 + int(sun_alt * 0.3), 55 + int(sun_alt * 0.3))
        for y in range(H // 2, H):
            depth = (y - H // 2) / (H // 2)
            c = tuple(min(255, int(v * (0.4 + 0.6 * depth))) for v in road_color)
            draw.line([(0, y), (W, y)], fill=c)

        # Lane markings (perspective)
        cx = W // 2 + int(lane_offset * 100)
        for y in range(H // 2 + 10, H, 3):
            depth = (y - H // 2) / (H // 2)
            lane_w = int(200 * depth)
            curve_shift = int(curvature * (y - H // 2) ** 2 * 0.001)

            # Left lane
            lx = cx - lane_w // 2 + curve_shift
            draw.line([(lx, y), (lx + 3, y)], fill=(255, 255, 220), width=2)

            # Right lane
            rx = cx + lane_w // 2 + curve_shift
            draw.line([(rx, y), (rx + 3, y)], fill=(255, 255, 220), width=2)

            # Center dashes
            if y % 20 < 10:
                draw.line([(cx + curve_shift - 1, y), (cx + curve_shift + 1, y)],
                          fill=(255, 255, 180), width=1)

        # Traffic sign
        if sign_class:
            sign_x = self.rng.randint(W // 4, 3 * W // 4)
            sign_y = self.rng.randint(H // 4, H // 2)
            sign_size = self.rng.randint(20, 50)
            if "stop" in sign_class:
                draw.regular_polygon((sign_x, sign_y, sign_size), 8, fill=(200, 30, 30))
            elif "speed" in sign_class:
                draw.ellipse(
                    [sign_x - sign_size, sign_y - sign_size,
                     sign_x + sign_size, sign_y + sign_size],
                    fill=(255, 255, 255), outline=(200, 30, 30), width=3
                )
            elif "traffic_light" in sign_class:
                draw.rectangle(
                    [sign_x - 10, sign_y - 25, sign_x + 10, sign_y + 25],
                    fill=(30, 30, 30)
                )
                colors = {"red": (255, 0, 0), "green": (0, 255, 0), "yellow": (255, 255, 0)}
                for color_name, color in colors.items():
                    if color_name in sign_class:
                        draw.ellipse([sign_x - 6, sign_y - 8, sign_x + 6, sign_y + 8],
                                     fill=color)
            else:
                draw.polygon(
                    [(sign_x, sign_y - sign_size),
                     (sign_x - sign_size, sign_y + sign_size),
                     (sign_x + sign_size, sign_y + sign_size)],
                    fill=(255, 220, 50)
                )

        # Weather effects
        if weather["precipitation"] > 30:
            for _ in range(int(weather["precipitation"] * 2)):
                rx, ry = self.rng.randint(0, W), self.rng.randint(0, H)
                draw.line([(rx, ry), (rx + 1, ry + 8)], fill=(180, 200, 255, 100), width=1)

        if weather["fog"] > 30:
            fog_overlay = Image.new("RGB", (W, H), (200, 200, 210))
            img = Image.blend(img, fog_overlay, alpha=weather["fog"] / 200)

        # Add noise for realism
        noise = np.random.randint(-10, 10, (H, W, 3), dtype=np.int16)
        img_array = np.clip(np.array(img).astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        return img

    def _render_segmentation(
        self, W: int, H: int, lane_offset: float, curvature: float,
        sign_class: Optional[str]
    ) -> Image.Image:
        """Render semantic segmentation map."""
        # Segmentation classes: 0=sky, 1=road, 2=lane_marking, 3=sign, 4=vegetation
        seg = np.zeros((H, W), dtype=np.uint8)

        # Sky
        seg[:H // 2, :] = 0

        # Road
        cx = W // 2 + int(lane_offset * 100)
        for y in range(H // 2, H):
            depth = (y - H // 2) / (H // 2)
            lane_w = int(250 * depth)
            curve_shift = int(curvature * (y - H // 2) ** 2 * 0.001)
            left = max(0, cx - lane_w // 2 + curve_shift)
            right = min(W, cx + lane_w // 2 + curve_shift)
            seg[y, left:right] = 1

            # Lane markings
            lx = cx - lane_w // 2 + curve_shift
            rx = cx + lane_w // 2 + curve_shift
            for mx in [lx, rx, cx + curve_shift]:
                if 0 <= mx < W:
                    seg[y, max(0, mx - 2):min(W, mx + 2)] = 2

        # Vegetation on sides
        seg[:H // 2, :] = 0  # sky
        for y in range(H // 2, H):
            depth = (y - H // 2) / (H // 2)
            lane_w = int(250 * depth)
            curve_shift = int(curvature * (y - H // 2) ** 2 * 0.001)
            left_edge = max(0, cx - lane_w // 2 + curve_shift - 20)
            right_edge = min(W, cx + lane_w // 2 + curve_shift + 20)
            seg[y, :max(0, left_edge)] = 4
            seg[y, min(W, right_edge):] = 4

        # Colorize
        palette = {
            0: (70, 130, 180),   # sky
            1: (128, 64, 128),   # road
            2: (255, 255, 0),    # lane marking
            3: (220, 20, 60),    # sign
            4: (107, 142, 35),   # vegetation
        }
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_id, color in palette.items():
            rgb[seg == cls_id] = color

        return Image.fromarray(rgb)

    def _generate_lane_points(
        self, W: int, H: int, lane_offset: float, curvature: float
    ) -> List[List[float]]:
        """Generate lane boundary points."""
        cx = W / 2 + lane_offset * 100
        points = []
        for y_frac in np.linspace(0.5, 1.0, 20):
            y = int(y_frac * H)
            depth = (y - H / 2) / (H / 2)
            lane_w = 200 * depth
            curve_shift = curvature * (y - H / 2) ** 2 * 0.001
            points.append([
                float(cx - lane_w / 2 + curve_shift) / W,
                float(y_frac),
                float(cx + lane_w / 2 + curve_shift) / W,
                float(y_frac),
            ])
        return points

    def _generate_carla_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Generate frame from CARLA (stub for when CARLA is connected)."""
        # This would use the CARLA Python API to capture actual frames
        # For now, fall back to procedural
        return self._generate_procedural_frame(frame_idx)

    def _save_frame(self, frame_idx: int, frame_data: Dict):
        """Save a single frame's data to disk."""
        filename = f"{frame_idx:06d}.png"

        # RGB image
        frame_data["rgb"].save(self.output_dir / "images" / filename)

        # Segmentation
        frame_data["segmentation"].save(self.output_dir / "segmentation" / filename)

        # Lane labels
        lane_path = self.output_dir / "lane_labels" / f"{frame_idx:06d}.json"
        with open(lane_path, "w") as f:
            json.dump(frame_data["metadata"]["lane_points"], f)

        # Traffic sign crop (if present)
        if frame_data["metadata"]["traffic_sign"]:
            sign_data = {
                "class": frame_data["metadata"]["traffic_sign"],
                "class_id": frame_data["metadata"]["sign_class_id"],
            }
            sign_path = self.output_dir / "traffic_signs" / f"{frame_idx:06d}.json"
            with open(sign_path, "w") as f:
                json.dump(sign_data, f)

    def _save_metadata(self):
        """Save full dataset metadata."""
        metadata = {
            "dataset_info": {
                "name": "QuantumDrive Synthetic Driving Dataset",
                "version": "1.0.0",
                "num_frames": len(self.metadata),
                "image_size": [self.config.image_width, self.config.image_height],
                "generated_at": datetime.now().isoformat(),
                "generator": "QuantumDrive CARLADataGenerator",
            },
            "classes": TRAFFIC_SIGNS,
            "weather_presets": list(WEATHER_PRESETS.keys()),
            "frames": self.metadata,
        }

        # Main metadata
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Labels.json for PyTorch dataset compatibility
        labels = {}
        for frame in self.metadata:
            labels[frame["filename"]] = {
                "class": frame.get("class", 0),
                "lane_offset": frame["lane_offset"],
            }
        with open(self.output_dir / "labels.json", "w") as f:
            json.dump(labels, f, indent=2)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="QuantumDrive Synthetic Data Generator")
    parser.add_argument("--num_frames", type=int, default=1000)
    parser.add_argument("--output", type=str, default="./dataset")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--use_carla", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = GeneratorConfig(
        output_dir=args.output,
        num_frames=args.num_frames,
        image_width=args.width,
        image_height=args.height,
        use_carla=args.use_carla,
        seed=args.seed,
    )
    generator = CARLADataGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
