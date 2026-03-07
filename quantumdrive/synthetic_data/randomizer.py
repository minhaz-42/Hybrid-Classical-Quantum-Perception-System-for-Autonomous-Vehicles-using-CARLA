"""
QuantumDrive — Domain Randomizer
==================================
Applies domain randomization to CARLA/synthetic data for robust training.

Randomizes:
    - Weather conditions (rain, fog, sun position, clouds)
    - Lighting (time of day, shadows, exposure)
    - Camera parameters (FOV, position, angle, distortion)
    - Traffic density and behavior
    - Road surface textures
    - Noise levels (sensor noise, motion blur)
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    raise ImportError("Pillow required: pip install Pillow")


@dataclass
class RandomizationConfig:
    """Domain randomization configuration."""
    weather_range: bool = True
    lighting_range: bool = True
    camera_jitter: bool = True
    color_jitter: bool = True
    noise_injection: bool = True
    motion_blur: bool = True
    occlusion: bool = True
    cutout: bool = True
    seed: int = 42


class DomainRandomizer:
    """
    Applies domain randomization transforms to driving scene images.
    """

    def __init__(self, config: Optional[RandomizationConfig] = None):
        self.config = config or RandomizationConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def randomize(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply random domain transforms to an image.

        Returns:
            Transformed image and dict of applied transforms.
        """
        applied = {}

        if self.config.color_jitter and self.rng.random() < 0.7:
            image, params = self._color_jitter(image)
            applied["color_jitter"] = params

        if self.config.lighting_range and self.rng.random() < 0.5:
            image, params = self._lighting_variation(image)
            applied["lighting"] = params

        if self.config.noise_injection and self.rng.random() < 0.4:
            image, params = self._add_noise(image)
            applied["noise"] = params

        if self.config.motion_blur and self.rng.random() < 0.3:
            image, params = self._motion_blur(image)
            applied["motion_blur"] = params

        if self.config.weather_range and self.rng.random() < 0.3:
            image, params = self._weather_effect(image)
            applied["weather_effect"] = params

        if self.config.cutout and self.rng.random() < 0.3:
            image, params = self._random_cutout(image)
            applied["cutout"] = params

        if self.config.camera_jitter and self.rng.random() < 0.2:
            image, params = self._camera_jitter(image)
            applied["camera_jitter"] = params

        return image, applied

    def _color_jitter(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Random color adjustments."""
        brightness = self.rng.uniform(0.6, 1.4)
        contrast = self.rng.uniform(0.6, 1.4)
        saturation = self.rng.uniform(0.5, 1.5)
        hue_shift = self.rng.randint(-15, 15)

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)

        # Hue shift via numpy
        arr = np.array(image).astype(np.int16)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + hue_shift, 0, 255)
        image = Image.fromarray(arr.astype(np.uint8))

        return image, {
            "brightness": brightness, "contrast": contrast,
            "saturation": saturation, "hue_shift": hue_shift,
        }

    def _lighting_variation(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Simulate lighting changes (time of day, exposure)."""
        exposure = self.rng.uniform(0.5, 1.5)
        shadow_intensity = self.rng.uniform(0, 0.4)

        # Exposure
        arr = np.array(image).astype(np.float32)
        arr *= exposure
        arr = np.clip(arr, 0, 255)

        # Random shadow region
        if shadow_intensity > 0.1:
            H, W = arr.shape[:2]
            sx = self.rng.randint(0, W // 2)
            sy = self.rng.randint(0, H // 2)
            sw = self.rng.randint(W // 4, W)
            sh = self.rng.randint(H // 4, H)
            arr[sy:sy + sh, sx:sx + sw] *= (1 - shadow_intensity)
            arr = np.clip(arr, 0, 255)

        image = Image.fromarray(arr.astype(np.uint8))
        return image, {"exposure": exposure, "shadow_intensity": shadow_intensity}

    def _add_noise(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Add sensor noise (Gaussian, salt & pepper)."""
        noise_type = self.rng.choice(["gaussian", "salt_pepper", "speckle"])
        arr = np.array(image).astype(np.float32)

        if noise_type == "gaussian":
            sigma = self.rng.uniform(5, 25)
            noise = self.rng.normal(0, sigma, arr.shape)
            arr += noise
        elif noise_type == "salt_pepper":
            prob = self.rng.uniform(0.01, 0.05)
            salt = self.rng.random(arr.shape[:2]) < prob / 2
            pepper = self.rng.random(arr.shape[:2]) < prob / 2
            arr[salt] = 255
            arr[pepper] = 0
        else:  # speckle
            noise = self.rng.normal(0, 0.1, arr.shape)
            arr += arr * noise

        arr = np.clip(arr, 0, 255)
        image = Image.fromarray(arr.astype(np.uint8))
        return image, {"type": noise_type}

    def _motion_blur(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply motion blur effect."""
        kernel_size = self.rng.choice([3, 5, 7])
        image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))
        return image, {"kernel_size": kernel_size}

    def _weather_effect(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply weather overlay effects."""
        effect = self.rng.choice(["rain", "fog", "sun_glare"])
        arr = np.array(image).astype(np.float32)
        H, W = arr.shape[:2]

        if effect == "rain":
            num_drops = self.rng.randint(50, 300)
            for _ in range(num_drops):
                x = self.rng.randint(0, W)
                y = self.rng.randint(0, H)
                length = self.rng.randint(5, 20)
                y2 = min(H - 1, y + length)
                arr[y:y2, min(W - 1, x), :] = [200, 220, 255]

        elif effect == "fog":
            fog_density = self.rng.uniform(0.1, 0.4)
            fog = np.full_like(arr, 200)
            arr = arr * (1 - fog_density) + fog * fog_density

        elif effect == "sun_glare":
            cx = self.rng.randint(W // 4, 3 * W // 4)
            cy = self.rng.randint(0, H // 3)
            Y, X = np.ogrid[:H, :W]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            glare = np.exp(-dist / (W * 0.3)) * 150
            arr += glare[:, :, np.newaxis]

        arr = np.clip(arr, 0, 255)
        image = Image.fromarray(arr.astype(np.uint8))
        return image, {"effect": effect}

    def _random_cutout(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Random erasing / cutout augmentation."""
        arr = np.array(image)
        H, W = arr.shape[:2]

        num_patches = self.rng.randint(1, 4)
        patches = []
        for _ in range(num_patches):
            ph = self.rng.randint(H // 10, H // 4)
            pw = self.rng.randint(W // 10, W // 4)
            py = self.rng.randint(0, H - ph)
            px = self.rng.randint(0, W - pw)
            fill = self.rng.randint(0, 255, 3)
            arr[py:py + ph, px:px + pw] = fill
            patches.append({"x": int(px), "y": int(py), "w": int(pw), "h": int(ph)})

        image = Image.fromarray(arr)
        return image, {"patches": patches}

    def _camera_jitter(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Simulate camera position/angle jitter via affine transform."""
        W, H = image.size
        dx = self.rng.uniform(-10, 10)
        dy = self.rng.uniform(-5, 5)
        angle = self.rng.uniform(-3, 3)

        image = image.rotate(angle, translate=(dx, dy), fillcolor=(0, 0, 0))
        return image, {"dx": dx, "dy": dy, "angle": angle}
