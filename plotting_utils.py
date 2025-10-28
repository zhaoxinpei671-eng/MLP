"""Utility functions for saving training curves without blocking notebook execution."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

try:  # Pillow is optional and may be unavailable in some environments
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]


def ensure_plot_dir(path: Path | str = "./plots") -> Path:
    """Create (if needed) and return the directory that will host plot artifacts."""
    plot_dir = Path(path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _has_pillow() -> bool:
    return Image is not None and ImageDraw is not None


def save_line_chart(
    x_values: Sequence[float],
    series_list: Sequence[Sequence[float]],
    series_labels: Sequence[str],
    title: str,
    y_label: str,
    output_path: Path | str,
    *,
    image_size: tuple[int, int] = (1000, 500),
) -> Optional[Path]:
    """Render a simple line chart with Pillow.

    Returns the output path on success or ``None`` when Pillow is unavailable.
    """
    if not _has_pillow() or not x_values or not series_list:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        width, height = image_size
        margin_left, margin_right, margin_top, margin_bottom = 80, 40, 60, 80
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default() if ImageFont is not None else None

        x_min, x_max = min(x_values), max(x_values)
        y_min = min(min(series) for series in series_list)
        y_max = max(max(series) for series in series_list)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        # Draw axes
        draw.line(
            (margin_left, height - margin_bottom, width - margin_right, height - margin_bottom),
            fill="black",
            width=2,
        )
        draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill="black", width=2)

        # Y ticks
        num_y_ticks = 5
        for i in range(num_y_ticks + 1):
            y_val = y_min + (y_max - y_min) * i / num_y_ticks
            y_pix = height - margin_bottom - (y_val - y_min) / (y_max - y_min) * plot_height
            draw.line((margin_left - 6, y_pix, margin_left, y_pix), fill="black", width=1)
            if font:
                draw.text((10, y_pix - 6), f"{y_val:.3f}", fill="black", font=font)

        # X ticks
        num_x_ticks = min(len(x_values) - 1, 10)
        if num_x_ticks <= 0:
            num_x_ticks = 1
        for i in range(num_x_ticks + 1):
            x_idx = int(round(i * (len(x_values) - 1) / num_x_ticks))
            x_val = x_values[x_idx]
            x_pix = margin_left + (x_val - x_min) / (x_max - x_min) * plot_width
            draw.line((x_pix, height - margin_bottom, x_pix, height - margin_bottom + 6), fill="black", width=1)
            if font:
                draw.text((x_pix - 10, height - margin_bottom + 10), f"{x_val}", fill="black", font=font)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for idx, (series, label) in enumerate(zip(series_list, series_labels)):
            points = []
            for xv, yv in zip(x_values, series):
                x_pix = margin_left + (xv - x_min) / (x_max - x_min) * plot_width
                y_pix = height - margin_bottom - (yv - y_min) / (y_max - y_min) * plot_height
                points.append((x_pix, y_pix))
            color = colors[idx % len(colors)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            elif len(points) == 1:
                x_pix, y_pix = points[0]
                draw.ellipse((x_pix - 3, y_pix - 3, x_pix + 3, y_pix + 3), fill=color)

            legend_x = width - margin_right - 180
            legend_y = margin_top + idx * 22
            draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), fill=color)
            if font:
                draw.text((legend_x + 20, legend_y - 2), label, fill="black", font=font)

        if font:
            draw.text((width / 2 - 80, margin_top - 40), title, fill="black", font=font)
            draw.text((width / 2 - 30, height - margin_bottom + 40), "Epoch", fill="black", font=font)
            draw.text((15, margin_top - 30), y_label, fill="black", font=font)

        img.save(output_path)
        return output_path
    except Exception:
        return None


def save_training_curves(
    epoch_indices: Sequence[int],
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    test_maes: Sequence[float],
    output_dir: Path | str,
) -> dict[str, Optional[Path]]:
    """Persist standard training curves to ``output_dir`` using Pillow."""
    plot_dir = ensure_plot_dir(output_dir)
    results = {
        "loss_curves": save_line_chart(
            epoch_indices,
            [train_losses, test_losses],
            ["Train Loss", "Test Loss"],
            "Training and Validation Loss",
            "Loss",
            plot_dir / "loss_curves.png",
        ),
        "test_mae": save_line_chart(
            epoch_indices,
            [test_maes],
            ["Test MAE"],
            "Test Mean Absolute Error",
            "Mean Absolute Error",
            plot_dir / "test_mae.png",
        ),
    }
    return results


def save_loss_vs_epoch(
    epoch_indices: Sequence[int],
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    output_dir: Path | str,
    output_name: str = "loss_vs_epoch.png",
) -> Optional[Path]:
    """Save a dedicated loss-vs-epoch chart."""
    plot_dir = ensure_plot_dir(output_dir)
    return save_line_chart(
        epoch_indices,
        [train_losses, test_losses],
        ["Train Loss", "Test Loss"],
        "Loss vs. Epoch",
        "Loss",
        plot_dir / output_name,
    )


__all__ = [
    "ensure_plot_dir",
    "save_line_chart",
    "save_training_curves",
    "save_loss_vs_epoch",
]
