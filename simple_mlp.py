"""Minimal multilayer perceptron utilities that avoid third-party dependencies."""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

Number = float


# Constant auxiliary feature support --------------------------------------------------------

#
# Some training workflows require injecting a fixed auxiliary feature into the network
# without changing the CSV schema or notebook code.  The following constants capture the
# specification "append the constant 120 after normalization".  With the default
# normalization (mean=120, std=1) the resulting feature takes the value 0.0, but the
# explicit constants keep the intent self-documenting and make future adjustments easy.
CONSTANT_FEATURE_VALUE: Number = 120.0
CONSTANT_FEATURE_MEAN: Number = 120.0
CONSTANT_FEATURE_STD: Number = 1.0
NORMALIZED_CONSTANT_FEATURE: Number = (CONSTANT_FEATURE_VALUE - CONSTANT_FEATURE_MEAN) / CONSTANT_FEATURE_STD


def read_csv_dataset(
    path: Path,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> Tuple[List[List[Number]], Optional[List[List[Number]]]]:
    """Load the specified feature/target columns from a CSV file."""
    features: List[List[Number]] = []
    targets: List[List[Number]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"File {path} does not contain a header row.")

        missing_features = [col for col in feature_columns if col not in reader.fieldnames]
        if missing_features:
            raise ValueError(
                f"Missing feature columns {missing_features} in {path.name}; columns available: {reader.fieldnames}"
            )

        if target_columns:
            missing_targets = [col for col in target_columns if col not in reader.fieldnames]
            if missing_targets:
                raise ValueError(
                    f"Missing target columns {missing_targets} in {path.name}; columns available: {reader.fieldnames}"
                )

        for row_index, row in enumerate(reader, start=2):
            try:
                feature_row = [float(row[column]) for column in feature_columns]
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_index} in {path.name} contains invalid feature data: {exc}"
                ) from exc
            features.append(feature_row)

            if target_columns:
                try:
                    target_row = [float(row[column]) for column in target_columns]
                except ValueError as exc:
                    raise ValueError(
                        f"Row {row_index} in {path.name} contains invalid target data: {exc}"
                    ) from exc
                targets.append(target_row)

    if not features:
        raise ValueError(f"File {path} did not contain any data rows.")

    return features, targets if target_columns else None


@dataclass
class Standardizer:
    means: List[Number]
    stds: List[Number]

    @classmethod
    def fit(cls, rows: Sequence[Sequence[Number]]) -> "Standardizer":
        if not rows:
            raise ValueError("Cannot fit a standardizer on empty data.")

        num_features = len(rows[0])
        sums = [0.0 for _ in range(num_features)]
        sums_sq = [0.0 for _ in range(num_features)]

        for row in rows:
            if len(row) != num_features:
                raise ValueError("Inconsistent feature dimensionality encountered.")
            for idx, value in enumerate(row):
                sums[idx] += value
                sums_sq[idx] += value * value

        count = float(len(rows))
        means = [total / count for total in sums]
        stds: List[Number] = []
        for idx in range(num_features):
            mean = means[idx]
            variance = (sums_sq[idx] / count) - mean * mean
            std = math.sqrt(variance) if variance > 0 else 1.0
            stds.append(std)
        return cls(means=means, stds=stds)

    def transform(self, rows: Sequence[Sequence[Number]]) -> List[List[Number]]:
        transformed: List[List[Number]] = []
        for row in rows:
            if len(row) != len(self.means):
                raise ValueError("Feature count does not match fitted standardizer.")
            transformed.append(
                [(value - self.means[idx]) / self.stds[idx] for idx, value in enumerate(row)]
            )
        return transformed

    def fit_transform(self, rows: Sequence[Sequence[Number]]) -> List[List[Number]]:
        fitted = self.fit(rows)
        self.means = fitted.means
        self.stds = fitted.stds
        return self.transform(rows)


class SimpleMLP:
    """Multi-layer perceptron implemented with basic Python operations."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        learning_rate: float = 1e-3,
        l1_coeff: float = 0.0,
        seed: int = 42,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output dimensions")

        self.original_input_dim = layer_sizes[0]
        self.layer_sizes = list(layer_sizes)
        self.layer_sizes[0] = self.original_input_dim + 1
        self.learning_rate = learning_rate
        self.l1_coeff = l1_coeff
        rng = random.Random(seed)

        self.weights: List[List[List[Number]]] = []
        self.biases: List[List[Number]] = []

        for input_dim, output_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = math.sqrt(6.0 / (input_dim + output_dim))
            weight_matrix = [
                [rng.uniform(-limit, limit) for _ in range(output_dim)] for _ in range(input_dim)
            ]
            bias_vector = [0.0 for _ in range(output_dim)]
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def _augment_inputs(self, inputs: Sequence[Number]) -> List[Number]:
        if len(inputs) == self.layer_sizes[0]:
            # Already augmented (e.g., internal reuse) – return a mutable copy to
            # avoid accidental side effects.
            return list(inputs)
        if len(inputs) != self.original_input_dim:
            raise ValueError(
                f"Expected {self.original_input_dim} input features but received {len(inputs)}."
            )
        augmented = list(inputs)
        augmented.append(NORMALIZED_CONSTANT_FEATURE)
        return augmented

    @staticmethod
    def _sigmoid(value: Number) -> Number:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def forward(self, inputs: Sequence[Number]) -> Tuple[List[List[Number]], List[List[Number]]]:
        augmented_inputs = self._augment_inputs(inputs)
        activations: List[List[Number]] = [list(augmented_inputs)]
        linear_outputs: List[List[Number]] = []

        current = list(augmented_inputs)
        for layer_idx, (weight_matrix, bias_vector) in enumerate(zip(self.weights, self.biases)):
            z_values: List[Number] = []
            for neuron_idx in range(len(bias_vector)):
                total = bias_vector[neuron_idx]
                for input_idx in range(len(current)):
                    total += current[input_idx] * weight_matrix[input_idx][neuron_idx]
                z_values.append(total)
            linear_outputs.append(z_values)

            if layer_idx == len(self.weights) - 1:
                current = [z for z in z_values]
            else:
                current = [z if z > 0.0 else 0.0 for z in z_values]
            activations.append(current)
        return activations, linear_outputs

    def predict_single(self, inputs: Sequence[Number]) -> List[Number]:
        activations, _ = self.forward(inputs)
        return activations[-1]

    def predict_batch(self, inputs: Sequence[Sequence[Number]]) -> List[List[Number]]:
        return [self.predict_single(row) for row in inputs]

    def train(
        self,
        features: Sequence[Sequence[Number]],
        targets: Sequence[Sequence[Number]],
        epochs: int,
        batch_size: int = 32,
        verbose_interval: int = 10,
    ) -> List[Number]:
        if len(features) != len(targets):
            raise ValueError("Feature and target counts must match.")
        if not features:
            raise ValueError("No training data provided.")

        num_samples = len(features)
        indices = list(range(num_samples))
        losses: List[Number] = []
        dataset_size = float(num_samples)

        for epoch in range(1, epochs + 1):
            random.shuffle(indices)
            cumulative_sse = 0.0
            sample_count = 0

            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                grad_w = [
                    [
                        [0.0 for _ in range(len(self.weights[layer_idx][0]))]
                        for _ in range(len(self.weights[layer_idx]))
                    ]
                    for layer_idx in range(len(self.weights))
                ]
                grad_b = [[0.0 for _ in bias_layer] for bias_layer in self.biases]
                batch_sse = 0.0

                for sample_idx in batch_indices:
                    inputs = features[sample_idx]
                    expected = targets[sample_idx]
                    activations, linear_outputs = self.forward(inputs)
                    output_activation = activations[-1]
                    output_dim = len(output_activation)

                    sample_sse = sum((output_activation[i] - expected[i]) ** 2 for i in range(output_dim))
                    batch_sse += sample_sse

                    deltas: List[List[Number]] = [
                        [0.0 for _ in bias_layer] for bias_layer in self.biases
                    ]

                    last_layer = len(self.weights) - 1
                    for neuron_idx in range(len(self.biases[last_layer])):
                        prediction = output_activation[neuron_idx]
                        target_value = expected[neuron_idx]
                        diff = prediction - target_value
                        deltas[last_layer][neuron_idx] = 2.0 * diff

                    for layer_idx in range(len(self.weights) - 2, -1, -1):
                        next_layer = layer_idx + 1
                        for neuron_idx in range(len(self.biases[layer_idx])):
                            backprop_sum = 0.0
                            for next_idx in range(len(self.biases[next_layer])):
                                backprop_sum += self.weights[layer_idx + 1][neuron_idx][next_idx] * deltas[next_layer][next_idx]
                            pre_activation = linear_outputs[layer_idx][neuron_idx]
                            derivative = 1.0 if pre_activation > 0.0 else 0.0
                            deltas[layer_idx][neuron_idx] = backprop_sum * derivative

                    for layer_idx in range(len(self.weights)):
                        prev_activation = activations[layer_idx]
                        for input_idx in range(len(prev_activation)):
                            for neuron_idx in range(len(self.biases[layer_idx])):
                                grad_w[layer_idx][input_idx][neuron_idx] += prev_activation[input_idx] * deltas[layer_idx][neuron_idx]
                        for neuron_idx in range(len(self.biases[layer_idx])):
                            grad_b[layer_idx][neuron_idx] += deltas[layer_idx][neuron_idx]

                batch_size_actual = float(len(batch_indices))
                for layer_idx in range(len(self.weights)):
                    for input_idx in range(len(self.weights[layer_idx])):
                        for neuron_idx in range(len(self.weights[layer_idx][input_idx])):
                            grad = grad_w[layer_idx][input_idx][neuron_idx] / batch_size_actual
                            weight = self.weights[layer_idx][input_idx][neuron_idx]
                            if self.l1_coeff:
                                if weight > 0:
                                    grad += self.l1_coeff / dataset_size
                                elif weight < 0:
                                    grad -= self.l1_coeff / dataset_size
                            self.weights[layer_idx][input_idx][neuron_idx] -= self.learning_rate * grad
                    for neuron_idx in range(len(self.biases[layer_idx])):
                        grad = grad_b[layer_idx][neuron_idx] / batch_size_actual
                        self.biases[layer_idx][neuron_idx] -= self.learning_rate * grad

                cumulative_sse += batch_sse
                sample_count += len(batch_indices)

            data_loss = cumulative_sse / max(1, sample_count)
            if self.l1_coeff:
                total_abs_weights = sum(
                    abs(weight)
                    for layer in self.weights
                    for row in layer
                    for weight in row
                )
                l1_term = (self.l1_coeff * total_abs_weights) / dataset_size
            else:
                l1_term = 0.0

            epoch_loss = data_loss + l1_term
            losses.append(epoch_loss)
            if verbose_interval and (epoch % verbose_interval == 0 or epoch == 1):
                print(f"Epoch {epoch:03d} - Loss: {epoch_loss:.6f}")
        return losses


def mean_squared_error(predictions: Sequence[Sequence[Number]], targets: Sequence[Sequence[Number]]) -> Number:
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target lengths must match.")
    total = 0.0
    count = 0
    for pred_row, target_row in zip(predictions, targets):
        if len(pred_row) != len(target_row):
            raise ValueError("Predictions and targets have mismatched dimensions.")
        for pred_value, target_value in zip(pred_row, target_row):
            diff = pred_value - target_value
            total += diff * diff
            count += 1
    return total / count if count else 0.0


def train_test_split(
    features: Sequence[Sequence[Number]],
    targets: Sequence[Sequence[Number]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[List[Number]], List[List[Number]], List[List[Number]], List[List[Number]]]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    indices = list(range(len(features)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_size = max(1, int(len(indices) * test_ratio))

    test_indices = set(indices[:test_size])
    train_features: List[List[Number]] = []
    test_features: List[List[Number]] = []
    train_targets: List[List[Number]] = []
    test_targets: List[List[Number]] = []

    for idx, feature_row in enumerate(features):
        if idx in test_indices:
            test_features.append(list(feature_row))
            test_targets.append(list(targets[idx]))
        else:
            train_features.append(list(feature_row))
            train_targets.append(list(targets[idx]))

    return train_features, test_features, train_targets, test_targets


def format_predictions(
    features: Sequence[Sequence[Number]],
    predictions: Sequence[Sequence[Number]],
    targets: Optional[Sequence[Sequence[Number]]],
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    max_rows: int = 5,
) -> str:
    headers = list(feature_columns)
    if targets is not None:
        headers.extend(target_columns)
    headers.extend([f"pred_{col}" for col in target_columns])

    rows: List[List[str]] = []
    preview_count = min(len(features), max_rows)
    for idx in range(preview_count):
        row_values: List[str] = [f"{features[idx][col_idx]:.4f}" for col_idx in range(len(feature_columns))]
        if targets is not None:
            row_values.extend(f"{targets[idx][col_idx]:.4f}" for col_idx in range(len(target_columns)))
        row_values.extend(f"{predictions[idx][col_idx]:.4f}" for col_idx in range(len(target_columns)))
        rows.append(row_values)

    column_widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            column_widths[idx] = max(column_widths[idx], len(value))

    header_line = " | ".join(header.ljust(column_widths[idx]) for idx, header in enumerate(headers))
    separator_line = "-+-".join("-" * column_widths[idx] for idx in range(len(headers)))
    body_lines = [
        " | ".join(row[col_idx].ljust(column_widths[col_idx]) for col_idx in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def plot_loss_curve(
    losses: Sequence[Number],
    save_path: Optional[Union[str, Path]] = None,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Render the training loss history to a simple SVG line chart."""

    if not losses:
        raise ValueError("losses must contain at least one value")

    svg_path = Path(save_path) if save_path is not None else Path("loss_curve.svg")
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    min_loss = min(losses)
    max_loss = max(losses)
    margin = 40.0
    plot_width = max(1.0, width - 2 * margin)
    plot_height = max(1.0, height - 2 * margin)

    def x_coord(index: int) -> float:
        if len(losses) == 1:
            return margin + plot_width / 2.0
        step = plot_width / float(len(losses) - 1)
        return margin + index * step

    def y_coord(loss_value: Number) -> float:
        if max_loss == min_loss:
            return margin + plot_height / 2.0
        normalized = (loss_value - min_loss) / (max_loss - min_loss)
        return height - margin - normalized * plot_height

    points = " ".join(f"{x_coord(idx):.2f},{y_coord(value):.2f}" for idx, value in enumerate(losses))

    axis_lines = [
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#444' stroke-width='1' />",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#444' stroke-width='1' />",
    ]

    labels = [
        f"<text x='{margin}' y='{margin - 10}' fill='#222' font-size='14'>Loss</text>",
        f"<text x='{width - margin}' y='{height - margin + 25}' fill='#222' font-size='14' text-anchor='end'>Epoch</text>",
        f"<text x='{margin + 5}' y='{height - margin - 5}' fill='#666' font-size='12'>epoch 1</text>",
        f"<text x='{width - margin - 5}' y='{margin + 15}' fill='#666' font-size='12' text-anchor='end'>epoch {len(losses)}</text>",
        f"<text x='{margin + 5}' y='{margin + 15}' fill='#666' font-size='12'>max: {max_loss:.6f}</text>",
        f"<text x='{margin + 5}' y='{margin + 30}' fill='#666' font-size='12'>min: {min_loss:.6f}</text>",
    ]

    svg_content = """<?xml version='1.0' encoding='UTF-8'?>
<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>
  {axes}
  <polyline fill='none' stroke='#2E86AB' stroke-width='2' points='{points}' />
  {labels}
</svg>
""".format(
        width=width,
        height=height,
        axes="\n  ".join(axis_lines),
        points=points,
        labels="\n  ".join(labels),
    )

    with svg_path.open("w", encoding="utf-8") as handle:
        handle.write(svg_content)

    return svg_path
