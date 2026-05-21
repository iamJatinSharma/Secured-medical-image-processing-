"""
Ensemble Predictor — combines model predictions with MC Dropout uncertainty.

Uses model_registry infrastructure for loading and inference. Supports
weighted averaging across multiple models and Monte Carlo Dropout for
uncertainty quantification.
"""

import time
import logging
import random
from typing import Optional

import numpy as np
from PIL import Image

from ml.model_registry import MODEL_CONFIGS, predict as registry_predict

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble inference with optional MC Dropout uncertainty estimation.

    Parameters
    ----------
    model_names : list[str] | None
        Models to include. Defaults to ``["resnet50_breast"]``.
    weights : list[float] | None
        Per-model weights for averaging. Defaults to equal weighting.
    """

    def __init__(
        self,
        model_names: Optional[list] = None,
        weights: Optional[list] = None,
    ):
        self.model_names = model_names or ["resnet50_breast"]

        # Validate model names
        for name in self.model_names:
            if name not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model '{name}'. "
                    f"Available: {list(MODEL_CONFIGS.keys())}"
                )

        # Validate / default weights
        if weights is not None:
            if len(weights) != len(self.model_names):
                raise ValueError(
                    f"weights length ({len(weights)}) != "
                    f"model_names length ({len(self.model_names)})"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(self.model_names)
            self.weights = [1.0 / n] * n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> dict:
        """
        Weighted-average ensemble prediction.

        Parameters
        ----------
        image_path : str
            Path to the image file on disk.

        Returns
        -------
        dict
            label, confidence, all_probabilities, models_used,
            processing_time_ms
        """
        start = time.time()
        model_results = []

        for name in self.model_names:
            result = registry_predict(name, image_path)
            model_results.append(result)

        # All models in the ensemble must share the same class list for
        # meaningful probability averaging.  When a single model is used
        # (the common case) this is trivially satisfied.
        classes = MODEL_CONFIGS[self.model_names[0]]["classes"]

        # Weighted average of probabilities
        combined_probs = np.zeros(len(classes))
        for result, weight in zip(model_results, self.weights):
            probs = np.array([result["all_probabilities"].get(c, 0.0) for c in classes])
            combined_probs += weight * probs

        # Normalise (should already sum to ~1, but guard against float drift)
        prob_sum = combined_probs.sum()
        if prob_sum > 0:
            combined_probs = combined_probs / prob_sum

        idx = int(np.argmax(combined_probs))
        confidence = float(combined_probs[idx])

        elapsed = int((time.time() - start) * 1000)

        return {
            "label": classes[idx],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                c: round(float(p), 4) for c, p in zip(classes, combined_probs)
            },
            "models_used": list(self.model_names),
            "processing_time_ms": elapsed,
        }

    def predict_with_uncertainty(
        self,
        image_path: str,
        n_forward: int = 10,
        model_name: Optional[str] = None,
    ) -> dict:
        """
        MC Dropout uncertainty estimation.

        Enables dropout at inference time by calling ``model.train()`` and
        runs *n_forward* stochastic forward passes.  The mean softmax
        output is used as the prediction; the standard deviation across
        passes quantifies predictive uncertainty.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        n_forward : int
            Number of stochastic forward passes (default 10).
        model_name : str | None
            Which model to use.  Defaults to the first model in the
            ensemble (``self.model_names[0]``).

        Returns
        -------
        dict
            label, confidence, uncertainty, all_probabilities,
            individual_passes, uncertain (bool flag), processing_time_ms
        """
        start = time.time()
        target_model = model_name or self.model_names[0]

        if target_model not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{target_model}'")

        config = MODEL_CONFIGS[target_model]
        classes = config["classes"]

        try:
            return self._mc_dropout_torch(
                target_model, image_path, n_forward, classes, config, start
            )
        except ImportError:
            logger.warning("PyTorch unavailable — falling back to mock MC Dropout")
            return self._mc_dropout_mock(classes, n_forward, start)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mc_dropout_torch(
        model_name: str,
        image_path: str,
        n_forward: int,
        classes: list,
        config: dict,
        start: float,
    ) -> dict:
        """Real MC Dropout using PyTorch model."""
        import torch
        import torch.nn.functional as F
        from ml.model_registry import _load_model, _preprocess_image

        model = _load_model(model_name)
        if model is None:
            raise ImportError("Model returned None (PyTorch unavailable)")

        tensor = _preprocess_image(image_path, config["input_size"])

        # Enable dropout layers (train mode) but keep BatchNorm frozen
        model.train()
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.eval()

        all_probs = []
        with torch.no_grad():
            for _ in range(n_forward):
                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
                all_probs.append(probs)

        # Restore eval mode so the cached model is left in a clean state
        model.eval()

        all_probs = np.array(all_probs)  # (n_forward, num_classes)
        mean_probs = all_probs.mean(axis=0)
        std_probs = all_probs.std(axis=0)

        # Normalise mean (float-drift guard)
        prob_sum = mean_probs.sum()
        if prob_sum > 0:
            mean_probs = mean_probs / prob_sum

        idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[idx])
        uncertainty = float(std_probs[idx])

        elapsed = int((time.time() - start) * 1000)

        return {
            "label": classes[idx],
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "all_probabilities": {
                c: round(float(p), 4) for c, p in zip(classes, mean_probs)
            },
            "std_per_class": {
                c: round(float(s), 4) for c, s in zip(classes, std_probs)
            },
            "individual_passes": [
                {c: round(float(p), 4) for c, p in zip(classes, row)}
                for row in all_probs
            ],
            "uncertain": confidence < 0.7,
            "n_forward_passes": n_forward,
            "model_used": model_name,
            "processing_time_ms": elapsed,
        }

    @staticmethod
    def _mc_dropout_mock(
        classes: list,
        n_forward: int,
        start: float,
    ) -> dict:
        """Mock MC Dropout when PyTorch is unavailable."""
        all_probs = []
        for _ in range(n_forward):
            raw = [random.random() for _ in classes]
            total = sum(raw)
            all_probs.append([p / total for p in raw])

        all_probs = np.array(all_probs)
        mean_probs = all_probs.mean(axis=0)
        std_probs = all_probs.std(axis=0)

        prob_sum = mean_probs.sum()
        if prob_sum > 0:
            mean_probs = mean_probs / prob_sum

        idx = int(np.argmax(mean_probs))
        confidence = float(mean_probs[idx])
        uncertainty = float(std_probs[idx])

        elapsed = int((time.time() - start) * 1000)

        return {
            "label": classes[idx],
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "all_probabilities": {
                c: round(float(p), 4) for c, p in zip(classes, mean_probs)
            },
            "std_per_class": {
                c: round(float(s), 4) for c, s in zip(classes, std_probs)
            },
            "individual_passes": [
                {c: round(float(p), 4) for c, p in zip(classes, row)}
                for row in all_probs
            ],
            "uncertain": confidence < 0.7,
            "n_forward_passes": n_forward,
            "model_used": "mock",
            "processing_time_ms": elapsed,
        }
