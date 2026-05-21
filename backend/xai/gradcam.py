import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GradCAM:
    """Grad-CAM implementation for CNN visualization.

    Generates class-discriminative heatmaps by combining feature map
    activations with gradient-based importance weights.

    Args:
        model: A PyTorch CNN model (e.g. ResNet50).
        target_layer: The layer to compute Grad-CAM on (e.g. model.layer4[-1]).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def _preprocess(self, image):
        """Convert PIL Image or numpy array to normalized tensor.

        Returns:
            Tensor of shape (1, 3, 224, 224).
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        resized = cv2.resize(image, (224, 224))
        # HWC -> float32 [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        for c in range(3):
            tensor[:, :, c] = (tensor[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        # HWC -> CHW -> NCHW
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = torch.from_numpy(tensor).unsqueeze(0)
        return tensor

    def generate(self, image, target_class=None):
        """Generate a Grad-CAM heatmap for the given image.

        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB, uint8.
            target_class: Class index to compute CAM for. If None, uses
                the model's predicted (argmax) class.

        Returns:
            Numpy array (H, W) with float values in [0.0, 1.0].
        """
        # Remember original size for final resize
        if isinstance(image, Image.Image):
            orig_h, orig_w = np.array(image).shape[:2]
        else:
            orig_h, orig_w = image.shape[:2]

        input_tensor = self._preprocess(image)
        input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients
        self.model.zero_grad()

        # Backward from the target class score
        score = output[0, target_class]
        score.backward()

        # Grad-CAM computation
        # gradients shape: (1, C, H, W)
        gradients = self._gradients
        activations = self._activations

        # Global average pooling over spatial dims -> (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps -> (1, 1, H, W)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Remove batch and channel dims -> (H, W)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to original image dimensions
        heatmap = cv2.resize(cam, (orig_w, orig_h))

        return heatmap

    def remove_hooks(self):
        """Remove registered hooks from the target layer."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """Blend a Grad-CAM heatmap onto an original image.

    Args:
        original_image: Numpy array (H, W, 3) uint8.
        heatmap: Numpy array (H, W) float in [0.0, 1.0].
        alpha: Blending factor for the heatmap (default 0.4).

    Returns:
        Numpy array (H, W, 3) uint8 — the blended image.
    """
    # Scale heatmap to 0-255 uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply JET colormap -> (H, W, 3) BGR
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert BGR -> RGB to match original_image convention
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Blend
    output = (1 - alpha) * original_image.astype(np.float32) + alpha * colored.astype(np.float32)
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output
