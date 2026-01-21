"""Voice tensor generation and mutation for random walk optimization."""

from typing import List, Optional

import torch


class VoiceGenerator:
    """Generates and mutates voice tensors for the random walk algorithm.

    Uses statistics from a population of voice tensors to guide mutations
    in a meaningful direction through the voice tensor space.
    """

    def __init__(
        self,
        voices: List[torch.Tensor],
        starting_voice: Optional[str] = None
    ):
        """Initialize with a population of voice tensors.

        Args:
            voices: List of voice tensors to compute statistics from
            starting_voice: Optional path to a specific voice tensor to start from
        """
        self.voices = voices

        # Compute statistics across all voices
        self.stacked = torch.stack(voices, dim=0)
        self.mean = self.stacked.mean(dim=0)
        self.std = self.stacked.std(dim=0)
        self.min = self.stacked.min(dim=0)[0]
        self.max = self.stacked.max(dim=0)[0]

        # Set starting voice
        if starting_voice:
            self.starting_voice = torch.load(starting_voice, weights_only=True)
        else:
            self.starting_voice = self.mean

    def generate_voice(
        self,
        base_tensor: Optional[torch.Tensor] = None,
        diversity: float = 1.0,
        device: str = "cpu",
        clip: bool = False
    ) -> torch.Tensor:
        """Generate a new voice tensor by mutating a base tensor.

        Applies Gaussian noise scaled by the population standard deviation
        and a diversity factor to explore the voice tensor space.

        Args:
            base_tensor: The base tensor to mutate (default: population mean)
            diversity: Scale factor for the mutation (typically 0.01-0.15)
            device: Device to generate the tensor on
            clip: Whether to clip values to observed min/max range

        Returns:
            New mutated voice tensor
        """
        if base_tensor is None:
            base_tensor = self.mean.to(device)
        else:
            base_tensor = base_tensor.clone().to(device)

        # Generate random noise with same shape
        noise = torch.randn_like(base_tensor, device=device)

        # Scale noise by standard deviation and diversity factor
        scaled_noise = noise * self.std.to(device) * diversity

        # Add scaled noise to base tensor
        new_tensor = base_tensor + scaled_noise

        if clip:
            new_tensor = torch.clamp(new_tensor, self.min.to(device), self.max.to(device))

        return new_tensor

    def interpolate(
        self,
        voice1: torch.Tensor,
        voice2: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """Interpolate between two voice tensors.

        Args:
            voice1: First voice tensor
            voice2: Second voice tensor
            alpha: Interpolation factor (0.0 = voice1, 1.0 = voice2)

        Returns:
            Interpolated voice tensor
        """
        midpoint = (voice1 + voice2) / 2
        diff = voice2 - voice1
        return midpoint + alpha * diff
