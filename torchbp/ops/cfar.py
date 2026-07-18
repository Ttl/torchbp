import torch
from torch import Tensor

__all__ = [
    "cfar_2d",
]


def cfar_2d(
    img: Tensor, Navg: tuple, Nguard: tuple, threshold: float, peaks_only: bool = False
) -> Tensor:
    """
    Constant False Alaram Rate detection for 2D image.

    Parameters
    ----------
    img : Tensor
        Absolute valued 2D image. If 3D then the first dimension is batch dimension.
    Navg : tuple
        Number of averaged cells in 2D (N1, N0).
    Nguard : tuple
        Number of guard cells in 2D (N1, N0).
        Both dimensions need to be less or same than the same entry in Navg.
    threshold : float
        Threshold for detection.
    peaks_only : bool
        Reject pixels that are not higher than their immediate neighbors.
        Defaults is False.

    Returns
    -------
    detection : Tensor
        Detections tensor same shape as the input image. For each pixel 0 for no
        detection, positive value is the SNR of detection at that position.
    """
    if img.dim() == 3:
        nbatch = img.shape[0]
        N0 = img.shape[1]
        N1 = img.shape[2]
    elif img.dim() == 2:
        nbatch = 1
        N0 = img.shape[0]
        N1 = img.shape[1]
    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if threshold <= 0:
        raise ValueError("Threshold should be positive")
    if len(Navg) != 2:
        raise ValueError("Navg dimension should be 2")
    if len(Nguard) != 2:
        raise ValueError("Nguard dimension should be 2")
    if Nguard[0] > Navg[0]:
        raise ValueError("Nguard[0] > Navg[0]")
    if Nguard[1] > Navg[1]:
        raise ValueError("Nguard[1] > Navg[1]")
    if Navg[0] < 0:
        raise ValueError("Navg[0] < 0")
    if Navg[1] < 0:
        raise ValueError("Navg[1] < 0")
    if Nguard[0] < 0:
        raise ValueError("Nguard[0] < 0")
    if Nguard[1] < 0:
        raise ValueError("Nguard[1] < 0")

    return torch.ops.torchbp.cfar_2d.default(
        img,
        nbatch,
        N0,
        N1,
        Navg[0],
        Navg[1],
        Nguard[0],
        Nguard[1],
        threshold,
        peaks_only,
    )
