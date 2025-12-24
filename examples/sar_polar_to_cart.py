#!/usr/bin/env python
# Visualize pickled radar image
import pickle
import matplotlib.pyplot as plt
import torchbp
from torchbp.util import entropy
import torch
import sys
from torchvision.transforms.functional import resize

if __name__ == "__main__":
    filename = "sar_img.p"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    with open(filename, "rb") as f:
        sar_img, mission, grid, grid_polar, origin, origin_angle = pickle.load(f)

    dev = "cuda"
    sar_img = torch.from_numpy(sar_img).to(dtype=torch.complex64, device=dev)
    fc = mission["fc"]
    print("Entropy", entropy(sar_img).item())

    # Increase Cartesian image size
    oversample = 1
    # Increases image size, but then resamples it down by the same amount
    # Can be used for multilook processing, when the input polar format data
    # resolution is higher than can fit into the Cartesian grid
    multilook = 2
    grid["nx"] = int(oversample * grid["nx"] * multilook)
    grid["ny"] = int(oversample * grid["ny"] * multilook)

    plt.figure()
    origin = torch.from_numpy(origin).to(dtype=torch.float32, device=dev)
    # Amplitude scaling in image
    m = 20 * torch.log10(torch.median(torch.abs(sar_img))) - 3
    m = m.cpu().numpy()
    m2 = m + 40

    sar_img_cart = torchbp.ops.polar_to_cart(
        torch.abs(sar_img),
        origin,
        grid_polar,
        grid,
        fc,
        origin_angle
    )
    extent = [grid["x"][0], grid["x"][1], grid["y"][0], grid["y"][1]]
    img_db = torch.abs(sar_img_cart) + 1e-10
    out_shape = [img_db.shape[-2] // multilook, img_db.shape[-1] // multilook]
    img_db = resize(img_db, out_shape).squeeze()
    img_db = 20 * torch.log10(img_db)
    img_db = img_db.cpu().numpy()

    plt.imshow(img_db.T, origin="lower", aspect="equal", vmin=m, vmax=m2, extent=extent)
    plt.grid(False)
    plt.savefig("sar_img_cart.png", dpi=700)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show(block=True)
