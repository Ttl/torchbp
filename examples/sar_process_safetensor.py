#!/usr/bin/env python
# Example SAR data processing script.
# Sample data can be downloaded from: https://hforsten.com/sar.safetensors.zip
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pickle
import torch
import torchbp
from torchbp.util import make_polar_grid
from safetensors.torch import safe_open

plt.style.use("ggplot")


def grid_extent(pos, att, min_range, max_range, bw=0, origin_angle=0):
    """
    Return grid dimension that contain the radar data.

    Parameters
    ----------
    pos : np.array
        Platform xyz-position vector. Shape: [N, 3].
    att : np.array
        Antenna Euler angle vector. Shape: [N, 3].
    min_range : float
        Minimum range from radar in m.
    max_range : float
        Maximum range from radar in m.
    bw : float
        Antenna beam width in radians.
    origin_angle : float
        Input position rotation angle.

    Returns
    -------------
    x, y : tuple
        Minimum and maximum X and Y coordinates for image grid.
    """
    x = None
    y = None
    for b in [-bw, 0, bw]:
        yaw = att[:, 2] + b + origin_angle
        pos = pos[:, :2]
        range_vector = np.array([np.cos(yaw), np.sin(yaw)]).T
        fc_range = pos + range_vector * max_range
        max_x = (np.min(fc_range[:, 0]), np.max(fc_range[:, 0]))
        max_y = (np.min(fc_range[:, 1]), np.max(fc_range[:, 1]))
        fc_range = pos + range_vector * min_range
        min_x = (np.min(fc_range[:, 0]), np.max(fc_range[:, 0]))
        min_y = (np.min(fc_range[:, 1]), np.max(fc_range[:, 1]))
        xn = (min(min_x[0], max_x[0]), max(min_x[1], max_x[1]))
        yn = (min(min_y[0], max_y[0]), max(min_y[1], max_y[1]))
        if x is None:
            x = xn
            y = yn
        else:
            x = (min(xn[0], x[0]), max(x[1], xn[1]))
            y = (min(yn[0], y[0]), max(y[1], yn[1]))
    return x, y


def load_data(filename):
    tensors = {}
    with safe_open(filename, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        mission = f.metadata()
        mission = {k: float(mission[k]) for k in mission.keys()}
    return mission, tensors


if __name__ == "__main__":
    filename = "sar.safetensors"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    # Final image dimensions
    x0 = 1
    x1 = 2000
    # Image dimensions during autofocus, typically smaller than the final image
    autofocus_x0 = 400
    autofocus_x1 = 1200
    autofocus_theta_limit = 0.8
    # Azimuth range in polar image in sin of radians. 1 for full 180 degrees.
    theta_limit = 1
    # Decrease the number of sweeps to speed up the calculation
    nsweeps = 10000 # Max 51200
    sweep_start = 0
    # Maximum number of autofocus iterations
    max_steps = 15
    # Maximum autofocus position update in wavelengths
    # Optimal value depends on the maximum error in the image
    max_step_limit = 0.5  # Try 5 with 50k sweeps
    data_dtype = torch.complex64  # Can be `torch.complex32` to save VRAM

    # Windowing functions
    range_window = "hamming"
    angle_window = ("taylor", 4, 50)
    # FFT oversampling factor, decrease to 2 to save some VRAM
    fft_oversample = 3
    dev = torch.device("cuda")
    # Distance in radar data corresponding to zero actual distance
    # Slightly higher than zero due to antenna feedlines and other delays.
    d0 = 0.5
    # TX antenna distance to RX antenna
    ant_tx_dy = -96.6e-3

    # Calculate initial estimate using PGA
    initial_pga = False

    c0 = 299792458

    # Load the input data
    try:
        mission, tensors = load_data(filename)
    except FileNotFoundError:
        print("Input file {filename} not found.")

    sweeps = tensors["data"][sweep_start:nsweeps].to(dtype=torch.float32)
    pos = tensors["pos"][sweep_start:nsweeps].cpu().numpy()
    att = tensors["att"][sweep_start:nsweeps].cpu().numpy()
    counts = tensors["counts"][sweep_start:nsweeps]
    nsweeps = sweeps.shape[0]
    del tensors

    bw = mission["bw"]
    fc = mission["fc"]
    fs = mission["fsample"]
    origin_angle = mission["origin_angle"]
    tsweep = sweeps.shape[-1] / fs
    sweep_interval = mission["pri"]
    res = c0 / (2 * mission["bw"])

    # Calculate Cartesian grid that fits the radar image
    antenna_bw = 50 * np.pi / 180
    x, y = grid_extent(pos, att, x0, x1, bw=antenna_bw, origin_angle=origin_angle)
    nx = int((x[1] - x[0]) / res)
    ny = int((y[1] - y[0]) / res)
    grid = {"x": x, "y": y, "nx": nx, "ny": ny}
    print("mission", mission)
    print("grid_cart", grid)

    # Calculate polar grid
    d = np.linalg.norm(pos[-1] - pos[0])
    wl = c0 / fc
    spacing = d / wl / nsweeps
    # Critically spaced array would be 0.25 wavelengths apart
    ntheta = int(1 + nsweeps * spacing * theta_limit / 0.25)
    nr = int((x1 - x0) / res)
    az = att[:, 2]
    mean_az = np.angle(np.mean(np.exp(1j * az)))
    grid_polar = make_polar_grid(
        x0,
        x1,
        nr,
        ntheta,
        theta_limit=theta_limit,
        squint=mean_az if theta_limit < 1 else 0,
    )

    nr = int((autofocus_x1 - autofocus_x0) / res)
    ntheta = int(1 + nsweeps * spacing * autofocus_theta_limit / 0.25)
    grid_polar_autofocus = make_polar_grid(
        autofocus_x0,
        autofocus_x1,
        nr,
        ntheta,
        theta_limit=autofocus_theta_limit,
        squint=mean_az,
    )
    print("grid", grid_polar)
    print("grid autofocus", grid_polar_autofocus)

    pos = torch.from_numpy(pos).to(dtype=torch.float32, device=dev)
    vel = torch.zeros_like(pos)  # Velocity is not used in the current implementation
    att = torch.from_numpy(att).to(dtype=torch.float32, device=dev)

    # Generate window functions
    nsamples = sweeps.shape[-1]
    wr = signal.get_window(range_window, nsamples)
    wr /= np.mean(wr)
    wr = torch.tensor(wr).to(dtype=torch.float32, device=dev)
    wa = torch.tensor(
        signal.get_window(angle_window, sweeps.shape[0], fftbins=False)
    ).to(dtype=torch.float32, device=dev)
    wa /= torch.mean(wa)

    # Residual video phase compensation
    nsamples = sweeps.shape[-1]
    f = torch.fft.rfftfreq(int(nsamples * fft_oversample), d=1 / fs).to(dev)
    rvp = torch.exp(-1j * torch.pi * f**2 * tsweep / bw)
    r_res = c0 / (2 * bw * fft_oversample)
    del f

    # Timestamp of each sweep
    data_time = sweep_interval * counts

    v = torch.diff(pos, dim=0, prepend=pos[0].unsqueeze(0)) / sweep_interval
    pos_mean = torch.mean(pos, dim=0)
    v_orig = v.detach().clone()

    # Apply windowing
    sweeps *= wa[:, None, None].cpu()
    sweeps *= wr[None, None, :].cpu()

    # FFT radar data in blocks to decrease the maximum needed VRAM
    n = int(nsamples * fft_oversample)
    fsweeps = torch.zeros((sweeps.shape[0], n // 2 + 1), dtype=data_dtype, device=dev)
    blocks = 16
    block = (sweeps.shape[0] + blocks - 1) // blocks
    for b in range(blocks):
        s0 = b * block
        s1 = min((b + 1) * block, sweeps.shape[0])
        fsw = torch.fft.rfft(
            sweeps[s0:s1, 0, :].to(device=dev), n=n, norm="forward", dim=-1
        )
        fsw = torch.conj(fsw)
        fsw *= rvp[None, :]
        fsweeps[s0:s1] = fsw.to(dtype=data_dtype)
    del sweeps
    del fsw

    pos = pos.to(device=dev)
    vel = vel.to(device=dev)
    att = att.to(device=dev)
    data_time = data_time.to(device=dev)

    if max_steps > 1:
        if initial_pga:
            print("Calculating initial estimate with PGA")
            origin = torch.tensor([torch.mean(pos[:,0]), torch.mean(pos[:,1]), 0],
                    device=dev, dtype=torch.float32)[None,:]
            pos_centered = pos - origin
            sar_img, phi = torchbp.autofocus.gpga_ml_bp_polar(None, fsweeps,
                    pos_centered, vel, att, fc, r_res, grid_polar_autofocus,
                    window_width=nsweeps//8, d0=d0, target_threshold_db=20,
                    ant_tx_dy=ant_tx_dy)

            d = torchbp.util.phase_to_distance(phi, fc)
            d -= torch.mean(d)
            pos[:,0] = pos[:,0] + d

        print("Calculating autofocus. This might take a while. Press Ctrl-C to interrupt.")
        sar_img, origin, pos, steps = torchbp.autofocus.bp_polar_grad_minimum_entropy(
            fsweeps,
            data_time,
            pos,
            vel,
            att,
            fc,
            r_res,
            grid_polar_autofocus,
            wa,
            tx_norm=None,
            max_steps=max_steps,
            lr_max=10000,
            d0=d0,
            ant_tx_dy=ant_tx_dy,
            pos_reg=0.1,
            lr_reduce=0.8,
            verbose=True,
            convergence_limit=0.01,
            max_step_limit=max_step_limit,
            grad_limit_quantile=0.99,
            fixed_pos=0,
        )
        v = torch.diff(pos, dim=0, prepend=pos[0].unsqueeze(0)) / sweep_interval

        plt.figure()
        plt.title("Original and optimized velocity")
        p = v.detach().cpu().numpy()
        plt.plot(p[:, 0], label="vx opt")
        plt.plot(p[:, 1], label="vy opt")
        plt.plot(p[:, 2], label="vz opt")
        po = v_orig.detach().cpu().numpy()
        plt.plot(po[:, 0], label="vx")
        plt.plot(po[:, 1], label="vy")
        plt.plot(po[:, 2], label="vz")
        plt.legend(loc="best")
        plt.xlabel("Sweep index")
        plt.ylabel("Velocity (m/s)")

    origin = torch.tensor(
        [torch.mean(pos[:, 0]), torch.mean(pos[:, 1]), 0],
        device=dev,
        dtype=torch.float32,
    )[None, :]
    pos_centered = pos - origin
    print("Focusing final image")
    sar_img = torchbp.ops.backprojection_polar_2d( fsweeps, grid_polar, fc,
            r_res, pos_centered, vel, att, d0, ant_tx_dy).squeeze()
    print("Entropy", torchbp.util.entropy(sar_img).item())
    sar_img = sar_img.cpu().numpy()

    plt.figure()
    extent = [
        grid_polar["r"][0],
        grid_polar["r"][1],
        grid_polar["theta"][0],
        grid_polar["theta"][1],
    ]
    abs_img = np.abs(sar_img)
    m = 20 * np.log10(np.median(abs_img)) - 13
    plt.imshow(
        20 * np.log10(abs_img).T, aspect="auto", origin="lower", extent=extent, vmin=m
    )
    plt.grid(False)
    plt.xlabel("Range (m)")
    plt.ylabel("Angle (sin(radians))")
    print("Exporting image")
    plt.savefig("sar_img.png", dpi=400)

    # Export image as pickle file
    with open("sar_img.p", "wb") as f:
        origin = origin.cpu().numpy().squeeze()
        pickle.dump((sar_img, mission, grid, grid_polar, origin, origin_angle), f)

    plt.show(block=True)
