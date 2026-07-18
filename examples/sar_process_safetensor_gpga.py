#!/usr/bin/env python
# Example SAR data processing script.
# Sample data can be downloaded from: https://hforsten.com/sar.safetensors.zip
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pickle
import torch
import torchbp
from torchbp.util import make_polar_grid
from torchbp.grid import PolarGrid, CartesianGrid
from safetensors.torch import safe_open
from sar_process_safetensor import grid_extent
plt.style.use("ggplot")

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

    # Windowing functions
    range_window = "hamming"
    angle_window = ("taylor", 4, 50)
    # FFT oversampling factor. Increase to decrease interpolation error.
    fft_oversample = 1.5
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    print("Device:", dev)

    synchronize = lambda: None if dev == "cpu" else torch.cuda.synchronize()

    # Distance in radar data corresponding to zero actual distance
    # Slightly higher than zero due to antenna feedlines and other delays.
    d0 = 0.5
    data_dtype = torch.complex64  # Can be `torch.complex32` to save VRAM
    # Use fast factorized backprojection, slightly reduces the image quality
    # but is faster.
    ffbp = True
    # Autofocus image to improve image quality.
    autofocus = True

    c0 = 299792458

    # Open the input data. The raw sweeps are not loaded here: safetensors
    # supports reading slices on demand, so the sweeps are streamed from
    # the file chunk by chunk as the backprojection consumes them and the
    # raw data is never fully resident in memory.
    try:
        f_st = safe_open(filename, framework="pt", device="cpu")
    except FileNotFoundError:
        print(f"Input file {filename} not found.")
        sys.exit(1)
    mission = {k: float(v) for k, v in f_st.metadata().items()}
    data_slice = f_st.get_slice("data")
    nsweeps = min(nsweeps, data_slice.get_shape()[0] - sweep_start)
    nsamples = data_slice.get_shape()[-1]
    pos = f_st.get_tensor("pos")[sweep_start:sweep_start+nsweeps].cpu().numpy()
    att = f_st.get_tensor("att")[sweep_start:sweep_start+nsweeps].cpu().numpy()
    counts = f_st.get_tensor("counts")[sweep_start:sweep_start+nsweeps]

    bw = mission["bw"]
    fc = mission["fc"]
    fs = mission["fsample"]
    origin_angle = mission["origin_angle"]
    tsweep = nsamples / fs
    sweep_interval = mission["pri"]
    res = c0 / (2 * mission["bw"])

    # Calculate Cartesian grid that fits the radar image
    antenna_bw = 50 * np.pi / 180
    x, y = grid_extent(pos, att, x0, x1, bw=antenna_bw, origin_angle=origin_angle)
    nx = int((x[1] - x[0]) / res)
    ny = int((y[1] - y[0]) / res)
    grid = CartesianGrid(x_range=x, y_range=y, nx=nx, ny=ny)
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

    # Generate window functions
    wr = signal.get_window(range_window, nsamples)
    wr /= np.mean(wr)
    wr = torch.tensor(wr).to(dtype=torch.float32, device=dev)
    wa = torch.tensor(
        signal.get_window(angle_window, nsweeps, fftbins=False)
    ).to(dtype=torch.float32, device=dev)
    wa /= torch.mean(wa)

    n = int(nsamples * fft_oversample)
    fft_oversample = n / nsamples
    # Modulation frequency to center the data spectrum to DC for decreased
    # interpolation error.
    data_fmod = -torch.pi * (1 - (fft_oversample-1) / fft_oversample)

    f = torch.fft.rfftfreq(n, d=1 / fs).to(dev)
    # Residual video phase compensation
    rvp = torch.exp(-1j * torch.pi * f**2 * tsweep / bw)
    r_res = c0 / (2 * bw * fft_oversample)
    del f
    data_fmod_f = torch.exp(1j*data_fmod*torch.arange(n//2+1, device=dev))[None,:]

    # Lazy range compression: the windows, oversampled FFT and RVP multiply
    # run per chunk on dev when a backprojection loads sweeps, reading the
    # raw chunk straight from the safetensors file. Neither the raw nor the
    # range compressed data is ever fully resident; ffbp streams the data
    # one leaf subaperture at a time. The azimuth window is indexed with
    # the absolute sweep indices of the chunk.
    def range_compress(chunk, start, stop):
        x = chunk[:, 0, :].to(torch.float32) * wa[start:stop, None] * wr[None, :]
        fsw = torch.fft.rfft(x, n=n, norm="forward", dim=-1).conj()
        fsw *= data_fmod_f
        fsw *= rvp[None, :]
        return fsw

    fsweeps = torchbp.CallbackData(
        lambda a, b: data_slice[sweep_start + a:sweep_start + b],
        (nsweeps, n // 2 + 1), dtype=data_dtype, device=dev,
        transform=range_compress)

    if autofocus:
        # The autofocus re-reads the data every iteration, so cache the
        # range compressed sweeps: on the compute device when they fit
        # (equivalent to eager range compression), in pinned CPU RAM when
        # VRAM is short, on a disk file when RAM is short too.
        nbytes = nsweeps * (n // 2 + 1) * torch.empty(0, dtype=data_dtype).element_size()
        if nbytes > torchbp.data.available_ram() // 2:
            storage, storage_device = "disk", None
        elif dev == "cuda" and nbytes > torch.cuda.mem_get_info()[0] // 2:
            storage, storage_device = "ram", "cpu"
        else:
            storage, storage_device = "ram", None
        print("Range compressing to", storage if storage_device is None
              else f"{storage} ({storage_device})", "cache")
        tstart = time.time()
        fsweeps = torchbp.CachedData(
            fsweeps, storage=storage, storage_device=storage_device).fill()
        print(f"Range compression done in {time.time() - tstart:.3g} s")

    pos = pos.to(device=dev)

    origin = torch.tensor([torch.mean(pos[:,0]), torch.mean(pos[:,1]), 0],
            device=dev, dtype=torch.float32)[None,:]
    pos_centered = pos - origin

    if autofocus:
        print("Calculating autofocus. This might take a while.")
        synchronize()
        tstart = time.time()
        sar_img, pos_new = torchbp.autofocus.gpga_tde(
            None, fsweeps, pos_centered, fc, r_res,
            grid_polar_autofocus, d0=d0,
            azimuth_divisions=8, range_divisions=8,
            algorithm="ffbp" if ffbp else "bp",
            data_fmod=data_fmod, verbose=True
        )
        synchronize()
        print(f"Autofocus done in {time.time() - tstart:.3g} s")

        plt.figure()
        plt.plot((pos_new[:,0] - pos_centered[:,0]).cpu().numpy(), label="x")
        plt.plot((pos_new[:,1] - pos_centered[:,1]).cpu().numpy(), label="y")
        plt.plot((pos_new[:,2] - pos_centered[:,2]).cpu().numpy(), label="z")
        plt.xlabel("Sweep index")
        plt.ylabel("Solved position error (m)")
        plt.legend(loc="best")

        pos_centered = pos_new

    print("Focusing final image")
    synchronize()
    tstart = time.time()
    if ffbp:
        sar_img = torchbp.ops.ffbp(
            fsweeps, grid_polar, fc, r_res, pos_centered,
            stages=5, divisions=2, d0=d0, oversample_r=1.3, oversample_theta=1.3,
            interp_method=("knab", 6, 1.3), data_fmod=data_fmod, alias_fmod=None
        )
    else:
        # Direct backprojection needs the whole tensor at once; materialize
        # the lazy source in blocks to bound the range compression's peak
        # memory (ffbp streams it leaf by leaf and needs none of this).
        full = torch.zeros((len(fsweeps), n // 2 + 1), dtype=data_dtype, device=dev)
        block = (len(fsweeps) + 15) // 16
        for s0 in range(0, len(fsweeps), block):
            s1 = min(s0 + block, len(fsweeps))
            full[s0:s1] = fsweeps[s0:s1].load()
        fsweeps = full
        del full
        sar_img = torchbp.ops.backprojection_polar_2d(
            fsweeps, grid_polar, fc, r_res, pos_centered, d0,
            data_fmod=data_fmod
        )[0]
    synchronize()
    print(f"Final image created in {time.time() - tstart:.3g} s")
    print("Entropy", torchbp.util.entropy(sar_img).item())
    sar_img = sar_img.cpu().numpy()

    plt.figure()
    extent = [grid_polar.r0, grid_polar.r1, grid_polar.theta0, grid_polar.theta1]
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
        pickle.dump(
            (sar_img, mission, grid.to_dict(), grid_polar.to_dict(), origin, origin_angle), f
        )

    plt.show(block=True)
