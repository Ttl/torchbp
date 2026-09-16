# Example SAR data processing script

Instructions:

 1. Download the sample data from: https://hforsten.com/sar.safetensors.zip
 2. Unzip the file to this directory.
 3. Run `sar_process_safetensor_gpga.py` (generalized phase gradient autofocus,
    fast and good quality output) or `sar_process_safetensor_min_entropy.py`
    (optimization based minimum entropy autofocus, worse image quality and much
    slower). It will process the file and display polar formatted image.
    Processed image is also saved to disk for next step.
 4. Run `sar_polar_to_cart.py` to display the previously saved image in Cartesian grid.

Some processing parameters can be modified, either modify the file directly or
you can set the environment variables to change the parameters. For example
"NSWEEPS=51200 ./sar_process_safetensor_gpga.py" to use all the data.

Other environment variables are:
 * DTYPE. 64 (default) or 32 (half precision). 32 uses complex32 for the input
   data, which uses half precision for real and imaginary parts. 64 uses complex64
   which uses normal precision. Minimum entropy only supports 64.
 * AUTOFOCUS. 1 (default) or 0. 0 to disable autofocus.
 * FFBP. 1 (default) or 0. Backprojection is used if FFBP is disabled. Not
   supported with minimum entropy autofocus, it always uses backprojection.
