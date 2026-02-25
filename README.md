# Label More Radio Burst

Radio burst labeling tools and protocols.


<img width="400" alt="image" src="https://github.com/user-attachments/assets/cb0b8b80-6afd-4049-8b0e-d75e85689804" />


## File Naming Convention

All file names should follow this format:

`[datetime]_[duration]_[start_freq]_[end_freq]_[fmt].[ext]`

Example:

`yyyy-mm-ddTHH:MM:SS_900s_30MHz_80MHz_0image.png`

## File Format (`[fmt]`)

### 1) `0image`

Plain image (`png`, `jpg`, etc.).

- Spectrum image only
- Frequency: low to high from bottom to top
- Time: left to right
- Output size: `800x300` (W x H)

### 2.1) `ivstack2`

Stacked image of polarization I and V (`png`, `jpg`, etc.).

- Upper panel: log scale `0.5-200 sfu`
- Lower panel: pol-V `-0.5 to 0.5`

### 2.2) `ivstack3`

Stacked image of polarization I and V.

- Panel 1: log scale `0.5 to maxpercentile`
- Panel 2: log scale `0.5 to 200`
- Panel 3: `V/I` (pol ratio), display range `-0.4 to 0.4`
- Output size: `800x900` (W x H), no captions/colorbars

### 3) `ivmsi4`

Multispectral training data (`npz`), 4 channels.

- `msi` array shape: `(4, 300, 800)` = `(NCH, H, W)`
- Channel 0: linear scaling `0-200 sfu`, normalized to `[0, 1]`
- Channel 1: log scaling `0.5-200 sfu`, normalized to `[0, 1]`
- Channel 2: log scaling `0.5-p99.5(I)`, normalized to `[0, 1]`
- Channel 3: `V/I` polarization ratio, clipped to `[-0.4, 0.4]`

## YOLO Labeling Class Mapping

```text
0: Type III
1: Type III-b
2: Type II
3: Type-III-g
4: Type-IV
5: Type V
6: Type U
7: ionospheric caustics
8: corrupted data
```

## TODO

- Make `data_composer` to generate images and multispectral data
- Make `data_loader` to load multispectral data for trainer
