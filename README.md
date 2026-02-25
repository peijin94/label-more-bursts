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

### 2.1) `ivstack2`

Stacked image of polarization I and V (`png`, `jpg`, etc.).

- Upper panel: log scale `0.5-200 sfu`
- Lower panel: pol-V `-0.5 to 0.5`

### 2.2) `ivstack3`

Stacked image of polarization I and V.

- Panel 1: log scale `0.5 to maxpercentile`
- Panel 2: log scale `0.5 to 200`
- Panel 3: pol-V

### 3) `ivmsi3`

Multispectral training data (`npz`), 3 channels:

- Channel 1: linear scaling `0-200 sfu`, normalized to `[0, 1]`
- Channel 2: log scaling `0.5-200 sfu`, normalized to `[0, 1]`
- Channel 3: pol-V

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
