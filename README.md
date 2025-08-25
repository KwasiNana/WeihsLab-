# PitScan: Automated Detection of Pitting Corrosion from Micro-CT Scans

This repository contains a Python implementation of **PitScan**, an automated image analysis pipeline developed to detect and quantify **pitting corrosion** from micro-computed tomography (µCT) scans of metallic specimens.  

The code is based on the methodology described in the paper:  

> **Automated ex-situ detection of pitting corrosion and its effect on the mechanical integrity of rare earth magnesium alloy - WE43**  
> Kerstin van Gaalen, Felix Gremse, Felix Benn, Peter E. McHugh, Alexander Kopp, Ted J. Vaughan  
> *Bioactive Materials, Volume 8, 2022, Pages 545–558*  
> DOI: [10.1016/j.bioactmat.2021.06.024](https://doi.org/10.1016/j.bioactmat.2021.06.024)

---

## Overview

PitScan systematically evaluates the severity and phenomenology of **pitting corrosion** on µCT-derived images of cylindrical specimens.  
The algorithm enables **3D reconstruction** of pits and extraction of key **ASTM G46-94 corrosion parameters**, including:

- **Number of pits**  
- **Pit density**  
- **Pit depth** (max, average, distribution)  
- **Pitting factor** (deepest/average penetration)  
- **Minimum fitted radius**  
- **Minimum core width**  

Additionally, it generates **surface contour plots** of pit depth and provides a reproducible, non-destructive pipeline for corrosion quantification.

---

## Functionality

The main functionality includes:

1. **Contour detection**  
   - Extracts outer boundary of specimen cross-sections from µCT slices.

2. **Circle fitting & material ratio correction**  
   - Fits an enclosing circle to the contour and iteratively reduces the radius until a 20% material ratio threshold is reached, accounting for uniform degradation.

3. **2D pit detection**  
   - Identifies local deviations from the fitted radius (pits) along radial scan lines at 2° increments.

4. **3D pit reconstruction**  
   - Tracks pits across slices to reconstruct pits in 3D, determining their depth, volume, and spatial distribution.

5. **Quantitative metrics**  
   - Extracts geometric corrosion parameters including **maximum pit depth**, **average pit depth**, **pit density**, and **pitting factor**.

6. **Surface plots**  
   - Produces flattened surface contour plots of pit depth around the cylindrical gauge length for visualization.

---

## Outputs

- **Quantitative corrosion metrics**:  
  `(pit count, maximum pit depth, minimum radius, average radius, average pit depth, pitting factor)`

- **Surface scatter plot**:  
  Color-coded visualization of pit depth around the specimen surface.

---

## Usage

1. Prepare a folder containing µCT slice images (binary masks of the inner magnesium core).  
2. Update the script with your image folder path:
   ```python
   folder_one = "path/to/your/images"

3. Run the script
The program will prompt for voxel size (µm³). Enter based on scan metadata.
Outputs will be printed in the console and visualized via matplotlib.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
