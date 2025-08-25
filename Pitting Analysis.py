#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitting Analysis - Annotated Version (PitScan Framework)

This script implements framework similar to the "PitScan" framework described in:
    van Gaalen K. et al. (2022) 
    "Automated ex-situ detection of pitting corrosion and its effect on the 
     mechanical integrity of rare earth magnesium alloy - WE43"
    Bioactive Materials, 8, 545–558. https://doi.org/10.1016/j.bioactmat.2021.06.024

Purpose:
--------
Automated detection and quantification of pitting corrosion 
from micro-CT images of cylindrical specimens.

Main Features:
--------------
- Circle fitting with material ratio correction
- 2D pit detection (radial scans every 2°)
- 3D pit reconstruction across slices
- Extraction of key ASTM G46-94 corrosion parameters:
    * pit count
    * pit density
    * pit depth (max/avg/distribution)
    * pitting factor
    * min core width, fitted radius
- Surface scatter plots for visualization

Dependencies:
-------------
numpy, opencv-python, matplotlib, multiprocessing
"""

import cv2 as cv
import numpy as np
import math 
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from xml.dom import minidom


class pitwork:
    def __init__(self, images, initial_radius=0):
        """
        Initialize PitScan analysis.

        Args:
            images (list or np.array): paths to µCT slice images (binary masks of core).
            initial_radius (float): optional known initial specimen radius (µm).
        """
        self.voxel_size = input("Enter voxel size (µm³): ")   # needed to scale results
        self.images = images
        self.initial_radius = initial_radius
        self.pits_3d = None

    # --------------------------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------------------------

    def get_voxel_size(self, xml_file):
        """
        Extract voxel size from scanner metadata (XML).
        Returns product X*Y*Z (µm³).
        """
        file = minidom.parse(xml_file)
        models = file.getElementsByTagName('voxelSize')
        x = float(models[0].attributes['X'].value) 
        y = float(models[0].attributes['Y'].value) 
        z = float(models[0].attributes['Z'].value)
        return x * y * z

    def find_fitted_radius(self, image, material_ratio_threshold=0.2):
        """
        Fit a circle to the specimen contour, then iteratively shrink radius 
        until the 'material ratio' reaches threshold (~20%).

        This corrects for uniform degradation, leaving local pits as deviations.

        Returns:
            radius (float): corrected radius
            contours (list): detected specimen contours
            center (tuple): (x,y) circle center
        """
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        img_blur = cv.GaussianBlur(img, (5,5), 0)

        contours, _ = cv.findContours(img_blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # choose the largest contour
        contour = max(contours, key=len)

        (x,y), radius = cv.minEnclosingCircle(contour)

        # iterative shrink until material ratio satisfied
        while True:
            adjusted_circumference = 2 * np.pi * radius
            center = (int(x), int(y))

            blank = np.zeros_like(img, dtype="uint8")
            mask = cv.circle(blank, center, int(radius), 255, -1)

            covered = cv.bitwise_and(img, img, mask=mask)
            intersections = np.count_nonzero(covered)

            material_ratio = intersections / adjusted_circumference

            if material_ratio >= material_ratio_threshold or radius < 10:
                break
            radius -= 1

        return radius, contours, (x,y)

    # --------------------------------------------------------------------------
    # 2D PIT DETECTION
    # --------------------------------------------------------------------------

    def sample_cont(self, fitted_radius, center, mask, img):
        """
        Radially sample the specimen contour at 2° increments.
        Returns pixel coordinates representing deviations from fitted radius.
        """
        pixel_sum = []
        c = []

        for theta in range(0, 360, 2):
            # border point on circle
            bx = round(fitted_radius*np.cos(theta*np.pi/180) + center[0])
            by = round(fitted_radius*np.sin(theta*np.pi/180) + center[1])
            dir_v = np.array([bx-center[0], by-center[1]])
            n_v = dir_v / np.linalg.norm(dir_v)

            # step outward until mask edge found
            check = [round(center[1]), round(center[0])]
            for i in range(round((fitted_radius+20)/np.linalg.norm(n_v))):
                if int(mask[check[0], check[1]]/255) != 0:
                    check = [round(center[1]+ (i*n_v[1])), round(center[0] + (i*n_v[0]))]
                else:
                    try:
                        pixel_sum.append([check[0], check[1]])
                        c.append([[check[0], check[1]]])
                    except IndexError:
                        pass
                    break
        return pixel_sum, (np.array(c),)

    def detect_pits_2d(self, img):
        """
        Detect pits on a single µCT slice.

        Returns:
            numpy array: [depth deviation, pit mask, fitted radius, angle, fitted radius]
        """
        image = cv.imread(img, cv.IMREAD_GRAYSCALE)
        img_blur = cv.GaussianBlur(image, (5,5), 0)

        # threshold to get binary mask
        avg_t = min(img_blur[img_blur > 0])
        _, mask = cv.threshold(img_blur, avg_t, 255, cv.THRESH_BINARY)

        # fitted radius & center
        fitted_radius, _, center = self.find_fitted_radius(img)

        # radial sampling
        points = self.sample_cont(fitted_radius, center, mask, img)[0]

        distances = np.zeros(180)
        diff = np.zeros(180)
        pit_mask = np.zeros(180)
        giant_A = []

        pit_mode = False
        for i, p in enumerate(points):
            distances[i] = math.dist(p, center)
            diff[i] = abs(distances[i] - fitted_radius)

            # pit if local inward deviation
            if (fitted_radius - diff[i]) < 0.98*fitted_radius:
                pit_mode = True
                pit_mask[i] = 1
            elif pit_mode and diff[i] < diff[i-1]:
                pit_mode = False
            elif pit_mode:
                pit_mask[i] = 1

            giant_A.append([diff[i], pit_mask[i], fitted_radius, int(2*i), fitted_radius])
        return np.array(giant_A)

    # --------------------------------------------------------------------------
    # 3D PIT RECONSTRUCTION
    # --------------------------------------------------------------------------

    def process(self, file):
        """ Helper for multiprocessing pit detection across slices. """
        pit = self.detect_pits_2d(file)
        pit[:,2] = int(self.images.tolist().index(file))  # slice index
        return pit

    def reconstruct_pits_3d(self):
        """
        Reconstruct pits in 3D by stacking slice data.

        Returns:
            tuple: (count, depth_max, min_radius, avg_radius, avg_pit_depth, pitting_factor)
        """
        with Pool(cpu_count()) as pool:
            data = pool.map(self.process, self.images)

        A = np.stack(data)
        C = A[:,:,1]
        B = A[C == 1]

        min_rad = min(B[:,-1]*float(self.voxel_size))
        avg_rad = np.average(B[:,-1]*float(self.voxel_size))

        # pit depth statistics
        depth = []
        count = 0
        for i in range(0,360,2):
            base = B[B[:, -2] == i]
            threshold = 50/1500
            in_group = False
            for num in base[:,0]:
                if num > threshold:
                    if not in_group:
                        count += 1
                        depth.append(num)
                        in_group = True
                else:
                    in_group = False

        avg_p_depth = np.average(depth)*float(self.voxel_size)
        depth_max = max(depth)*float(self.voxel_size)
        pitting_factor = depth_max/avg_p_depth if avg_p_depth > 0 else None

        return count, depth_max, min_rad, avg_rad, avg_p_depth, pitting_factor

    # --------------------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------------------

    def get_surface_plot(self):
        """
        Generate a 2D scatter plot of pit depth vs specimen surface coordinates.
        """
        with Pool(cpu_count()) as pool:
            data = pool.map(self.process, self.images)

        A = np.stack(data)
        angle = A[:,:,-2].flatten()
        depth = (A[:,:,-1] - A[:,:,0]).flatten()*float(self.voxel_size)
        circumference = ((angle/360)*(2*np.pi*np.average(A[:,:,-1].flatten()))) * float(self.voxel_size)
        z = A[:,:,2].flatten()*float(self.voxel_size)

        plt.figure(figsize=(6,10))
        plt.scatter(circumference, z, c=depth, cmap='jet_r')
        cbar = plt.colorbar()
        cbar.set_label('Pit depth (µm)', fontsize=14)
        plt.xlabel('Circumference (µm)', fontsize=12)
        plt.ylabel('Length (µm)', fontsize=12)
        plt.title('Surface Scatter Plot of Pit Depth', fontsize=14)
        plt.show()


# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
if __name__ == "__main__":
    folder = "F:/WIRE CT/WIRE PAPER/ZXM DMEM NO D_3 NUM21/Masked num 21"
    img_path = os.listdir(folder)
    images = np.array([os.path.join(folder, i) for i in img_path])

    wire = PitScan(images)

    # Surface plot
    wire.get_surface_plot()

    # 3D pit reconstruction
    pit_3d = wire.reconstruct_pits_3d()
    print("3D Pit Results (count, depth_max, min_rad, avg_rad, avg_p_depth, pitting_factor):")
    print(pit_3d)
