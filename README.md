# Refrigitz Graphic – Olympic Edition: A Novel 2D‑to‑3D Conversion & Hollow Shell Regression Framework

**Authors:** tetrashop  
**Affiliation:** Independent Researcher, Refrigitz Project  
**Date:** June 2026

---

## Abstract

This paper presents **Refrigitz Graphic**, an advanced web‑based framework for converting single 2D images into photorealistic 3D models and providing an interactive 3D studio environment. The core contribution lies in a novel **Symmetric Hollow Regression** algorithm that reconstructs a closed, manifold shell mesh from a single RGB image while preserving the exact visual appearance of the input. Unlike conventional height‑map or depth‑estimation methods that produce flat or spike‑prone surfaces, our approach builds a smooth, evenly triangulated hollow shell whose centroid lies precisely at the mean depth of the object. The system achieves high fidelity through vertex‑color mapping, robust edge‑preserving smoothing, and a lightweight serverless architecture deployable on Vercel. The paper details the theoretical foundations, iterative development, encountered challenges, and the innovative solutions that led to a bug‑free, production‑ready application.

---

## 1. Introduction

Reconstructing 3D geometry from a single image is a fundamental problem in computer vision and graphics. Traditional monocular depth estimation techniques rely on deep neural networks, which are computationally expensive and often inaccessible in resource‑constrained environments. Refrigitz Graphic takes a different route: it leverages classical image‑processing algorithms (spherical mapping, gradient‑based surface integration, and Delaunay‑like triangulation) combined with a novel symmetric hollow‑shell regression to generate a closed 3D mesh. The project evolved through extensive debugging and theoretical refinement, ultimately delivering a robust API and an interactive web interface.

---

## 2. Theoretical Foundations

### 2.1 2D‑to‑3D Image Mapping (PNG Output)
The initial stage of the pipeline produces a “3D‑fied” 2D image using a spherical coordinate remapping inspired by the original Refrigitz C# codebase. Each pixel at coordinates \((x,y)\) is transformed into spherical coordinates:

\[
r = \sqrt{(x - W/2)^2 + (y - H/2)^2 + 1}, \quad
\theta = \arccos(1/r), \quad
\phi = \arctan2(y - H/2, x - W/2)
\]

A radial displacement \(dr\) is computed based on pixel luminance to simulate depth. The displaced points are accumulated in a target matrix \(c\), and then read back to form the output image \(e\) of width \(2W\). This method creates a parallax‑like effect but does not generate a true 3D model.

### 2.2 Surface‑from‑Gradients (Shape‑from‑Shading)
To obtain a continuous surface, we initially experimented with the Frankot‑Chellappa algorithm (Fourier‑domain integration). However, due to timeout constraints on Vercel (10 s limit), we adopted a fast cumulative‑sum approach:

\[
Z(x,y) = \frac{1}{2}\left( \sum_{x} \frac{\partial z}{\partial x} + \sum_{y} \frac{\partial z}{\partial y} \right)
\]

This yields a smooth height map while preserving edge details through a Bilateral Filter.

### 2.3 Symmetric Hollow Shell Regression (Core Innovation)
Instead of a simple height‑map with a fixed back plane, we introduce **Symmetric Hollow Regression**. Given the front‑surface depths \(Z_{\text{front}}(x,y)\), the back surface is defined as the mirror reflection about the mean depth:

\[
Z_{\text{back}}(x,y) = 2 \cdot \overline{Z} - Z_{\text{front}}(x,y)
\]

where \(\overline{Z}\) is the average of all \(Z_{\text{front}}\) values. This guarantees that the centroid of the entire mesh lies exactly at the mean depth, creating a physically plausible hollow shell with variable thickness. Side walls are stitched to form a watertight manifold, ensuring outward‑facing normals.

### 2.4 Vertex‑Color Fidelity
Each vertex is assigned the exact RGB color of the corresponding input pixel. This enables photorealistic rendering from the front view while maintaining the 3D structure when rotated.

---

## 3. Innovations and Contributions

1. **Symmetric Hollow Regression:** A novel lightweight method to convert a single depth map into a closed, manifold 3D shell without deep learning.
2. **AI‑Driven Debugging Process:** The development involved iterative AI‑assisted analysis to identify root causes of bugs (e.g., array index mismatches, timeout issues, inverted depth). This “AI purification” loop is documented as a methodology for robust software engineering.
3. **Edge‑Preserving Smoothing:** Replaced Gaussian blur with Bilateral Filtering to eliminate spikes (“young mountains” effect) while retaining sharp features.
4. **Adaptive Sampling & Uniform Grid:** Early experiments with random weighted sampling and Delaunay triangulation were eventually replaced by a fixed uniform grid (80×80) that ensures regular, non‑degenerate triangles, improving visual quality and stability.
5. **Serverless Optimization:** All algorithms were tuned to execute within Vercel’s 10‑second timeout, including image downscaling, efficient NumPy operations, and a fallback mechanism that returns the original image on failure.
6. **Integrated Web Studio:** A single‑page application combining 3D model viewer (Three.js), 2D paint module, primitive object insertion, and real‑time conversion.

---

## 4. Challenges and Solutions

| Challenge | Attempted Solutions | Final Solution |
|-----------|-------------------|----------------|
| Depth inversion (dark = high) | Manual `1-depth`, `invert` parameter | Use edge magnitude + symmetric regression to eliminate directionality |
| Spiky “young mountain” surfaces | Reduce Gaussian σ, increase point density | Bilateral filter with σ_spatial=3, uniform grid |
| “Hill‑effect” (loss of details) | Unsharp masking with various weights | Symmetric regression retains details via vertex colors |
| `Failed to fetch` / timeout | Reduce grid size, remove heavy loops | `grid_res=80`, fast cumulative integration, fallback PNG |
| `cannot reshape array` error | Separate RGB image for colors, grayscale for depth | Two separate image objects (RGB + L) |
| `list index out of range` in shape drawing | None | Parameter length validation with informative error messages |
| CORS errors | Manual header insertion | `flask-cors` library |
| `No module named 'core._2d_to_3d'` | Rename file | `converter.py` (files cannot start with a digit) |
| Object not hollow / normals incorrect | Fixed back plane, back‑face winding | Symmetric mirroring + edge stitching + normal averaging |

---

## 5. System Architecture

```

refrigitz-api/
├── api/
│   └── index.py              # Flask API endpoints
├── core/
│   ├── converter.py          # Spherical mapping for PNG output
│   ├── obj_generator.py      # Symmetric hollow shell generation
│   ├── shape_drawing.py      # 2D shape drawing (Line, Arc, Bezier, etc.)
│   ├── line.py, point3d.py, triangle.py, improvment_sort.py, interpolate.py
├── index.html                # Web GUI (3D viewer + paint + conversion)
├── requirements.txt          # Python dependencies
├── README.md                 # This paper
└── LICENSE.md                # MIT License

```

---

## 6. Usage

### 6.1 API Endpoints

- **Health check:** `GET /api/health`
- **2D‑to‑3D PNG:** `POST /api/2d-to-3d?format=png` with JSON `{"image": "<base64>"}`
- **2D‑to‑3D OBJ:** `POST /api/2d-to-3d?format=obj&invert=false&height=40` – returns a Wavefront OBJ with vertex colors.
- **Draw shape:** `POST /api/draw-shape` with JSON `{"shape": "rectangle", "color": "#ff0000", "params": [50,50,200,150]}`

### 6.2 Web Interface
Visit the root URL to access the full studio: upload an image, adjust height and inversion, generate the OBJ model (which loads directly into the 3D viewer), paint on a canvas, apply textures, and manipulate 3D primitives.

---

## 7. Deployment on Vercel

1. Push the repository to GitHub.
2. Import the project into Vercel.
3. The platform automatically detects Flask and installs dependencies from `requirements.txt`.
4. The API becomes available at `https://<your-project>.vercel.app`.

---

## 8. Conclusion

Refrigitz Graphic demonstrates that classical computer vision algorithms, combined with careful engineering and iterative AI‑assisted debugging, can produce a robust 3D reconstruction tool without relying on deep neural networks. The introduced Symmetric Hollow Regression offers a novel way to generate watertight 3D shells from single images, and the integrated web studio makes the technology accessible to everyone. The project is released under the MIT License to encourage further research and development.

---

*“Beyond Olympic – Bug‑Free, Resilient, Beautiful.”*
