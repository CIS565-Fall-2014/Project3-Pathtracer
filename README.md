CIS 565 Project3 : CUDA Pathtracer
===================

Fall 2014

Author: Dave Kotfis

##Overview

This is a GPU pathtracing project utilizing starter code providing much of the backend (scene file loading, CUDA/OpenGL interop, etc). Some of the features include:
- Shapes (Boxes, Spheres, Meshes)
- Materials (Diffuse, Specular, Refractive)
- Stream Compaction (Thrust, Scan+Fan w/ limited support)

##Progress

<img src="https://github.com/dkotfis/Project3-Pathtracer/master/images/Cornell Boxes Simple Coloring" "Validation of Initial Raycasting">

##Performance Analysis

##Future

My original intention was to extend the mesh representation to use an octree for spatial subdivision. Due to lack of time, I did not implement this (it required working tri-meshes first in order to show benefit). I will likely come back to implementing this on a later project.

Some of the research that this will/would draw upon include:
- "Out-of-Core Construction of Sparse Voxel Octrees" - Baert, Lagae, and Dutre. This is a method that can build octrees from tri-meshes of arbitrary size, and can utilize the GPU. 
- "An Efficient Parametric Algorithm for Octree Traversal" - Revelles, Urena, and Lastra. This investigates methods for getting the leaves intersecting a ray.
- "GigaVoxels: Ray-Guided Streaming for Efficient and Detailed Voxel Rendering" - Crassin et. al. This is an impressive GPU based approach to raycasting large N^3 trees.

