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

The first feature that I supported was generation of the original set of rays, and casting them into the screen once. I then colored the pixel according to whatever material was hit. This is a good starting place and sanity check for any raytracing work. This helped me work out many small bugs such as discrepancy in units between my work and the starter code (degrees/radians, etc.). The result is shown here:

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/CornellBoxesSimpleColoring.png" "Validation of Initial Raycasting">

The next step was to validate all of my collision calculations. I now rendered to the screen a grayscale image the was colored by the depth of the initial ray hit. Note: my collision calculations are based upon methods in "Real-Time Collision Detection" by Christer Ericson.

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/Depth Cube Intersection.png" "Collision Depth Evaluation">

I then started to evaluate path-tracing with a few iterations. The initial results seemed dim, and there were strange artifacts where I could see through some of the shapes in the foreground.

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/Increased Light.png" "Initial Pathtracing - Many Bugs">

One of the issues that I tracked down in this implementation was that the collision detection was returning with the first hit, and not the closest hit. This made the result dependent on the ordering of the objects in the config files. This was causing a large number of rays to escape the box and make the scene look dim. Fixing this made the impact shown here (notice still strange artifacts):

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/Fixed Bug with Ordering.png" "Pathtracing - Corrected Collision Ordering">

Next, I found another issue that the normal used from the collision check was the most recent collision normal, not the one corresponding to the closest collision. Fixing this issue made this result:

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/Cleaned up normal calculation.png" "Pathtracing - Corrected Normal Ordering">

At this point, the diffuse and specular materials were both looking pretty good. However, the top of the scene looks much brighter than I would have expected. I found one last bug where I was getting incorrect normals from my box intersections when the intersection was near an edge. Tightening up the bounds, and enforcing the collision to be associated with only a single face resulted in the following:

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/fixed bug with normals.png" "Functional Pathtracing">

Stream Compaction - I have a custom stream compaction algorithm utilizing scan-then-fan (described in "The CUDA Handbook" by Nicholas Wilt, Chapter 13.4) that I attempted to utilize for throwing out terminated rays at each iteration. However, I found that this method needs to be written to utilize multiple blocks correctly. Thus, it only seems to work on one block. To get something working, I hooked up the copy_if method provided in Thrust for stream compaction. The performance impact is analyzed below. 

The next feature that I added was refractive surfaces. I performed this calculation with basic Snell's law, and only allowed materials to be fully refractive with no reflection. Here is an example of a blue glass sphere using this:

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/another blue sphere.png" "Blue Glass Sphere">

Another interesting image is what the scene looks like with only 2 bounces per ray:

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/depth 2.png" "Max Depth 2">


##Performance Analysis

Stream Compaction - I compared the frame time difference between a brute force method of just no-oping in the kernel for all removed rays, and a compacted version with thrust that does not incur the overhead of starting up unneeded kernels. The scene for these tests was the Cornell blocks with 3 glossy spheres. The results show that past a depth of 4 bouncers per ray, the stream compaction starts to build a significant speed boost. At a depth of 2, however, the additional overhead of compacting the rays is not worthwhile, mostly because very few (if any) rays are terminated after the first bounce. Both methods start to cap off at 50 bounces, as most rays have likely left the scene by this point. The brute force method has less overhead than usual here since most warps have all threads go through the same early termination.

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/stream compaction performance.png" "Stream Compaction Impact">

For a single iteration, I also captured the number of rays at each trace depth. The result appears to support my theory.

<img src="https://raw.github.com/dkotfis/Project3-Pathtracer/master/images/active rays.png" "Active Rays">

Meshes

##Future

My original intention was to extend the mesh representation to use an octree for spatial subdivision. Due to lack of time, I did not implement this (it required working tri-meshes first in order to show benefit). I will likely come back to implementing this on a later project.

Some of the research that this will/would draw upon include:
- "Out-of-Core Construction of Sparse Voxel Octrees" - Baert, Lagae, and Dutre. This is a method that can build octrees from tri-meshes of arbitrary size, and can utilize the GPU. 
- "An Efficient Parametric Algorithm for Octree Traversal" - Revelles, Urena, and Lastra. This investigates methods for getting the leaves intersecting a ray.
- "GigaVoxels: Ray-Guided Streaming for Efficient and Detailed Voxel Rendering" - Crassin et. al. This is an impressive GPU based approach to raycasting large N^3 trees.

