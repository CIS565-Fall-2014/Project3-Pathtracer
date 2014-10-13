CIS 565 Project3 : CUDA Pathtracer
===================

Fall 2014

## PROJECT DESCRIPTION
For this project, I implemented a pathtracer on Nvidia GPU by using CUDA so that the rendering process is very fast.

## FEATURES
basic features:
* Raycasting from a camera into a scene through a pixel grid
* Diffuse surfaces
* Perfect specular reflective surfaces
* Cube&Sphere intersection testing
* Sphere surface point sampling
* Stream compaction optimization 
* 

additional features:
* Depth of field
* Fresnel Refraction and Reflection
* Supersampled antialiasing
* Motion blur

##Implementation
*First version of pathtracer
*

When I finished all the basic functions, the pathtracer gave me this wired output:

The problem is my color accumulation. Since I was simply adding the color for each iteration, everything becomes white. Also the emittance of the light is too large (15) that makes almost every pixel white. I fixed this by:
1. Average color through iterations.  
2. Add a distance feature to the ray and multiply exp(-Distance) to the final pixel color. After doing so, I get this much better result:
