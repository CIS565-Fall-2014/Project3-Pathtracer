CIS 565 Project3 : CUDA Pathtracer
===================

Fall 2014


## PROJECT DESCRIPTION
In this project, I implement a CUDA based pathtracer capable of
generating pathtraced rendered images extremely quickly. 

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

## RUNNING THE CODE
This project is tested in Visual Studio 2010 with CUDA 6.5.
The main function requires a scene description file (that is provided in data/scenes). 
You can change the scene file in command arguments section.


## IMPLEMENTATION
* Color Accumulation
I have touble accumulate color when I was testing my diffuse surface, here is what I got at first diffuse rendering:

Later I found out the color contribution of each iteration should be different, then I divide the color with 1/iteration and got much more reasonable result: 

* Stream Compaction
This path tracer is parallelized by ray. When parallelizing by pixel, some ray paths die out before the maximum ray depth is reached (either from absorption or lack of intersection), and we can get rid of these "dead" rays using stream compaction

This implementation uses a pool of rays, each ray has a flag indicating if the ray is alive or dead. With each new wave of raycasts, the current active rays are pulled from the pool and cast into the scene. Depending on whether or not geometry was intersected, the ray is marked as inactive or active. After each wave of raycasts, I use thrust scan and scatter funstion to cull dead rays and get a new raypool with smaller size. 

* Fresnel Transparency
I implement fresnel transparency to make the glass material looks more realistic. The implementation is basically the same as I did in CIS560 ray tracer project: first calculate the fresnel coefficients and decide if the ray is a reflect ray or refract ray by the random number portion.

* Depth of Field
Depth of Field is implemneted by selecting a depth plane away from the camera and jittering the camera position during each iteration.