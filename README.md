CIS 565 project 03 : CUDA path tracer
===================

## INTRODUCTION

This project is a CUDA-parallelized Monte Carlo pathtracer implemented for CIS 565 during my fall 2014 semester at Penn. Given a scene file that defines a camera, materials, and geometry, my path tracer is capable of rendering images with full global illumination. Additionally, my path tracer supports Fresnel refraction for glass materials and texture mapping for both spheres and cubes.

## PARALLELIZATION SCHEME

My path tracer is parallelized per ray rather than per pixel. Meaning, at the start of each iteration, one ray is generated for each pixel in the image buffer, and is stored in a ray pool. At each trace depth (basically every time a ray intersects geometry), rays are checked to see if they should be retired from the pool. In my path tracer, rays are retired if they (A) do not intersect with any piece of geometry in the scene, or (B) intersect with a light source. A retired ray is removed from the ray pool and will not be considered during future kernel calls to the GPU.

A per-ray parallelization scheme such as this prevents the unwanted case where some rays in a warp become inactive at a low trace depth while neighboring rays remain active until the max trace depth. In these circumstances, valuable GPU processing time is wasted on inactive rays that no longer contribute to the final rendered image result.

## STREAM COMPACTION

## SUPPORTED GEOMETRY AND MATERIALS

Currently, my path tracer supports sphere and cube geometry and ideal diffuse, perfectly specular, and glass materials. In the future, I plan to support arbitrary mesh objects and more complex BRDF models.

## FRESNEL REFRACTION

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/fresnel_refractions.jpg)

## TEXTURE MAPPING

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/texture_mapping.jpg)

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/texture_mapping_02.jpg)

## JITTERED SUPERSAMPLED ANTI-ALIASING

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/aa_without.jpg)

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/aa_with.jpg)

## PERFORMANCE ANALYSIS

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/chart_stream_compaction_performance.jpg)

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/chart_block_size_comparison.jpg)

## FUTURE WORK

My next steps for this project include implementing:
* Direct lighting samples so my renders converge more efficiently.
* Bump maps.
* Image-based emittance.
* Depth of field.
* An obj loader.

## FUN

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/fun_01.jpg)

![alt tag](https://raw.githubusercontent.com/drerucha/Project3-Pathtracer/master/data/readme_pics/fun_02.jpg)

## SPECIAL THANKS

I want to give a quick shout-out to Patrick Cozzi who led the fall 2014 CIS 565 course at Penn, Harmony Li who was the TA for the same course, and Yining Karl Li who constructed much of the framework my path tracer was built upon. Thanks guys!