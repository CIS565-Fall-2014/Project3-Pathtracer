CIS 565 Project3 : CUDA Pathtracer
===================

## INTRODUCTION

## PARALLELIZATION SCHEME

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