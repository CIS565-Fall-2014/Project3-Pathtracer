CIS 565 Project3 : CUDA Pathtracer
===================

## INTRODUCTION
In this project, I implemented a CUDA based pathtracer capable of
generating pathtraced rendered images extremely quickly. 

## Performance 
I implemented path tracing with pixel parallelization and ray parallelization.

* Pixel parallelization

In each bounce, allocate a thread for each pixel. It means there are 800*800 threads each loop in this test case. There may be many wasted cycles, some rays have been killed when hitting the light or nothing.

* Ray parallelization

In each bounce, allocate a thread for each ray, instead of pixel. Construct a pool of rays; and in each bounce, removed terminated rays from the pool. Here, we can use stream compaction to just keep the active rays.

The following chart shows the number of rays and timing in each bounce. Each bounce will have fewer active rays, require fewer blocks and run faster.

![ScreenShot]()

Here is the timing comparation between pixel parallelization and ray parallelization. From the chart, the ray parallelization is of higher efficiency.

![ScreenShot]()


## Features
* Diffuse surface

The following image shows diffuse surface with soft shadow.

![ScreenShot]()

* Frenel

The following image shows reflective and refractive surface.

![ScreenShot]()