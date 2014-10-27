CIS 565 Project3 : CUDA Pathtracer
===================

## INTRODUCTION
In this project, I implemented a CUDA based pathtracer capable of
generating pathtraced rendered images extremely quickly. 

## Features 
I implemented the following basic features:

* Raycasting from a camera into a scene through a pixel grid
* Diffuse surfaces
* Perfect specular reflective surfaces
* Cube intersection testing
* Sphere surface point sampling
* Stream compaction optimization

And the following two extra features:

* Depth of field
* Refraction, i.e. glass

## Performance 
I implemented path tracing with pixel parallelization and ray parallelization.

* Pixel parallelization

In each bounce, allocate a thread for each pixel. It means there are 800*800 threads each loop in this test case. There may be many wasted cycles, some rays have been killed when hitting the light or nothing.

* Ray parallelization

In each bounce, allocate a thread for each ray, instead of pixel. Construct a pool of rays; and in each bounce, removed terminated rays from the pool. Here, we can use stream compaction to just keep the active rays.

The following chart shows the number of rays and timing in each bounce. Each bounce will have fewer active rays, require fewer blocks and run faster.

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/table.JPG)

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/RaysPerBounce.JPG)

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/TimingPerBounce.JPG)

Here is the timing comparation between pixel parallelization and ray parallelization. From the chart, the ray parallelization is of higher efficiency.

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/SC.JPG)


## Render Images
* Diffuse surface

The following image shows diffuse surface with soft shadow.

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/sample.PNG)

* Frenel

The following image shows reflective and refractive surface. The left one is glass-like material with perfect refraction; and the right one is mirror-like material with perfect reflection.

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/fresnel.PNG)

* Depth of Field

Depth of field refers to the range of distance that appears acceptably sharp. Here're the images of different focal length and lens.

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/DOF14_0.5.PNG)

Focal length is 14 and lens is 0.5.

## Optimization
When implementing the ray-box intersection, I used the following method first. But it runs pretty slow.
![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/code%20v2.PNG)

Then I change to the following version two. It runs 10 times faster. In this version, I used a boolean variable to store whether the ray is intersected with the box, and only one return at the end of the function. 

![ScreenShot](https://github.com/liying3/Project3-Pathtracer/blob/master/img/code%20v1.PNG)