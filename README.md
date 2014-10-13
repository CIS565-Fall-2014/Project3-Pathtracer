CIS 565 Project3 : CUDA Pathtracer
===================
Jiatong He

![alt tag](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/scene2_depth20.0.bmp)

Implemented features:
--------------------
* Raycasting from a camera into a scene through a pixel grid
* Simple diffuse surfaces (no specular highlights)
* Perfect specular reflective surfaces
* Cube intersection testing
* Sphere surface point sampling
* Stream compaction optimization (using Thrust)
* +Interactive camera (pan, zoom, tilt)
* +Depth of field

Nonworking features (base code is there):
-----------------------------------------
* Refraction (need to modify intersection tests?)

> Refraction code has been added to BSDF, but it turns up black.  May be due to intersection tests being incompatible with rays originating from inside the 

![alt tag](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/depth_comparison.bmp)
* A comparison of the same scene with different raytrace depths (4, 7, 20)

Performance
-----------
There are three main contributors to runtime that I would like to focus on:
### 1. Intersection Tests
#### Expectation
I expect that this step takes the longest in the code, and should, ignoring issues with block size/memory, result in linear increases in runtime proportional to the number of geometric bodies in the scene.  This is only with spheres and cubes, and will create significantly more overhead once meshes are implemented.  I believe this should be highest priority for optimization, preferably first with some acceleration structure such as a kd-tree (detailed in "Future Work").
#### Results

### 2. Raytrace Depth
#### Expectation
This is fairly straightforward, but I wanted to see the slope of the runtime increase due to higher raytrace depth.  I would expect that it is slightly less than linear (assuming that the blocks are set up optimally for stream compaction, which they may not be), since the number of threads needed per level decreases every level.
#### Results

### 3. Number of Iterations
#### Expectation
Since this is mostly independent of the GPU (it's based a looped call by the host), it should be expected to be linear.  I want to see if it slows down or speeds up over time, if at all (time/#iterations).

Extra Features
--------------
### Interactive Camera
![video](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/pathtracer_camera_demo.mp4)
* Short video demo of the mouse camera controls.

> see pathtracer_camera_demo.mp4
>
> Use the mouse to control the camera.
>
> Left click-drag will move the camera.
>
> Right click-drag will tilt the camera.
>
> Middle click-drag will move the camera forward/backward.

#### Performance

The mouse control does not actively affect the performance of the raytrace.  However, the lack of acceleration in the raytrace means that what the user sees while moving the camera are
only a few iterations.  The raytrace continuously refreshes while the camera is moved.  Pausing the mouse movement will allow it to continue.  I think that while showing a fully-formed image
during a mouse drag is impractical, faster renders will allow the user to see more clearly-formed images before moving again.

#### Improvement/Optimization

Since this is external user input, I don't think that optimization is an issue.  An improvement over the current system would be to have actual rotation of the camera, which would be more flexible and intuitive.
Currently I am simply adjusting the view vector by some vector coplanar to the view plane rather than rotating it, mainly because glm::rotate gave me really strange errors.

### Depth of Field

> Added a camera field, DEPTH, that defines the focal distance of the camera.  Nonpositive DEPTH values should result in no depth of field.
>
> This was implemented by finding the intersection point with the focal plane for each ray cast from the camera, and then jittering the origin of the ray by some random amount, and updating the direction so that it still passes through that point on the focal plane.

#### Performance
The performance impact of this change is negligible.  It is calculated a single time for each ray cast from the camera, and is far surpassed in runtime by the raytrace itself.

#### Acceleration
None.  I kept the implementation very simple in order to prevent it from affecting runtime while still being visually effective.

#### CPU/GPU comparison
We make resolutionX * resolutionY calculations for each iteration of the path tracing, which is done in parallel on the GPU.  The CPU would need to make those resolutionX * resolutionY calculations sequentially.
Because the resolution is typically low enough, the CPU implementation should not be much slower than the GPU implementation.  However, the calculations needed are simple, so there's no reason to not use the GPU.

#### Improvement
The current random "blur" factor seems to prefer a single direction.  This should be due to the random seed; assuming a uniform distribution from [0,1] on the random number, it should generate an even split.

Debugging
---------
For now, I only have two images to show that I used during debugging:
![normals debug](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/debug_normals.bmp)
* Normals of the planes, +x/y/z = r/g/b respectively.

> The first, and easiest, image to create, other than a basic collision test.  I wanted to make sure that the sphere collision was fully functional and the cube collision I implemented was working.  As it turned out, it wasn't at first, and for some reason that now escapes me, the cubes were being distorted.

![first bounce debug](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/debug_bounce.bmp)
* Single sample of the random rays obtained from the BSDF on the first bounce.

> I created this image because initially, the back white wall was remaning pure white while the bottom and top were receiving global illumination from the red/green walls.  This image suggested that the bounced rays were correct, and I later found the issue in my color equation.

Challenges
----------
### RNG
The random number seed is causing me a ton of trouble.  I'm not certain I fully understand the effect of the seed... certain seeds will completely ruin the image, while others generate strange artifacts.  This is an issue I still have, outlined in the "Problems" section.

Problems
--------
### Artifacts
![alt tag](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/scene1.0.bmp)

* (see the squares on the back wall)
I am fairly certain that these artifacts are either being caused by float precision (though I think I use epsilon equality everywhere) or the random number generator seed.  Putting in a different seed results in vastly different results, some completely wrong.

### Noise (slow to converge)
![alt tag](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/iter_comparison.bmp)
* 20, 200, and 2000 iterations of the raytrace with depth of 7.
This one I'm not sure about.  It could be a matter of simply allowing the rays to be more variable, but ultimately the images appear much more spotty than they should be, even after a reasonable number of iterations.  Making the light larger might solve the problem, but I'm not sure that's the right solution.

### Clipping issues
![alt tag](https://raw.githubusercontent.com/JivingTechnostic/Project3-Pathtracer/master/windows/Project3-Pathtracer/Project3-Pathtracer/scene2.0.bmp)
* (the smaller spheres should be behind the large sphere on the left)
I honestly have no idea what's causing this right now.  I will have to take a look at the code later to figure out what's causing this.  Strangely enough, they do not clip behind the sphere to the right, only the one on the left.  It may be directional, but I doubt that.

Future Work
-----------
#### Priority 0
* Finish refraction
* Fix clipping issues
* Implement meshes
* Implement acceleration data structure
