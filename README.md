CIS 565 Project3 : CUDA Pathtracer
===================
Jiatong He

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

Performance
-----------

Extra Features
--------------
### Interactive Camera

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

#### Performance
The performance impact of this change is negligible.  It is calculated a single time for each ray cast from the camera, and is far surpassed in runtime by the raytrace itself.

#### Acceleration
None.  I kept the implementation very simple in order to prevent it from affecting runtime while still being visually effective.

#### CPU/GPU comparison
We make resolutionX * resolutionY calculations for each iteration of the path tracing, which is done in parallel on the GPU.  The CPU would need to make those resolutionX * resolutionY calculations sequentially.
Because the resolution is typically low enough, the CPU implementation should not be much slower than the GPU implementation.  However, the calculations needed are simple, so there's no reason to not use the GPU.

#### Improvement
The current random "blur" factor seems to prefer a single direction.  This should be due to the random seed; assuming a uniform distribution from [0,1] on the random number, it should generate an even split.

Challenges
----------
### RNG
The random number seed is causing me a ton of trouble.  I'm not certain I fully understand the effect of the seed... certain seeds will completely ruin the image, while others generate strange artifacts.  This is an issue I still have, outlined in the "Problems" section.

Problems
--------
* Artifacts
* Noise (slow to converge)
* Clipping issues