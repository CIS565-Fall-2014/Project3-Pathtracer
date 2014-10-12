CIS 565 Project 3: CUDA Pathtracer
==================================

* Kai Ninomiya (Arch Linux/Windows 8, Intel i5-4670, GTX 750)


Keybindings
-----------

* Escape: save image and exit program
* Space: save image early and continue rendering
* Arrow keys: rotate camera
* WASDRF keys: fly through space laterally (WASD) and vertically (RF) relative
  to the camera orientation


Base Code Features
------------------

* Configuration file reading
* Hemisphere sampling function (for diffuse)
* Objects: sphere


Features Implemented
--------------------

* Pathtracing algorithms
* Materials: diffuse, reflective, *refractive with Fresnel reflection*
* Camera: *movement* (controls above), *antialiasing*, *depth of field*
* Objects: cube, sphere *with correct normals when scaled*
* Performance: ray-level stream compaction

(Extras in *bold*.)


Renderings
----------

TODO


Performance
-----------

### Stream Compaction

In order to perform ray-level stream compaction, it was necessary to refactor
the rendering kernel into a single-ray step along the path. The result of this
is significantly more overhead, due primarily to performing stream compaction
between every step. At low path depths (e.g. 4), performance is lower with
stream compaction. However, stream compaction allows for extremely high path
depths (tested up to 1000) without very significant performance degradation.
This is because the vast majority of paths have terminated, and dead paths no
longer use kernel threads.

TODO: numbers

### Block sizes (with compaction)

TODO: graph

### Cube Intersection

Initially I did cube intersection naively, but this turned out to use way too
many GPU registers and had very bad performance. Rewriting based on Kay and
Kayjia's slab method reduced register usage by around 50 registers.


Extras
------

### Antialiasing

Samples are taken randomly from within each pixel.

*Performance:* Very small impact. TODO

### Depth of Field

Origin and direction of camera rays is varied randomly (in a uniform circular
distribution) to emulate a physical aperture.

*Performance:* Very small impact per-sample, but increases the number of
samples needed for visual smoothness due to the extreme variation between
samples. Implementation-wise, this is identical to analogous CPU code. TODO

### Fresnel Reflection/Refraction

I used Schlick's approximation to compute the fractions of light
reflected/refracted, then used that as a probability for the next path ray.

Reflection is implemented using glm::reflect.  Refraction uses glm::refract and
handles total internal reflection. Intersection code was modified to report
whether the intersection was inside or outside the object, which allows correct
handling of indices of refraction at interfaces. (This technically could have
been done by adopting a different normal direction convention and checking
dot products with that, but this is more readable.)

TODO: debug render here

*Performance:* TODO

### Camera movement

Keys for this are listed in the Keybindings section. This is implemented by
simply modifying the location of the camera, clearing the render, and starting
again.

### Scaled Sphere Normals

This is a minor thing, but I fixed the provided code to use inverse transpose
transformations to calculate the sphere normals.


Parameter Comparison Renderings
-------------------------------

TODO (compare sample counts and iteration counts)


Debug Renders
-------------

TODO


Bloopers
--------

TODO
