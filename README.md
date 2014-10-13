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
** Interactive camera (pan, zoom, tilt)
** Depth of field

Nonworking features (base code is there):
-----------------------------------------
* Refraction (need to modify intersection tests?)
> Refraction code has been added to BSDF, but it winds up black.  May be due to intersection tests being incompatible with rays originating from inside the 

Performance
-----------

Extra Features
--------------
* Interactive Camera

> Use the mouse to control the camera.
> Left click-drag will move the camera.
> Right click-drag will tilt the camera.
> Middle click-drag will move the camera forward/backward.

* Depth of Field

> Added a camera field, DEPTH, that defines the focal distance of the camera.  Nonpositive DEPTH values should result in no depth of field.

Challenges
----------

Problems
--------
* Artifacts
* Noise (slow to converge)
* Clipping issues