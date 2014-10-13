CIS 565 Project3 : CUDA Pathtracer
===================
Jiatong He

Implemented features:
* Raycasting from a camera into a scene through a pixel grid
* Simple diffuse surfaces (no specular highlights)
* Perfect specular reflective surfaces
* Cube intersection testing
* Sphere surface point sampling
* Stream compaction optimization (using Thrust)
** Interactive camera (pan, zoom, tilt)
** Depth of field

Nonworking features (base code is there):
* Refraction (need to modify intersection tests?)

Performance

Extra Features
* Interactive Camera
> Use the mouse to control the camera.
>>Left click-drag will move the camera.
>>Right click-drag will tilt the camera.
>>Middle click-drag will move the camera forward/backward.
* Depth of Field

Challenges

Problems
