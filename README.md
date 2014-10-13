CIS 565 Project3 : CUDA Pathtracer
===================

Fall 2014

Features:

Raycasting from a camera into a scene through a pixel grid
Diffuse surfaces
Perfect specular reflective surfaces
Cube intersection testing
Sphere surface point sampling
Stream compaction optimization

Additional features:
Depth of field
Calculating the ray jittering according to object position and camera position to make the image look like taken from a real camera with a large aperture.
This is a simple calculation run one time per ray. It only adds up several steps of single instructions so there's no noticable change on performance.
Time complexity GPU O(1), CPU O(n)

Refraction, i.e. glass
Calculate the direction of refracted ray (glass effect).
No impact on performance because it changes the direction of a ray instead of shooting new rays. And there is a random number deciding to use reflection or refraction.
Time complexity GPU O(1), CPU O(n)

Motion blur
Altering object's position while rendering a single image
No impact on performance. It simply changes the geometry position before each iteration.
Time complexity GPU O(1), CPU O(1)


