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
![ScreenShot](https://raw.githubusercontent.com/CyborgYL/Project3-Pathtracer/master/result/dof.PNG)
Calculating the ray jittering according to object position and camera position to make the image look like taken from a real camera with a large aperture.
This is a simple calculation run one time per ray. It only adds up several steps of single instructions so there's no noticable change on performance.
Time complexity GPU O(1), CPU O(n)

Refraction, i.e. glass
![ScreenShot](https://raw.githubusercontent.com/CyborgYL/Project3-Pathtracer/master/result/pathtracer.PNG)
Calculate the direction of refracted ray (glass effect).
No impact on performance because it changes the direction of a ray instead of shooting new rays. And there is a random number deciding to use reflection or refraction.
Time complexity GPU O(1), CPU O(n)

Motion blur
![ScreenShot](https://raw.githubusercontent.com/CyborgYL/Project3-Pathtracer/master/result/motion.PNG)
Altering object's position while rendering a single image
No impact on performance. It simply changes the geometry position before each iteration.
Time complexity GPU O(1), CPU O(1)

Stream Compaction
One iteration time measuring
![ScreenShot](https://raw.githubusercontent.com/CyborgYL/Project3-Pathtracer/master/result/performance.PNG)
Because I'm using a  GK110 Kepler architecture GPU, I put the loop inside __global__ function instead of calling the kernel in a loop at the host program. On the other side, stream compaction needs to call kernel from host every ray so there is huge expense in doing that for each pixel and each ray.
