CIS 565 Project3 : CUDA Pathtracer
Features implemented:

* Raycasting from a camera into a scene through a pixel grid
* Diffuse surfaces
* Perfect specular reflective surfaces
* Cube intersection testing
* Sphere surface point sampling
* Stream compaction optimization

Extra Features:
* Depth of field
* Refraction, i.e. glass
* OBJ Mesh loading and rendering

Third parth software:

*I used tinyobjloader for OBJ mesh loading. 

Still working on:
*texture mapping 
*subsurface scattering

The stream compaction won't affect performance very much unless the depth is too high. For OBJ mesh rendering, it cosumes too much GPU memory and performance for intersection test. To faster OBJ mesh rendering, we need extra bounding box technique to reduce the intersection test, such as AABB bounding box. 