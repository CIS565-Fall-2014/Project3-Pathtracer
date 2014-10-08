

CUDA Pathtracer
===============

Fall 2014
=========

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_AllEffects.png)

PROJECT DESCRIPTION
-------------------

This is a GPU path tracing program, with following features:

###Basic Features:
- Raycasting from a camera into a scene through a ray grid
- Diffuse surfaces
- Perfect specular reflective surfaces
- Cube intersection testing
- Sphere surface point sampling
- Stream compaction optimization

###Extra Features:
- Super sample anti-alisasing
- Refraction, i.e. glass
- OBJ Mesh loading and rendering
- Motion blur
- Bump mapping
- Texture mapping 
- Depth of field
- Interactive camera


####1.Super sample anti-alisasing
- Reference: http://en.wikipedia.org/wiki/Supersampling

- Overview write up and performance impact:
  
I add super sample anti-alisasing, which makes my render result smoother. To do this, I just jitter the initial rays randomly in each iteration. And here is the comparison between the scene with SSAA and the scene without SSAA:

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/DetailWithSSAA.bmp)

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/DetailWithNoSSAA.bmp)

- Accelerate the feature: NULL

- Compare to a CPU version: 
  
In CPU version we may use grid algorithm to jitter the initial rays, but in GPU version we jitter the rays randomly. Their results are almost the same.

- Further optimized: NULL


####2.Refraction
- Reference: http://en.wikipedia.org/wiki/Fresnel_equations

- Overview write up and performance impact:
  
I add fresnel reflection and refraction. And it enables me to add transparent objects in my scene. To do this, I just use the fresnel equations to compute the reflective and refractive coefficients whenever the ray hits a refractive object, and get the reflect ray and refract ray. However, as my cuda path tracer works iteratively, I can just return one ray each time. So I generate a random number to decide which ray to return, based on the reflective and refractive coefficients. And here the  upper left sphere is refractive:

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_WithRefraction.png)

- Accelerate the feature: NULL

- Compare to a CPU version: 
  
I think the main difference with CPU version is that I use a random number to decide which ray to pass on. But in CPU version(recursive), we pass both reflect ray and refract ray  and add their results together. However, I think the result is the same.

- Further optimized: NULL


####3.OBJ Mesh loading and rendering
- Reference: http://www.cplusplus.com/forum/general/87738/

- Overview write up and performance impact:
  
I add OBJ Mesh reader and render obj in my scene. To do this, I firstly learned the format of obj files and then  wirte a obj reader by myself. And due to the different size of the objs I load into my scene, I scale all of the objs to the size of (1,1,1).(Maybe smaller, as the obj's length,width,height aren't always the same) After that, I load each triangle mesh as a new object in my scene to do path trace and thus the more meshes the obj file has, the slower the render will be. Here is the scene with an obj loaded(I just use a tetra obj as it has only four meshes and can be rendered faster):

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_WithOBJ.png)

- Accelerate the feature:
  
I add bounding box to the obj object to accelerate the ray intersect part. 

- Compare to a CPU version: NULL

- Further optimized:

If the objs are complex, I still need long time to render each frame even I add BB for them. So, I think maybe I should use more accelerate methods like kd-tree to make the renderer faster.
 
 
####4.Motion blur
- Reference: http://www.cs.cmu.edu/afs/cs/academic/class/15462-s09/www/lec/13/lec13.pdf

- Overview write up and performance impact:
  
I add motion blur for objects. To do this, I add a new attribute for each object called MBV, and the object will move its position according to this velocity vector(this part is hard-coded in cudaRaytraceCore). And here the green sphere has the motion blur effect:

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_MotionBlur.png)

- Accelerate the feature: NULL

- Compare to a CPU version: NULL

- Further optimized:

The users can only set the velocity for each object, I think I can add more parameters and let users control how the object move in each frame.


####5.Bump mapping
- Reference: 

http://www.paulsprojects.net/tutorials/simplebump/simplebump.html

http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html

- Overview write up and performance impact:
  
I try to add bump map for objects, but can only realize normal map now. To do this, I add a new attribute for each object called BUMP, and when renderer reads the scene file, it also stores the normal map's color to a buffer. When it does ray intersect, it returns the intersect normal according to the corresponding color in the buffer. And here is the scene with normal map(See the details on the sphere and floor):

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_NormalMap.png)

- Accelerate the feature: NULL

- Compare to a CPU version: NULL

- Further optimized:

Bump mapping is the combine of normal mapping and height mapping(which changes points position according to the map color). And I need to add height map, or get height map from the normal map to finish bump mapping.



####6.Texture mapping
- Reference:  http://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html

- Overview write up and performance impact:
  
I add texture for cubes and spheres. To do this, I add a new attribute for each object called MAP, and when renderer reads the scene file, it also reads in the texture map's color to a buffer. When the program does path trace, it gets the color from this buffer according to the intersect point's position. And here is the scene with texture map(Much more colorful):

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_TextureMap.png)

- Accelerate the feature: NULL

- Compare to a CPU version: NULL

- Further optimized:

Right now the same texture map will be stored several times in the color buffer if it is assigned to different objects, as the map is an attribute of object. I think it will be better to give each texture map an index and read in the texrure map when a new texture index appears. In this way, much memory space can be saved.


####7.Depth of field
- Reference:  http://www.keithlantz.net/2013/03/path-tracer-depth-of-field/

- Overview write up and performance impact:
  
I add depth of field in my scene. To do this, I add two new attributes for camera, DOFL and DOFR. DOFL decides the camera's distance to focal plane(on z axis) and DOFR decides the blurradius(The higher, the more blurred the scene will be). According to these two parameters, in each iteration before the path tracer begins, the camera's position and the initial rays' direction will be jittered. In this way, I get a scene picture with depth of field:

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_DepthOfField.png)

- Accelerate the feature: NULL

- Compare to a CPU version: NULL

- Further optimized:

Maybe I can add more focal length to let the camera focus on more focal planes. Although it is not correct physically, we may get some fantastic results.


####8.Interactive camera
- Reference:  NULL

- Overview write up and performance impact:
  
This one is the easiest one and I just defines more keys in keyCallback function. When the key is pressed, renderer will clear the screen and redo the path trace based on the new camera's position. See more detail in the video link.

- Accelerate the feature: NULL

- Compare to a CPU version: NULL

- Further optimized:

Using mouse to control the camera is more convenient, but to do this, I must speed my renderer up at first.


### Scene Format

The scene format has changed due to the features I add.<br />

- Materials: Unchanged<br />

- Cameras: Add two new parameters<br />
DOFL(focal length)  //The camera's distance to focal plane(on z axis)  <br />
DOFR(blur radius)   //The blur extent<br />

Example:<br />
CAMERA<br />
RES         800 800<br />
FOVY        25<br />
ITERATIONS  5000<br />
FILE        test.bmp<br />
frame 0<br />
EYE         0 4.5 14<br />
VIEW        0 0 -1<br />
UP          0 1 0<br />
DOFL        14.0<br />
DOFR        0.7<br />


- Objects:  Add three new parameters<br />
MBV(Motion blur velocity)  //The velocity the object has <br />
MAP(Texture map)           //The object's texture map's path<br />
BUMP(Bump map)           //The object's bump map's path<br />


Example:<br />
OBJECT0<br />
cube<br />
material	 0<br />
frame 	0<br />
TRANS 	  0 0 0<br />
ROTAT  	   0 0 90<br />
SCALE  	   .01 10 10   <br />
MBV  	 0 0 0<br />
MAP  	texture/wood.jpg<br />
BUMP 	texture/bumpmap.jpg<br />



### Scene Control
|Key | Function
|------|----------
|Directional keys | `Move camera up/down/left/right`
|Z/C |  `Zoom in/out`
|D | `Enable/Disable depth of field`
|M| `Enable/Disable motion blur`
|N| `Enable/Disable bump map`
|T| ` Enable/Disable texture map`
|Space| `Enable/Disable stream compact`
|Esc| `Exit renderer`


### Video Link<br />
http://youtu.be/AwxrfiRXsPQ

