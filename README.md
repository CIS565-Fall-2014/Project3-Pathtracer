

CUDA Pathtracer
===============

Fall 2014
=========

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_AllEffects.png)

PROJECT DESCRIPTION
-------------------

This is a GPU path tracing program, with following features:

####Basic:
- Raycasting from a camera into a scene through a ray grid
- Diffuse surfaces
- Perfect specular reflective surfaces
- Cube intersection testing
- Sphere surface point sampling
- Stream compaction optimization

####Extra:
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
  
I add super sample anti-alisasing, which makes my render result much more beautiful. To do this, I just jitter the initial rays randomly in each iteration. And here is the comparison between the scene with SSAA and the scene without SSAA:

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/DetailWithSSAA.bmp)

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/DetailWithNoSSAA.bmp)

- Accelerate the feature: NULL

- Compare to a CPU version: 
  
In CPU version we may use grid algorithm to jitter the initial rays, but in GPU version we jitter the rays with Jitter algorithm. And their results are almost the same.

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

If the objs are complex, I still need long time to render each frame even I add BB for them. So, I think maybe I should use more accelerate methods like kd-tree to make the render faster.
 
 
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
  
I try to add bump map for objects, but can only realize normal map now. To do this, I add a new attribute for each object called BUMP, and when we read the scene file we also read in the normal map's color to a buffer. When we do ray intersect, we return the intersect normal according to the color we get from the normal map. And here is the scene with normal map(See the detail on the sphere and floor):

![Alt text](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_NormalMap.png)

- Accelerate the feature: NULL

- Compare to a CPU version: NULL

- Further optimized:

Bump mapping is the combine of normal mapping and height mapping(which changes points position according to the map color). And I need to add height map, or get height map from the normal map to finish bump mapping.






For each 'extra feature' you must provide the following analysis :
* overview write up of the feature
* performance impact of the feature
* if you did something to accelerate the feature, why did you do what you did
* compare your GPU version to a CPU version of this feature (you do NOT need to 
  implement a CPU version)
* how can this feature be further optimized (again, not necessary to implement it, but
  should give a roadmap of how to further optimize and why you believe this is the next
  step)

## SCENE FORMAT
This project uses a custom scene description format.
Scene files are flat text files that describe all geometry, materials,
lights, cameras, render settings, and animation frames inside of the scene.
Items in the format are delimited by new lines, and comments can be added at
the end of each line preceded with a double-slash.

Materials are defined in the following fashion:

* MATERIAL (material ID)								//material header
* RGB (float r) (float g) (float b)					//diffuse color
* SPECX (float specx)									//specular exponent
* SPECRGB (float r) (float g) (float b)				//specular color
* REFL (bool refl)									//reflectivity flag, 0 for
  no, 1 for yes
* REFR (bool refr)									//refractivity flag, 0 for
  no, 1 for yes
* REFRIOR (float ior)									//index of refraction
  for Fresnel effects
* SCATTER (float scatter)								//scatter flag, 0 for
  no, 1 for yes
* ABSCOEFF (float r) (float b) (float g)				//absorption
  coefficient for scattering
* RSCTCOEFF (float rsctcoeff)							//reduced scattering
  coefficient
* EMITTANCE (float emittance)							//the emittance of the
  material. Anything >0 makes the material a light source.

Cameras are defined in the following fashion:

* CAMERA 												//camera header
* RES (float x) (float y)								//resolution
* FOVY (float fovy)										//vertical field of
  view half-angle. the horizonal angle is calculated from this and the
  reslution
* ITERATIONS (float interations)							//how many
  iterations to refine the image, only relevant for supersampled antialiasing,
  depth of field, area lights, and other distributed raytracing applications
* FILE (string filename)									//file to output
  render to upon completion
* frame (frame number)									//start of a frame
* EYE (float x) (float y) (float z)						//camera's position in
  worldspace
* VIEW (float x) (float y) (float z)						//camera's view
  direction
* UP (float x) (float y) (float z)						//camera's up vector

Objects are defined in the following fashion:
* OBJECT (object ID)										//object header
* (cube OR sphere OR mesh)								//type of object, can
  be either "cube", "sphere", or "mesh". Note that cubes and spheres are unit
  sized and centered at the origin.
* material (material ID)									//material to
  assign this object
* frame (frame number)									//start of a frame
* TRANS (float transx) (float transy) (float transz)		//translation
* ROTAT (float rotationx) (float rotationy) (float rotationz)		//rotation
* SCALE (float scalex) (float scaley) (float scalez)		//scale

An example scene file setting up two frames inside of a Cornell Box can be
found in the scenes/ directory.

For meshes, note that the base code will only read in .obj files. For more 
information on the .obj specification see http://en.wikipedia.org/wiki/Wavefront_.obj_file.

An example of a mesh object is as follows:

OBJECT 0
mesh tetra.obj
material 0
frame 0
TRANS       0 5 -5
ROTAT       0 90 0
SCALE       .01 10 10 

Check the Google group for some sample .obj files of varying complexity.


## SELF-GRADING
* On the submission date, email your grade, on a scale of 0 to 100, to Harmony,
  harmoli+cis565@seas.upenn.com, with a one paragraph explanation.  Be concise and
  realistic.  Recall that we reserve 30 points as a sanity check to adjust your
  grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We
  hope to only use this in extreme cases when your grade does not realistically
  reflect your work - it is either too high or too low.  In most cases, we plan
  to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as
  the path tracer.  We will determine the weighting at the end of the semester
  based on the size of each project.

## SUBMISSION
Please change the README to reflect the answers to the questions we have posed
above.  Remember:
* this is a renderer, so include images that you've made!
* be sure to back your claims for optimization with numbers and comparisons
* if you reference any other material, please provide a link to it
* you wil not e graded on how fast your path tracer runs, but getting close to
  real-time is always nice
* if you have a fast GPU renderer, it is good to show case this with a video to
  show interactivity.  If you do so, please include a link.

Be sure to open a pull request and to send Harmony your grade and why you
believe this is the grade you should get.
