CIS 565 Project3 : CUDA Pathtracer
===================


## INTRODUCTION
Implemented a CUDA based pathtracer capable of
generating pathtraced rendered images, including diffuse, reflection, refraction. 

## BACIC FEATURES

* Raycasting from a camera into a scene through a pixel grid
* Diffuse surfaces
* Perfect specular reflective surfaces
* Cube intersection testing
* Sphere surface point sampling
* Stream compaction optimization

My cube intersection(completed in previous CPU ray tracer) was based on the logic and algorithm given by 
http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-box-intersection/ 


I have added some small things to make performance better, like Anti-Aliasing (using jitterred coordinates), 
and Importance Sampling (cos weighted).


## ADVANCED FEATURES


“Measuring and Modeling Anisotropic Reflection” by Gregory J. Ward
* Refraction, i.e. glass
Refraction was done based on snell's law and fresnel equation.
Both clear glass or colored glass can be handled here. Wondering doing some frosted glass...

##Texture mapping 
##Depth of field
in order to have to depth of field effects, specify camera aperture and focal length in scene txt file, 
and turn on "DEPTH_OF_FIELD" in "raytracerKernel.h" and "scene.h"


##OBJ Mesh loading and rendering
##Interactive camera
Interactive Camera is implemented to provide flexible in rendering angles, including pan, tilt, zoom, everything. 
Play with camera like a camera man! :) Rendering will start fresh every time camera is changed.
STEP_SIZE - step size of camera movements
*W - move up
*Q - move down
*S - move left
*D - move right
*Q - move forward (zoom in)
*E - move backward (zoom out)
*up - rotate up
*down - rotate down
*left - rotate left
*right - rotate right
, - rotate CCW
. - rotate CW
[![ScreenShot]( https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)] (http://www.youtube.com/embed/RtjJXwnUBZo)
<iframe width="420" height="315" src="//www.youtube.com/embed/RtjJXwnUBZo" frameborder="0" allowfullscreen></iframe>

* overview write up of the feature
* performance impact of the feature
* if you did something to accelerate the feature, why did you do what you did
* compare your GPU version to a CPU version of this feature (you do NOT need to 
  implement a CPU version)
* how can this feature be further optimized (again, not necessary to implement it, but
  should give a roadmap of how to further optimize and why you believe this is the next
  step)



## SCENE FORMAT
I have some scene files that are interesting to render:
* sampleScene
I modified the original file and in current version, there is diffuse item, highly reflective item, glass item.
WIHOUT depth of field.
* myScene
Everything same as "sampleScene" except this one is 
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



