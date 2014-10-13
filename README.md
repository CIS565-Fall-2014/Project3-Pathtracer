Radium Yang's Pathtracer

******************** 6000 Iterations, Ray Depth: 8, Depth of Field: focal length 13 *******************
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/depth%2Brefract%2Bbackreflect%2B6000.bmp)

##FEATURE LIST
- Basic
	* Raycasting
	* Diffuse surfaces
	* Perfect specular reflective surfaces
	* Cube intersection testing
	* Sphere surface point sampling
	* Stream compaction optimization
- Extra
	* Depth of field
	* Refraction, i.e. glass

###Feature Overview
For the Pathtracing Algorithm and basic features' implementations, I mainly followed the raytracing slides in CIS 560 and pathtracing slices in CIS 565.

Extra Features:
- Depth of field
  I referred to the algorithm posted here: http://ray-tracer-concept.blogspot.com/2011/12/depth-of-field.html
  Find a focal plane before the camera, the new ray direction should come from the image plane to the focal point.
	
- Reflection/ Refraction
  I referred to the algorithm posted here: http://ray-tracer-concept.blogspot.com/2011/12/refraction.html
  Interesting bug... when calculating the accumulated reflective/refractive factor, failed to limit the factor to be required number.
  Solution:in reflection, if randomnumber < hasReflective, do reflection; else do diffuse.
           in refraction, if randomnumber < refractive, do refraction; else do reflection.
  Before:
  ![alt tag]()
  After: 
  ![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/depth%2Brefract%2Bbackreflect%2B2000.bmp)
- Stream compaction
  reference: http://wiki.thrust.googlecode.com/hg-history/312700376feeadec0b1a679a259d66ff8512d5b3/html/group__stream__compaction.html#ga517b17ceafe31a9fc70ac5127bd626de
  To accelerate the performance, using thrust to do stream compaction to delete rays which have already been hit to the background or light from the raypool.
  Thus, after each iteration, the valid rays in the ray pool will be decreased, which will help improve the performance.

## Performance Analysis

## Progress Screenshots

Step 1: ray intersection + diffuse color + soft shadow (sample 20) test
///// trace depth: 8 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/tmp_2.bmp)

Step 2: use ray pool algorithm, diffuse + reflection + specular test
///// trace depth: 8 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/spec_1.bmp)

Step 3: add refraction & depth of field
///// trace depth: 8 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/depth_refract_1000.bmp)

Step 4: set customized scene (reflected walls) + performance tests

////// Iterations: 4000, trace depth: 2 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/4000_depth2.bmp)

////// Iterations: 4000, trace depth: 5 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/4000_depth5.bmp)

////// Iterations: 500, trace depth: 8 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/depth%2Brefract%2Bbackreflect%2B500.bmp)

////// Iterations: 6000, trace depth: 8 /////
![alt tag](https://github.com/radiumyang/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/depth%2Brefract%2Bbackreflect%2B6000.bmp)


===================

## RUNNING THE CODE
The main function requires a scene description file (that is provided in data/scenes). 
The main function reads in the scene file by an argument as such :
'scene=[sceneFileName]'

If you are using Visual Studio, you can set this in the Debugging > Command Arguments section
in the Project properties.


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


An example of a mesh object is as follows:

OBJECT 0
mesh tetra.obj
material 0
frame 0
TRANS       0 5 -5
ROTAT       0 90 0
SCALE       .01 10 10 

