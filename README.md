CIS 565 Project3 : CUDA Pathtracer
===================

Fall 2014

Due Wed, 10/8 (submit without penalty until Sun, 10/12)


## RUNNING THE CODE
* Make sure to look at sampleScene.txt and pay attention to DOF and APERATURE.

## REQUIREMENTS
I have so far implemented the following additional features:

* Depth of field
* Refraction, i.e. glass

## ANALYSIS DEPTH OF FIELD
![DOF5000.bmp](https://raw.githubusercontent.com/RTCassidy1/Project3-Pathtracer/master/DOF5000.bmp)
I incorporated Depth of Field by adding two parameters to the camera. 
* the DOF parameter is the distance from the eye to the focal point.  (it would be more aptly named focal length, note to self for future refactoring)
* the APERATURE parameter is the size of the aperature
The way the feature works is that when casting the first ray, after selecting within my jittered pixel I place the FocalPoint along this ray at DOF distance.  Then I randomly jitter my pixel position within the range of the aperature (so a larger aperature = smaller depth of field = more blur.
* Adding this feature did add a few more calculations into the initial raycast which slowed performance down slightly.  I was overly cautious in normalizing my direction vectors so it's possible I could remove a few of these normalizations for additional speedup. I also probably don't need to jitter within the pixel anymore, but kept it there so I still maintain AntiAliasing when aperature is 0.
* I didn't do anything particularly to speed this feature up. In RayTracing people often have to supersample the pixel to achieve this, but since we're already taking 1000s of samples per pixle I merely had to jitter around within the size of the aperature.

## ANALYSIS REFRACTION
![FirstRefraction.bmp](https://raw.githubusercontent.com/RTCassidy1/Project3-Pathtracer/master/FirstRefraction.bmp)
My BSDF is pretty simple, I would like to go through and make it more modular at a future time when I have a chance to refactor.  Instead of using Reflective and Refractive markers as flags, I used them as floats each representing the percentage of photons that hit is that will reflect or refract(transmit).  If a photon doesn't reflect or refract then it is treated as diffuse. 
* my BSDF accepts two random numbers from 0-1 as parameters.  I use these to determine if I will check the reflectance or refractance threshold first.  I then use the number to determine which way to treat the material. If my random number is less than the threshhold for reflectance (or refractance) I treat it as a reflection (refraction).  If it is above the threshold it falls through to the next test. If it fails everything else it's treated as diffuse.

## ANALYSIS STREAM COMPACTION
I also implemented stream compaction in an effort to speed up my renders.  Unfortunately it has so far offered little to no performance increase with depth 5 or 10.  I think this is because my implementation is not efficient enough and has too much memory access overhead.  I used thrust to scan an array for retired threads, but then implemented my own function to compact the rayStates.  I think with a little more research of Thrust I can implement my rayState array as a thrust vector and use built in functions to prune it on each depth iteration. 
* I also planned to look at shared memory, but haven't yet had the chance.  There are a lot of parallel streams doing ray-intersection tests with the same geometry, so I speculate there could be an increase in efficiency by moving the geometery (and possibly materials) into shared memory.  This will have an overhead to actually move them, and the shear number of threads may actually be hiding any latency in the accesses, but I haven't had a chance to look at it yet.

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
* REFL (bool refl)									//how reflective is it? 0 means diffuse
* REFR (float refr)									//how transparent is it? 0 means opaque
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
* DOF (float x)											//the Distance to the focal point
* APERATURE (float x)						//the size of the aperature (should be large)

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
