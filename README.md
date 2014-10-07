![](https://github.com/wulinjiansheng/Project3-Pathtracer/blob/master/windows/Project3-Pathtracer/Project3-Pathtracer/Final%20Images/FinalScene_AllEffects.png)

CUDA Pathtracer
===============

Fall 2014
=========

PROJECT DESCRIPTION
-------------------

This is a GPU path tracing program, with following features:

####Basic:
- Raycasting from a camera into a scene through a pixel grid
- Diffuse surfaces
- Perfect specular reflective surfaces
- Cube intersection testing
- Sphere surface point sampling
- Stream compaction optimization

####Extra:
- Texture mapping 
- Bump mapping
- Depth of field
- Refraction, i.e. glass
- OBJ Mesh loading and rendering
- Interactive camera
- Motion blur
- Anti-Alisasing

####1.Texture mapping
I add texture map for cube and sphere and the 
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
