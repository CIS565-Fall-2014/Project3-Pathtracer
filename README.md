CIS 565 Project3 : CUDA Pathtracer
===================

Fall 2014

Due Wed, 10/8 (submit without penalty until Sun, 10/12)

## INTRODUCTION
In this project, you will implement a CUDA based pathtracer capable of
generating pathtraced rendered images extremely quickly. Building a pathtracer can be viewed as a generalization of building a raytracer, so for those of you who have taken 460/560, the basic concept should not be very new to you. For those of you that have not taken
CIS460/560, raytracing is a technique for generating images by tracing rays of
light through pixels in an image plane out into a scene and following the way
the rays of light bounce and interact with objects in the scene. More
information can be found here:
http://en.wikipedia.org/wiki/Ray_tracing_(graphics). Pathtracing is a generalization of this technique by considering more than just the contribution of direct lighting to a surface.

Since in this class we are concerned with working in generating actual images
and less so with mundane tasks like file I/O, this project includes basecode
for loading a scene description file format, described below, and various other
things that generally make up the render "harness" that takes care of
everything up to the rendering itself. The core renderer is left for you to
implement.  Finally, note that while this basecode is meant to serve as a
strong starting point for a CUDA pathtracer, you are not required to use this
basecode if you wish, and you may also change any part of the basecode
specification as you please, so long as the final rendered result is correct.

## CONTENTS
The Project3 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio
  solution and the OSX and Linux makefiles reference this folder for all 
  source; the base source code compiles on Linux, OSX and Windows without 
  modification.  If you are building on OSX, be sure to uncomment lines 4 & 5 of
  the CMakeLists.txt in order to make sure CMake builds against clang.
* data/scenes/ contains an example scene description file.
* renders/ contains an example render of the given example scene file. 
* windows/ contains a Windows Visual Studio 2010 project and all dependencies
  needed for building and running on Windows 7. If you would like to create a
  Visual Studio 2012 or 2013 projects, there are static libraries that you can
  use for GLFW that are in external/bin/GLFW (Visual Studio 2012 uses msvc110, 
  and Visual Studio 2013 uses msvc120)
* external/ contains all the header, static libraries and built binaries for
  3rd party libraries (i.e. glm, GLEW, GLFW) that we use for windowing and OpenGL
  extensions

## RUNNING THE CODE
The main function requires a scene description file (that is provided in data/scenes). 
The main function reads in the scene file by an argument as such :
'scene=[sceneFileName]'

If you are using Visual Studio, you can set this in the Debugging > Command Arguments section
in the Project properties.

## REQUIREMENTS
In this project, you are given code for:

* Loading, reading, and storing the scene scene description format
* Example functions that can run on both the CPU and GPU for generating random
  numbers, spherical intersection testing, and surface point sampling on cubes
* A class for handling image operations and saving images
* Working code for CUDA-GL interop

You will need to implement the following features:

* Raycasting from a camera into a scene through a pixel grid
* Diffuse surfaces
* Perfect specular reflective surfaces
* Cube intersection testing
* Sphere surface point sampling
* Stream compaction optimization

You are also required to implement at least 2 of the following features:

* Texture mapping 
* Bump mapping
* Depth of field
* Refraction, i.e. glass
* OBJ Mesh loading and rendering
* Interactive camera
* Motion blur
* Subsurface scattering

The 'extra features' list is not comprehensive.  If you have a particular feature
you would like to implement (e.g. acceleration structures, etc.) please contact us 
first!

For each 'extra feature' you must provide the following analysis :
* overview write up of the feature
* performance impact of the feature
* if you did something to accelerate the feature, why did you do what you did
* compare your GPU version to a CPU version of this feature (you do NOT need to 
  implement a CPU version)
* how can this feature be further optimized (again, not necessary to implement it, but
  should give a roadmap of how to further optimize and why you believe this is the next
  step)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
## BASE CODE TOUR
You will be working in three files: raytraceKernel.cu, intersections.h, and
interactions.h. Within these files, areas that you need to complete are marked
with a TODO comment. Areas that are useful to and serve as hints for optional
features are marked with TODO (Optional). Functions that are useful for
reference are marked with the comment LOOK.

* *[IN PROGRESS]* raytraceKernel.cu contains the core raytracing CUDA kernel. You will need to
  complete:
    * cudaRaytraceCore() handles kernel launches and memory management; this
      function already contains example code for launching kernels,
      transferring geometry and cameras from the host to the device, and transferring
      image buffers from the host to the device and back. You will have to complete
      this function to support passing materials and lights to CUDA.
    * **[DONE]** raycastFromCameraKernel() is a function that you need to implement. This
      function once correctly implemented should handle camera raycasting. 
    * raytraceRay() is the core raytracing CUDA kernel; all of your pathtracing
      logic should be implemented in this CUDA kernel. raytraceRay() should
      take in a camera, image buffer, geometry, materials, and lights, and should
      trace a ray through the scene and write the resultant color to a pixel in the
      image buffer.
	* Remember that **stream compaction optimization** needs to be done somewhere
	  in here!

* intersections.h contains functions for geometry intersection testing and
  point generation. You will need to complete:
    * **[DONE]** boxIntersectionTest(), which takes in a box and a ray and performs an
      intersection test. This function should work in the same way as
      sphereIntersectionTest().
    * **[DONE]** getRandomPointOnSphere(), which takes in a sphere and returns a random
      point on the surface of the sphere with an even probability distribution.
      This function should work in the same way as getRandomPointOnCube(). You can
      (although do not necessarily have to) use this to generate points on a sphere
      to use a point lights, or can use this for area lighting.

* interactions.h contains functions for ray-object interactions that define how
  rays behave upon hitting materials and objects. You will need to complete:
    * **[DONE]** getRandomDirectionInSphere(), which generates a random direction in a
      sphere with a uniform probability. This function works in a fashion
      similar to that of calculateRandomDirectionInHemisphere(), which generates a
      random cosine-weighted direction in a hemisphere.
    * **[DONE (to the best of my ability)]** calculateBSDF(), which takes in an incoming ray, normal, material, and
      other information, and returns an outgoing ray. You can either implement
      this function for ray-surface interactions, or you can replace it with your own
      function(s).

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	  
You will also want to familiarize yourself with:

* sceneStructs.h, which contains definitions for how geometry, materials,
  lights, cameras, and animation frames are stored in the renderer. 
* utilities.h, which serves as a kitchen-sink of useful functions

## NOTES ON GLM
This project uses GLM, the GL Math library, for linear algebra. You need to
know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed
  via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is
  used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but
  pre-Fermi cards do not play nice with GLM matrices. As such, in this project,
  GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found
  in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is
  provided as multiplyMV() in intersections.h.

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

## THIRD PARTY CODE POLICY
* Use of any third-party code must be approved by asking on our Google Group.  
  If it is approved, all students are welcome to use it.  Generally, we approve 
  use of third-party code that is not a core part of the project.  For example, 
  for the ray tracer, we would approve using a third-party library for loading 
  models, but would not approve copying and pasting a CUDA function for doing 
  refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another
  student's code, is an academic integrity violation, and will result in you
  receiving an F for the semester.

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

Things to keep in mind for "performance analysis":<br>
(just copy this list over for future projects)

1.  See how changing block/tile size affects performance (Project-1)
2.  Change input parameters to see how rendering time, etc is affected. Plot
    some graphs to provide visual aids (Project-2)
	  * I feel like only stream compaction analysis will be something that can
	    be illustrated with a graph.
3.  Throw out some ideas as to how things could be optimized further (Project-2)
4.  Provide extra feature analysis as specified (**Project-3**)
      * These descriptions should be thorough, but I don't think they warrant
	    graphs.

Be sure to open a pull request and to send Harmony your grade and why you
believe this is the grade you should get.
