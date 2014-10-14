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
/***************************************************************************************************************/
## Implementation
All required features are implemented;
2 extra features, Refraction and Subsurface scattering are also provided. 

![Alt text](https://github.com/chiwsy/Project3-Pathtracer/blob/master/test.png)

![Alt text](https://github.com/chiwsy/Project3-Pathtracer/blob/master/test_mark.png)

![Alt text](https://github.com/chiwsy/Project3-Pathtracer/blob/master/compare.png)

These 3 images show the features implemented and the performance of the render with and without stream compaction.

