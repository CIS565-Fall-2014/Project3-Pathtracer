CIS 565 Project 3: CUDA Pathtracer
==================================

* Kai Ninomiya (Arch Linux/Windows 8, Intel i5-4670, GTX 750)


Keybindings
-----------

* Escape: save image and exit program
* Space: save image early and continue rendering
* Arrow keys: rotate camera
* WASDRF keys: fly through space laterally (WASD) and vertically (RF) relative
  to the camera orientation


Base Code Features
------------------

* Configuration file reading
* Hemisphere sampling function (for diffuse)
* Objects: sphere


Features Implemented
--------------------

* Pathtracing algorithms
* Materials: diffuse, reflective, **refractive with Fresnel reflection**
* Camera: **movement** (controls above), **antialiasing**, **depth of field**
* Objects: cube, sphere **with correct normals when scaled**
* Performance: ray-level stream compaction

(Extras in **bold**.)


Renderings
----------

Combined test render:
![](images/22_brighter_d16s2000.png)

Annotated:
![](images/24_annotated.png)

With depth of field:
![](images/23_ultimate_d16s2000.png)


Performance
-----------

### Stream Compaction

In order to perform ray-level stream compaction, it was necessary to refactor
the rendering kernel into a single-ray step along the path. The result of this
is significantly more overhead, due primarily to performing stream compaction
between every step. At low path depths (e.g. 4), performance is lower with
stream compaction. However, stream compaction allows for extremely high path
depths (tested up to 1000) without very significant performance degradation.
This is because the vast majority of paths have terminated, and dead paths no
longer use kernel threads.

TODO: numbers

### Block sizes (with compaction)

TODO: graph

### Cube Intersection

Initially I did cube intersection naively, but this turned out to use way too
many GPU registers and had very bad performance. Rewriting based on Kay and
Kayjia's slab method reduced register usage by around 50 registers.


Extras
------

### Antialiasing

Samples are taken randomly from within each pixel.

**Performance:** Negligible impact.

*Mean rendering time per sample for an arbitrary example scene*

|   Before |    After |
| --------:| --------:|
| 33.19 ms | 33.19 ms |

Without:
![](images/15_slightly_better_depth1.png)

With:
![](images/16_antialiasing_depth1.png)


### Depth of Field

Origin and direction of camera rays is varied randomly (in a uniform circular
distribution) to emulate a physical aperture.

**Performance:** Negligible impact per-sample. However, it increases the number
of samples needed for visual smoothness due to the extreme variation between
samples. Implementation-wise, this is identical to analogous CPU code.

*Mean rendering time per sample for an arbitrary example scene*

|   Before |    After |
| --------:| --------:|
| 98.6  ms | 97.5  ms |

### Fresnel Reflection/Refraction

I used Schlick's approximation to compute the fractions of light
reflected/refracted, then used that as a probability for the next path ray.

Reflection is implemented using glm::reflect.  Refraction uses glm::refract and
handles total internal reflection. Intersection code was modified to report
whether the intersection was inside or outside the object, which allows correct
handling of indices of refraction at interfaces. (This technically could have
been done by adopting a different normal direction convention and checking
dot products with that, but this is more readable.)

(See debug render in the Debug Renders section below.)

**Performance:** Some performance impact per-sample. This is probably due to
the additional Fresnel factor computation and the additional random branch
calculation based on that factor.

*Mean rendering time per sample for an arbitrary example scene*

|   Before |    After |
| --------:| --------:|
| 233.2 ms | 249.9 ms |

### Camera movement

Keys for this are listed in the Keybindings section. This is implemented by
simply modifying the location of the camera, clearing the render, and starting
again.

### Scaled Sphere Normals

This is a minor thing, but I fixed the provided code to use inverse transpose
transformations to calculate the sphere normals.

(Error image in bloopers.)


Parameter Comparison Renderings
-------------------------------

Higher iteration counts always improved image smoothness, since more samples
were averaged over time. Higher path depths seem to correspond with bright
spots which never get optimized out, for some reason.

Depth 16, 500 samples:
![](images/22_ultimate_d16s500)

Depth 16, 2000 samples:
![](images/22_ultimate_d16s2000)

Depth 256, 500 samples:
![](images/22_ultimate_d256s500)

Depth 256, 2000 samples;
![](images/22_ultimate_d256s2000)


Earlier Renders
---------------

Diffuse-only:
![](images/08_diffuse_5000.png)

Diffuse + Reflective:
![](images/12_refactored.png)

With Direct Lighting, depth=1 (not included in final version):
![](images/15_slightly_better_depth1.png)

With Direct Lighting, depth=8
![](images/15_slightly_better_depth8.png)


Debug Renders
-------------

Normals:
![](images/01_debug_nor.png)

Positions:
![](images/02_debug_pos.png)

Materials/emittance:
![](images/04_debug_emit.png)

Direct lighting lit areas (not included in final version):
![](images/14_direct_lighting_depth1.png)

Fresnel reflected light factor (shown here for all reflective surfaces, but to
be only applied to refractive surfaces):
![](images/20_fresnel_debug_d16s500.png)


Bloopers
--------

Seed error:
![](images/06_seed_error_500.png)

Code refactoring error:
![](images/10_refactor_error.png)

Sphere normal error (from provided code):
![](images/22_bad_sphere_scaling_d16s500.png)
