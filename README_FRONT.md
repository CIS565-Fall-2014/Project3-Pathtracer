CIS 565 Project 3: CUDA Pathtracer
==================================

#### Michael Li

### _Overview_

[Text overview, some pictures.]






### _Details on what I implemented_

##### getRandomPointOnSphere()

Used:

* http://mathworld.wolfram.com/SpherePointPicking.html
* http://tutorial.math.lamar.edu/Classes/CalcIII/SphericalCoords.aspx


##### boxIntersectionTest()

Makes use of a helper function "planeIntersectionTest()". I've implemented this
stuff in CIS 560 before so the code is being reused here.

See:

* http://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm
* https://github.com/citizen-of-infinity/SCHOOL-CIS-560-Raytracer/blob/master/CIS560hw2/intersect.cpp
* Slide 793 in the FALL 2013 notes for CIS 560 - basically, using ^-1^T to correctly
  map the normal from normalized space to world space.


##### getRandomDirectionInSphere()

It is *incredibly* unclear what is supposed to be going on with this function
and with calculateRandomDirectionInHemisphere(). There is not a single bit of
useful documentation in the code provided, and it becomes clear that neither
of these two "random" functions is random in the slightest. Through some testing
in the main() function I was able to determine that
calculateRandomDirectionInHemisphere() expects its 2 float inputs to be between
0 and 1, and I will do the same for this function. In that case, it's almost the
same thing as getRandomPointOnSphere().
  

##### raytraceKernel.cu work

Since the 3 incomplete functions I need to finish up are dependent on each other,
I'll be working on them at the same time.  

I decided to proceed in a series of checkpoints to make debugging easier.

###### Checkpoint 1:

Fix how cudaimage/colors and the PBO work so that the colors[] array now ACCUMULATES the value
it gets on each iteration, and make sure the right stuff is sent to the PBO and
the .bmp output.

This causes the screen to display an ever-more featureless gray as the iteration
count increases, as expected. (Over time, a bunch of random colors averages out
to 50% gray.) The static is no longer completely "fresh" with each iteration.

![checkpoint 1-1](images/chkpt1-1.png)
![checkpoint 1-2](images/chkpt1-2.png)
![checkpoint 1-3](images/chkpt1-3.png)

The output .bmp file exactly matches the last screenshot above (I capped
ITERATIONS to 1000 to make things faster).

###### Checkpoint 2:

Deterministically initialize all rays from camera (use raycastFromCameraKernel())
and have them return the color of the first thing they intersect.

I already have some ideas as to what this should look like, based on posts from
other students on the forums.







### _Extra feature analysis_

["extra features", performance analysis]







### _Overall comments_

[Potential ideas for better optimization]
