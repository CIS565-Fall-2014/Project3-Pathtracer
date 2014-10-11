Vulcan
============

Vulcan is a GPU accelerated, physically-based path tracer written in CUDA and C++.  You can look [here] for a more complete breakdown and development blog. 

Features
--------

Here's a quick down break down of the features:
  - Supports triangle soup meshes, as well as cube and sphere primitives.
  - Supports a variety of simple BRDFs - Lambertian diffuse, Blinn-Phong specularity, Fresnel reflection / refraction, Oren-Nayar rough diffuse, Cook-Torrance rough specular
  - Supports a few super sampling techniques - motion blur, depth of field, and anti-aliasing.

![Base PT Image][base pt image]

This image shows 3 spheres floating in a Cornell box.  All 3 have simple diffuse BRDFs.  The red sphere is exhibiting slight motion blur.


![Base PT Image][ct fres]

This scene has the same geometry, but the spheres now have more interesting BRDFs. The red sphere now exhibits Cook-Torrance specularity, which gives a rougher appearance than the more common Blinn-Phong model.  The other 2 spheres are exhibiting Fresnel reflection/refraction.

![Base PT Image][fres]

This image is a visual explanation of Fresnel reflection/refraction.  The left sphere is pure refraction, while the right sphere is pure reflection.  The center sphere combines the two.  The combination is somewhat mathematical, but it involves analysing the incoming and outgoing ray direction.

![Base PT Image][obj]

Here's an image to show triangular meshes being rendered in the system.  The white spheres are lights.

Overall, the functionality of the renderer is quite limited, but I think it's a good start for further study, and I learned a lot about GPU programming through working on this project.

##Performance

Now, I'm going to talk a bit about the performance of the renderer, as well as the impact of some of the features on that performance.

###Stream Compaction

I used a technique called stream compaction to speed up the renderer.  Without going into unnecessary detail, stream compaction involves thinking about data in a slightly different way.  If you're not familiar with how a GPU works, you'll need to know that a GPU has a large number of lightweight threads.  That means that you can do a lot of (simple) operations at the same time.  That's why things that run on the GPU have the reputation of being very fast.

An easy way to think about rendering on the GPU is to think of each pixel as a single thread.  That works pretty well, because at the end of the day, each pixel can be rendered individually, without any information from other pixels.  That's good, because inter-thread communication can be slow.  There is one problem with this methodology, however.  Imagine 2 threads running, and 1 doesn't hit anything, while the other gets trapped between 2 mirrors.  The first thread has finished its work, while the second will have to do a lot of work before finishing.  This is not an efficient use of your resources!  A smart thing to do would be to recognize that the first thread is done, and put it to work helping the second.  But how do you do that?  One way is to use a stream compaction!

A better way is to think of each thread as a single bounce of a ray through the scene.  That way, when you run into the case described above, you can use your resources in a more complete way.

#### Stream compaction performace impact : 5.5x speedup!

That's great, but I could still push the performance of this technique further.  There are better stream compaction algorithm out there, and I also imagine that there are implementations that are much faster than mine.

---------

###Super Sampling (DOF, Anti-Aliasing, Motion Blur)

Depth of Field, Anti-Aliasing, and Motion Blur can be accomplished in a similar fashion.  Instead of taking a single sample per frame, you take multiple samples!  For DOF, you sample the camera's position.  This attempts to simulate how a real camera would achieve depth of field (although it is a rather crude attempt).  For anti-aliasing, you sample the pixel position.  And for motion blur, you sample an object position in space.  You may be asking, won't all those extra samples be really expensive?  Luckily for us, we're already taking multiple samples every frame!  To eliminate the noise inherent to naive path tracers, we have to run the algorithm many times per frame!

#### Super Sampling performace impact : Free!

That's great!  You get these things for free!  These would also be free in a CPU implementation.

###Triangle Mesh Primitives

The nice thing about triangle meshes is that you can render more interesting objects than cubes and spheres!  However, that interest comes with a price.  Each triangle intersection test takes about as long as a test for a sphere or cube.  And your meshes will usually have many triangles!  That means that your renderer has a lot more work to do.  One simple speed up to try to combat this problem is to use boudning boxes, which prevent your renderer from testing against all of those triangle when it doesn't have to.

#### Triangle Mesh Primitives performace impact : Depends, scene complexity is generally much higher.

One glaring shortcoming of this renderer is the lack of a sophisticated acceleration structure, like a KD-Tree of BVH.  On the CPU, you run into the exact same problem of scene complexity.

###Cook-Torrance, Oren-Nayar, Fresnel

The first 2 BRDFs are fancier progressions of the more common Lambertian diffuse and Blinn-Phong specular.  Fresnel is a combination of reflection and refractino.  All 3 are a bit slower, as they are more complicated to compute.  However, given the other major bottlenecks present, the minor slowdowns presented by these are not a major concern.


#### Cook-Torrance, Oren-Nayar, Fresnel performace impact : Slight slowdown.

These BRDFs would be slightly faster on the CPU, as they all have branches that would benefit from the CPUs branch predictors.

[base pt image]:http://3.bp.blogspot.com/-WENKFpXCcew/Uj4BJwYMTFI/AAAAAAAABeM/6RbhNzNDNfI/s320/pt.bmp
[ct fres]:http://4.bp.blogspot.com/-spe4bKN8_Og/Uj38S9UgVeI/AAAAAAAABdw/P3HqOcest9s/s400/pathTrace.bmp
[fres]:http://1.bp.blogspot.com/-4llLdG19UOQ/Uj4BJzeYdfI/AAAAAAAABeI/DSnifT4AbhQ/s400/comp.bmp
[obj]:http://3.bp.blogspot.com/-sN4J1YlqzBk/UlnmSjjTEbI/AAAAAAAABjw/DrQV4XTGyNs/s640/a_raw.1.bmp
[here]:http://blog.jeremynewlin.info/search/label/vulcan