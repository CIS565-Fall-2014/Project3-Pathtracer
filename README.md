##Result:
![ResultImage](AntiAliasing_depth_10_iteration_5000_Cut.bmp)

#Compare the different iteration number:
* Iteration Number: 100
![ResultImage](AntiAliasing_depth_10_iteration_100_Cut.bmp)

* Iteration Number: 200
![ResultImage](AntiAliasing_depth_10_iteration_200_Cut.bmp)

* Iteration Number: 500
![ResultImage](AntiAliasing_depth_10_iteration_500_Cut.bmp)

* Iteration number: 1000
![ResultImage](AntiAliasing_depth_10_iteration_1000_Cut.bmp)

* Iteration number: 5000
![ResultImage](AntiAliasing_depth_10_iteration_5000_Cut.bmp)
Conclusion: The effect of increasing the iteration number is obvious. However, the difference of results of iteration number higher than 5000 are hard to distinguish by eyes.

##Compare the different depth:
* Depth: 3
![ResultImage](AntiAliasing_depth_3_iteration_5000_Cut.bmp)

* Depth: 5
![ResultImage](AntiAliasing_depth_5_iteration_5000_Cut.bmp)

* Depth: 10
![ResultImage](AntiAliasing_depth_10_iteration_5000_Cut.bmp)
Conclusion: The difference between depth 3 and depth 5 is distinguishable, but the difference between depth 5 and depth 10 is almost nondistinctive. Since I provide a indirect influence coefficient 0.5 to do my path tracer. And this coefficient represents the influence rate from the indirect radiance to the direct radiance.
Besides, our color RGB value is from 0~255, that means even the 5th depth has dramatical color change, it would only influence 4 of 256 color change in our result. 


##Anti-aliasing effect:
For the normal sampling, we cast ray from camera location through each pixel center.
For the anti-aliasing sampling, we still cast ray from camera locatin but not exactly through pixel center every time. Our ray will past through a random point with a given distance from the pixel center.
If we are doing the simple ray tracer, the anti-aliasing will increase the computation loading dramatically because we need to cast multiple rays for each pixel. However, while we are doing the path tracer, we have to sample a pixel multiple times. We could almost get anti-aliasing effect for free.
![ResultImage](antialiasing description.bmp)
![ResultImage](antialiasing comparision.bmp)


##Change the object material
* Change different refraction rate 2.6
![ResultImage](AntiAliasing_depth_5_iteration_1000_Refraction_2.6_Cut.bmp)

* Change specular exponential:
![ResultImage](Material Comparision.bmp)

##Depth of field effect:
* Description: The way that I used to create the depth of field effect is to set up a focal length and apperture radius. The first step is to cast the initial ray from camera position and past throught the pixel which we want to sample.
Then, use the focal length to compute the point start from camera center along the initial ray. Following, generate random point on the circular plane which is centered on camera position. Lastly, cast ray from the random point and past through the focal point and use this ray to do the sampling.
![ResultImage](DOF Description.bmp)

* Focal length:10  Aperture radius: 0.2
![ResultImage](focallength_10_aperture_0.2_depth_10_iteration_1000_Cut.bmp)

* Focal length:10  Aperture radius: 0.5
![ResultImage](focallength_10_aperture_0.5_depth_10_iteration_1000_Cut.bmp)

* Focal length:14  Aperture radius: 0.2
![ResultImage](focallength_14_aperture_0.2_depth_10_iteration_1000_Cut.bmp)

##Motion blur effect:
Motion blur effect is easy to implement. We could move the object every certain frames and do our path tracer. Then we could get the motion blur effect for free.
![ResultImage](MotionBlur2_depth_5_iteration_1000_Cut.bmp)

##Change view direction and camera interaction:
![ResultImage](change view direction_depth_5_iteration_1000_Cut.bmp)

##Texture mapping effect:
![ResultImage](testure mapping_Cut.bmp)
![ResultImage](testure mapping2_Cut.bmp)



Debug process
![ResultImage](normal vector_Cut.bmp)