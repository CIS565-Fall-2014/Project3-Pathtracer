#CUDA Pathtracer (developing)

This is a standalone pathtracer implemented using CUDA and OpenGL.


##Features:
###Features from Pathtracing
  -softshadow
  -area light
  -color bleeding
  -global illumination

###Special Features:
  -Ray parallel instead of pixel parallel more maximum performance

  -Stream compaction on rays for each depth level

  -BSDF using Russian Roulette 
  
  -Depth of Field, by jittering eye position and set image plane at focal length

  -Fresnel Coefficients for reflection/refraction

  -Caustics (free from above)

  -Obj loading, polygon mesh rendering

  -Anti-Alisasing jittered pixle position

Assuming light unpolarized

![](fresnel1.bmp)

use 1/2 (RS + RP) to get coefficient for reflection

##Result:
![](DOF.0.bmp)
![](SSS3.bmp)

##Performance

#Bugs to fix: 
cube intersection/normal problem when dimension <= 0.01

output image doesn't match OpenGL rendering

