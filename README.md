#CUDA Pathtracer (developing)

This is a standalone pathtracer implemented using CUDA and OpenGL.


##Features Implemented:
###Free Features from Pathtracing
  -softshadow
  -area light
  -color bleeding
  -global illumination

###Special Features:
  -Fresnel Coefficients for reflection/refraction
  -Caustics (free from above)

Assuming light unpolarized

![](fresnel1.bmp)

use 1/2 (RS + RP) to get coefficient for reflection

##Result:
![](scene2.bmp)

##Performance

#Bugs to fix: 
cube intersection/normal problem when dimension <= 0.01

output image doesn't match OpenGL rendering

