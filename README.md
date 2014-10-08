#CUDA Pathtracer

This is a standalone pathtracer implemented on the GPU using CUDA and OpenGL.
##Feature Highlights:
 -Diffuse,Reflection, Refraction

 -Global Illumination,Soft Shadow, Caustics, Color Bleeding

 -Fresnel Coefficients for reflection/refraction
![](std1.bmp)

 -Subsurface Scattering
![](SSS3.bmp)

 -Depth of Field
![](DOF.0.bmp)

 -Polygon Mesh Support
![](Obj1.bmp)

##Performance Features:
 -Ray Parallel

 -Stream Compaction of Rays


###Feature Implementation Explained:

  -softshadow
  -area light
  -color bleeding
  -global illumination

  -Ray parallel instead of pixel parallel more maximum performance

  -Stream compaction on rays for each depth level (use thrust library)

  -BSDF using Russian Roulette 
  
  -Depth of Field, by jittering eye position and set image plane at focal length

  -Fresnel Coefficients for reflection/refraction

   Assuming light unpolarized

   ![](fresnel1.bmp)

   use 1/2 (RS + RP) to get coefficient for reflection

  -Caustics (free from above)

  -Obj loading (using tinyObjLoader), polygon mesh rendering

  -Anti-Alisasing jittered pixel position



###Performance Analysis




