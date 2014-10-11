// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include <iostream>

struct intersection{
  int objectID; //staticGeom object;
  glm::vec3 intersectionPoint;
  glm::vec3 normal;
  float distance;
  bool inside;
};

struct rayBounce{
  bool alive;
  ray r;
  int index;
  glm::vec3 color;
  intersection hit;

  rayBounce(){
    alive=true;
    index=0;
    color=glm::vec3(1,1,1);
  }

  rayBounce(int i){
    alive=true;
    index=i;
    color=glm::vec3(1,1,1);
  }
};

cameraData cam;
glm::vec3* cudaimage;
staticGeom* geomList;
int nLights;
staticGeom* cudageoms;
material* cudaMats;
staticGeom* lightList;
staticGeom* cudaLights;
glm::vec3** cudamesh;
glm::vec3* cudameshverts;
glm::vec3* cudameshtris;
glm::vec3** meshList;
int nGeoms;

rayBounce *raysA, *raysB, *cpuRays;
int *indicesA, *indicesB, *cpuIndices;

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float time){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0/time;
      color.y = image[index].y*255.0/time;
      color.z = image[index].z*255.0/time;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(float seed){
  thrust::default_random_engine rng(hash(seed));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov,
                        float focusDistance, float dof){
  ray r;
  r.origin = eye;

  glm::vec3 A = glm::cross(view,up);
  glm::vec3 B = glm::cross(A,view);
  float tempH = glm::length(view)*glm::tan(glm::radians(fov.x))/glm::length(A);
  glm::vec3 H = A*tempH;
  glm::vec3 V = B*(glm::length(view)*glm::tan(glm::radians(fov.y))/glm::length(B));

  glm::vec3 d = eye+view + float(2*(x/(resolution.x-1))-1)*H + float(1-2*(y/(resolution.y-1)))*V;
  r.direction = glm::normalize(d-eye);

  //dof
  glm::vec3 aimed = eye + focusDistance*r.direction;
  glm::vec3 random = generateRandomNumberFromThread(resolution, time, x,y);
  glm::vec3 start = eye + random*dof;
  r.direction = glm::normalize(aimed-start);
  r.origin    = start;

  //anti
  glm::vec3 jitter = generateRandomNumberFromThread(resolution, time, x,y);
  r.direction.x+=(jitter.x-1)*.0025;
  r.direction.y+=(jitter.y-1)*.0025;
  r.direction.z+=(jitter.z-1)*.0025;
  return r;
}

__host__ __device__ ray reflect(const ray& in, const intersection& hit){
  ray out;
  out.origin = in.origin;

  out.direction = glm::normalize(in.direction - 2.0f*glm::dot(hit.normal,in.direction)*hit.normal);
  out.origin = hit.intersectionPoint + 0.001f*out.direction;

  return out;
}

__host__ __device__ ray refract(const ray& in, const intersection& hit, float n1, float n2){
  
  ray out;
  
  if (hit.inside){
    float temp = n1;
    n1 = n2;
    n2 = temp;
  }

  float n = n1/n2;
  float cosThetaI = glm::dot((in.direction),(hit.normal));
  float sinThetaT2 = n*n*(1-cosThetaI*cosThetaI);
  float cosThetaT = sqrt(1-sinThetaT2);
  
  out.origin = hit.intersectionPoint;
  out.direction = n*in.direction + (n*cosThetaI - cosThetaT)*hit.normal;
  out.direction = glm::normalize(out.direction);
  out.origin += 0.001f*out.direction;

  return out;
}

__host__ __device__ ray diffuseReflection(const intersection& hit, glm::vec2 resolution, float time, int x, int y){
  ray out;
  //reflect ray in random hemisphere direction
  glm::vec3 seed = generateRandomNumberFromThread(resolution, time, x,y);
  out.direction = calculateRandomDirectionInHemisphere(hit.normal, seed.x, seed.y);
  out.origin = hit.intersectionPoint + 0.001f*out.direction;
  return out;
}

__host__ __device__ bool intersect(ray& r, staticGeom* geoms, int numberOfGeoms, intersection& hit, glm::vec3** cudaMesh){
  float distance = 100000000, check = -1;
  glm::vec3 ip, n;
  bool inside = false;
  for (int i=0; i<numberOfGeoms; i++){
    if (geoms[i].type==SPHERE){
      check = sphereIntersectionTest(geoms[i], r, ip, n, inside);
      if (check!=-1 && check<distance){
        distance = check;

        hit.objectID = i;
        hit.intersectionPoint = ip;
        hit.normal = n;
        hit.distance = distance;
        hit.inside = inside;
      }
    }
    else if (geoms[i].type==CUBE){
      check = boxIntersectionTest(geoms[i], r, ip, n, inside);
      if (check!=-1 && check<distance){
        distance = check;

        hit.objectID = i;
        hit.intersectionPoint = ip;
        hit.normal = n;
        hit.distance = distance;
        hit.inside = inside;
      }
    }
    else if (geoms[i].type==MESH){
      check = meshIntersectionTest(geoms[i], cudaMesh[i*2], cudaMesh[i*2+1], r, ip, n, inside);
      if (check!=-1 && check<distance){
        distance = check;

        hit.objectID = i;
        hit.intersectionPoint = ip;
        hit.normal = n;
        hit.distance = distance;
        hit.inside = inside;
      }
    }
  }
  if (distance<100000000){
    return true;
  }
  else{
    hit.distance = -1;
    hit.intersectionPoint = glm::vec3(0,0,0);
    hit.normal = glm::vec3(0,0,0);
    hit.objectID = -1;
    return false;
  }
}

__host__ __device__ float schlick(const intersection& hit, const ray& in, float n1, float n2){
  if (hit.inside){
    float temp = n1;
    n1 = n2;
    n2 = temp;
  }

  float n = n1/n2;
  float cosThetaI = -glm::dot((in.direction),(hit.normal));
  float sinThetaT2 = n*n*(1-cosThetaI*cosThetaI);
  float cosThetaT = sqrt(1-sinThetaT2);

  bool tir = sinThetaT2>=1;

  float R;
  float R0 = ((n1-n2)/(n1+n2))*((n1-n2)/(n1+n2));

  if (n1<=n2){
    float tempVal = 1-cosThetaI;
    R = R0 + (1-R0)*tempVal*tempVal*tempVal*tempVal*tempVal;
  }
  else if (!tir){
    float tempVal = 1-cosThetaT;
    R = R0 + (1-R0)*tempVal*tempVal*tempVal*tempVal*tempVal;
  }
  else{
    R = 1;
  }

  return R;
}

__host__ __device__ float cookTorrance(const glm::vec3& eye, const intersection& hit, const glm::vec3& lightPos,
                                       const material& mat, const ray& in){
  //cook torrance
  glm::vec3 EYE = glm::normalize(eye-hit.intersectionPoint);
  glm::vec3 L = glm::normalize(lightPos-hit.intersectionPoint);
  glm::vec3 H = glm::normalize(L + EYE);
  glm::vec3 N = hit.normal;

  //Beckmann distribution
  float alpha = acos(glm::dot(N, H));
  float m = mat.specularExponent;
  float cosAlpha = cos(alpha);

  float D = 100*exp(-(alpha*alpha)/(m*m));

  float F = schlick(hit, in, 1.000293, mat.indexOfRefraction);

  float G = min(1.0f, 2*glm::dot(H,N)*glm::dot(EYE,N)/glm::dot(EYE,H));
  G = min(G, 2*glm::dot(H,N)*glm::dot(L,N)/glm::dot(EYE,H));

  float result = D*F*G/(4*PI*glm::dot(EYE,N));
  result = glm::clamp(result, 0.0f, 1.0f);
  return result;
}

__host__ __device__ glm::vec3 diffuseWithShadow(const intersection& hit, staticGeom* geoms, int nGeoms, material* mats, int nMats, staticGeom* lights, int nLight,
                    const glm::vec2& resolution, float time, int x, int y, const glm::vec3& view, const ray& in, const glm::vec3& eye,
                    glm::vec3** cudaMesh){
  glm::vec3 out(0,0,0);
  ray r;
  intersection shadowHit;
  for (int i=0; i<nLight; i++){
    r.origin = hit.intersectionPoint;
    glm::vec3 endPoint = getRandomPointOnCube(lights[i], time*(x + (y * resolution.x)));
    r.direction = glm::normalize(endPoint-hit.intersectionPoint);
    r.origin += 0.001f*r.direction;
    if (intersect(r, geoms, nGeoms, shadowHit, cudaMesh) && mats[geoms[shadowHit.objectID].materialid].emittance>0){
      float kd = glm::dot(hit.normal, glm::normalize(lights[i].translation-hit.intersectionPoint));
      kd = glm::clamp(kd,0.0f,1.0f);

      //roughness for Oren-Nayar and Cook-Torrance
      float s = mats[geoms[hit.objectID].materialid].roughness;

      float A = 1-0.5f*s*s/(s*s+0.33f);
      float B = 0.45f*s*s/(s*s+0.09f);

      glm::vec3 toLight = glm::normalize(lights[i].translation-hit.intersectionPoint);
      glm::vec3 toEye   = glm::normalize(eye-hit.intersectionPoint);

      float thetaI = acos(kd);
      float thetaR = acos(glm::dot(toEye,hit.normal));

      float phiI = atan(toLight.y/toLight.x);
      float phiR = atan(toEye.y/toEye.x);

      float a = max(thetaR,thetaI);
      float b = min(thetaR,thetaI);

      kd = kd * (A + (B*max(0.0f, cos(phiI-phiR))*sin(a)*tan(b)));

      out += kd*mats[geoms[hit.objectID].materialid].color;

      // float dotBP = glm::dot(hit.normal, glm::normalize(r.direction+glm::normalize(-view)));
      // dotBP = glm::pow(glm::clamp(dotBP,0.0f,1.0f),40.0f);

      if (s>0){
        float kSpec = cookTorrance(eye, hit, endPoint, mats[geoms[hit.objectID].materialid], in);
        kSpec = glm::clamp(kSpec,0.0f,1.0f);
        glm::vec3 s = mats[geoms[hit.objectID].materialid].specularColor*kSpec;
        out += s;
      }
    }
    else{
      // return (mats[shadowHit.object.materialid].color);
    }
  }
  return out;
}

__host__ __device__ glm::vec3 diffuse(const intersection& hit, staticGeom* geoms, int nGeoms, material* mats, int nMats, staticGeom* lights, int nLight,
                    const glm::vec2& resolution, float time, int x, int y, const glm::vec3& view, const ray& in, const glm::vec3& eye){
  glm::vec3 out(0,0,0);
  ray r;
  intersection shadowHit;
  for (int i=0; i<nLight; i++){
    r.origin = hit.intersectionPoint;
    glm::vec3 endPoint = getRandomPointOnCube(lights[i], time*(x + (y * resolution.x)));
    r.direction = glm::normalize(endPoint-hit.intersectionPoint);
    r.origin += 0.001f*r.direction;
    float kd = glm::dot(hit.normal, glm::normalize(lights[i].translation-hit.intersectionPoint));
    kd = glm::clamp(kd,0.0f,1.0f);
    
    //roughness for Oren-Nayar and Cook-Torrance
    float s = mats[geoms[hit.objectID].materialid].roughness;

    float A = 1-0.5f*s*s/(s*s+0.33f);
    float B = 0.45f*s*s/(s*s+0.09f);

    glm::vec3 toLight = glm::normalize(lights[i].translation-hit.intersectionPoint);
    glm::vec3 toEye   = glm::normalize(eye-hit.intersectionPoint);

    float thetaI = acos(kd);
    float thetaR = acos(glm::dot(toEye,hit.normal));

    float phiI = atan(toLight.y/toLight.x);
    float phiR = atan(toEye.y/toEye.x);

    float a = max(thetaR,thetaI);
    float b = min(thetaR,thetaI);

    kd = kd * (A + (B*max(0.0f, cos(phiI-phiR))*sin(a)*tan(b)));

    out += kd*mats[geoms[hit.objectID].materialid].color;

    // float dotBP = glm::dot(hit.normal, glm::normalize(r.direction+glm::normalize(-view)));
    // dotBP = glm::pow(glm::clamp(dotBP,0.0f,1.0f),40.0f);

    if (s>0){
      float kSpec = cookTorrance(eye, hit, endPoint, mats[geoms[hit.objectID].materialid], in);
      kSpec = glm::clamp(kSpec,0.0f,1.0f);
      glm::vec3 s = mats[geoms[hit.objectID].materialid].specularColor*kSpec;
      out += s;
    }
  }
  return out;
}

__host__ __device__ ray fresnel(const ray& r, const material& mat, const intersection& hit,
                                float seed){
  ray fresRay = r;

  float n1 = 1.000293;
  float n2 = mat.indexOfRefraction;

  float R = schlick(hit, r, n1,n2);
  float randomCheck = generateRandomNumberFromThread(seed).x;

  if (randomCheck<R){
    fresRay = reflect(fresRay,hit);
  }
  else{
    fresRay = refract(fresRay,hit,n1,n2);
  }

  return fresRay;
}

__global__ void lighttraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int nMats, 
                            staticGeom* lights, int nLights){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  staticGeom source = lights[0];
  glm::vec3 startPoint = getRandomPointOnCube(source, time*(x + (y * resolution.x)));
  glm::vec3 seed = generateRandomNumberFromThread(resolution, time, x,y);
  glm::vec3 startDir = glm::normalize(getRandomDirectionInSphere(seed.x, seed.y));

  ray r;
  r.origin = startPoint + 0.001f*startDir;
  r.direction = startDir;

  glm::vec3 A = glm::cross(cam.view,cam.up);
  glm::vec3 B = glm::cross(A,cam.view);
  float tempH = glm::length(cam.view)*glm::tan(glm::radians(cam.fov.x))/glm::length(A);
  glm::vec3 H = A*tempH;
  glm::vec3 V = B*(glm::length(cam.view)*glm::tan(glm::radians(cam.fov.y))/glm::length(B));

  glm::vec3 testDir = glm::normalize(startPoint - cam.position);
  glm::vec3 p = cam.position + testDir;

  glm::vec3 m = cam.position + cam.view;
  glm::vec3 mv = m+V;
  glm::vec3 mh = m+H;

  glm::vec3 ip;
  ip.y = mv.y-p.y;
  ip.x = resolution.x - (mh.x-p.x);

  // if ((x<=resolution.x && y<=resolution.y)){
  //   // if (ip.x<=resolution.x && ip.y<=resolution.y){
  //   //   colors[index]+=glm::vec3(1,1,0);
  //   // }
  //   // else{
  //   //   colors[index]+=glm::vec3(0,0,1);
  //   // }
  //   if (abs(glm::length(p-cam.position)-1)>0.00001f || abs(glm::length(m-cam.position)-1)>0.00001f){
  //     colors[index]+=glm::vec3(1,1,1);
  //   }
  // }
  if (true){
    index = (ip.y*resolution.x);
    colors[index] += glm::vec3(1,0,1);
  }
}

__global__ void pathtraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int nMats, 
                            staticGeom* lights, int nLights, glm::vec3** cudaMesh){

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if((x<=resolution.x && y<=resolution.y)){
     
      ray r = raycastFromCameraKernel(resolution, time, x,y, cam.position, cam.view, cam.up, cam.fov, cam.focusDistance, cam.dof);
      intersection hit;
      int depth=0;
      glm::vec3 pixelColor = glm::vec3(1,1,1);

      while (depth<100){
        if (!intersect(r, geoms, numberOfGeoms, hit,cudaMesh)){
          pixelColor = glm::vec3(0,0,0);
          break;
        }
        material mat = mats[geoms[hit.objectID].materialid];
        depth++;

        if (mat.emittance>0){ //Stop tracing when you hit a light
          pixelColor = pixelColor*mat.color*mat.emittance;
          break;
        }
        else{
          pixelColor = pixelColor*mat.color; //Non lights attenuate the color

          if (mat.hasReflective && mat.hasRefractive){  //Fresnel
            r=fresnel(r, mat, hit, index*time*(rayDepth)/2.0f);
          }
          else if (mat.hasReflective){ //mirror
            r = reflect(r,hit); 
          }
          else if (mat.hasRefractive){ //glass
            float n1 = 1.000293;
            float n2 = mat.indexOfRefraction;

            r = refract(r,hit,n1,n2);
            // pixelColor = pixelColor*mat.color;
          }
          else{ //diffuse
            //Add direct lighting contribution for specularity (Cook-Torrance microfacet model)
            if(mat.specularExponent>0.0){
              for (int i=0; i<nLights; i++){
                glm::vec3 lightPos = getRandomPointOnCube(lights[i], time*index*(rayDepth)/2.0f);
                float ct = cookTorrance(cam.position, hit, lightPos, mat, r);
                pixelColor+=ct*mat.specularColor;
              }
            }
            // r = diffuseReflection(hit, resolution, time, x,y);
            ray out;
            // reflect ray in random hemisphere direction
            glm::vec3 seed = generateRandomNumberFromThread(time*index*(rayDepth)/2.0f);
            out.direction = calculateRandomDirectionInHemisphere(hit.normal, seed.x, seed.y);
            out.origin = hit.intersectionPoint + 0.001f*out.direction;
            r = out;
          }
        }
      }

      colors[index] += pixelColor;
     }
}

//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int nMats, 
                            staticGeom* lights, int nLights, glm::vec3** cudaMesh){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
    ray r = raycastFromCameraKernel(resolution, time, x,y, cam.position, cam.view, cam.up, cam.fov, cam.focusDistance, cam.dof);
    intersection hit;

    int depth=0;

    float multColor[3];
    multColor[0]=1; multColor[1]=1; multColor[2]=1;
    while (intersect(r, geoms, numberOfGeoms, hit, cudaMesh) && depth<10){
      material mat = mats[geoms[hit.objectID].materialid];
      depth++;

      if (mat.emittance>0){
        colors[index] += mat.color*mat.emittance;
        break;
      }
      else if (mat.hasReflective && mat.hasRefractive){
        r=fresnel(r, mat, hit, time*index*(rayDepth+1));
        multColor[0] = multColor[0]*mat.color.x;
        multColor[1] = multColor[1]*mat.color.y;
        multColor[2] = multColor[2]*mat.color.z;
      }
      else if (mat.hasReflective){
        r = reflect(r,hit);
        multColor[0] = multColor[0]*mat.color.x;
        multColor[1] = multColor[1]*mat.color.y;
        multColor[2] = multColor[2]*mat.color.z;
      }
      else if (mat.hasRefractive){
        float n1 = 1.000293;
        float n2 = mat.indexOfRefraction;

        r = refract(r,hit,n1,n2);
        multColor[0] = multColor[0]*mat.color.x;
        multColor[1] = multColor[1]*mat.color.y;
        multColor[2] = multColor[2]*mat.color.z;
      }
      else{ //diffuse
        glm::vec3 diffColor = diffuseWithShadow(hit, geoms, numberOfGeoms, mats, nMats, lights, nLights, resolution, time, x, y, cam.view, r, cam.position, cudaMesh);
        // glm::vec3 diffColor = diffuse(hit, geoms, numberOfGeoms, mats, nMats, lights, nLights, resolution, time, x, y, cam.view, r, cam.position);
        
        //reflected attentuation
        diffColor.x*=multColor[0];
        diffColor.y*=multColor[1];
        diffColor.z*=multColor[2];

        colors[index]+=diffColor;
        
        //ambient
        colors[index] += 0.1f*mat.color;
        break;
      }
    }
   }
}

//Core raytracer kernel
__global__ void matID(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int nMats, 
                            staticGeom* lights, int nLights, glm::vec3** cudaMesh){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
    ray r = raycastFromCameraKernel(resolution, time, x,y, cam.position, cam.view, cam.up, cam.fov, cam.focusDistance, cam.dof);
    intersection hit;

    int depth=0;

    float multColor[3];
    multColor[0]=1; multColor[1]=1; multColor[2]=1;
    while (intersect(r, geoms, numberOfGeoms, hit, cudaMesh) && depth<100){
      material mat = mats[geoms[hit.objectID].materialid];
      depth++;

      if (mat.emittance>0){
        colors[index] += mat.color*mat.emittance;
        break;
      }
      else if (mat.hasReflective){
        r = reflect(r,hit);
      }
      else{ //diffuse
        glm::vec3 diffColor = mat.color;
        // glm::vec3 diffColor = diffuse(hit, geoms, numberOfGeoms, mats, nMats, lights, nLights, resolution, time, x, y, cam.view, r, cam.position);
        
        //reflected attentuation
        diffColor.x*=multColor[0];
        diffColor.y*=multColor[1];
        diffColor.z*=multColor[2];

        colors[index]+=diffColor;
        break;
      }
    }
   }
}

__global__ void sum(int* in, int* out, int n, int d1){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  if (k<n){
    int ink = in[k];
    if (k>=d1){
      out[k] = in[k-d1] + ink;
    }
    else{
      out[k] = ink;
    }
  }
}

__global__ void shift(int* in, int* out, int n){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  out[0] = 0;
  if (k<n && k>0){
    out[k] = in[k-1];
  }
}

__global__ void streamCompaction(rayBounce* inRays, int* indices, rayBounce* outRays, int numRays){
  int k = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (k<numRays){
    rayBounce inRay = inRays[k];
    if (inRay.alive){
      outRays[indices[k]-1] = inRay;
    }
  }
}

__global__ void firstBounce(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int nMats, 
                            staticGeom* lights, int nLights, glm::vec3** cudaMesh, rayBounce* rays, int* in
                            ){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
    ray r = raycastFromCameraKernel(resolution, time, x,y, cam.position, cam.view, cam.up, cam.fov, cam.focusDistance, cam.dof);
    intersection hit;

    rays[index].r=r;
    rays[index].alive=true;
    rays[index].index = index;
    in[index]=1;

    if (intersect(r, geoms, numberOfGeoms, hit, cudaMesh)){
      material mat = mats[geoms[hit.objectID].materialid];
      if (mat.emittance>0){
        rays[index].alive=false;
        in[index]=0;
        colors[index]+=mat.color*mat.emittance;
      }
      else{

        rays[index].color=mat.color;

        if (mat.hasReflective && mat.hasRefractive){
          rays[index].r = fresnel(r, mat, hit, time*index*float(rayDepth)/2.0f);
        }
        else if (mat.hasReflective){
          rays[index].r = reflect(r,hit);
        }
        else if (mat.hasRefractive){
          float n1 = 1.000293;
          float n2 = mat.indexOfRefraction;

          rays[index].r = refract(r,hit,n1,n2);
          // pixelColor = pixelColor*mat.color;
        }
        else{ 

          //Add direct lighting contribution for specularity (Cook-Torrance microfacet model)
          if(mat.specularExponent>0.0){
            for (int i=0; i<nLights; i++){
              glm::vec3 lightPos = getRandomPointOnCube(lights[i], time*index*(rayDepth)/2.0f);
              float ct = cookTorrance(cam.position, hit, lightPos, mat, r);
              rays[index].color+=ct*mat.specularColor;
            }
          }

          ray out;
          // reflect ray in random hemisphere direction
          glm::vec3 seed = generateRandomNumberFromThread(time*index*float(rayDepth)/2.0f);
          out.direction = calculateRandomDirectionInHemisphere(hit.normal, seed.x, seed.y);
          // out.direction = getRandomDirectionInSphere(seed.x, seed.y);
          // if (glm::dot(out.direction, hit.normal)<0){
            // out.direction*=-1.0f;
          // }
          out.origin = hit.intersectionPoint + 0.001f*out.direction;
          rays[index].r = out;
        }
      }
    }
    else{
      rays[index].alive=false;
      in[index]=0;
    }
}
}
__global__ void bounce(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, int nMats, 
                            staticGeom* lights, int nLights, glm::vec3** cudaMesh, rayBounce* rays, int* in,
                            int numRays){
  
  int id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(id<numRays){
    ray r = rays[id].r;
    intersection hit;

    rays[id].alive=true;
    in[id]=1;

    if (intersect(r, geoms, numberOfGeoms, hit, cudaMesh)){
      material mat = mats[geoms[hit.objectID].materialid];
      if (mat.emittance>0){
        rays[id].alive=false;
        in[id]=0;
        colors[rays[id].index]+=(rays[id].color*mat.color*mat.emittance);
      }
      else{

        rays[id].color*=mat.color;

        if (mat.hasReflective && mat.hasRefractive){
          rays[id].r = fresnel(r, mat, hit, time*id*float(rayDepth)/2.0f);
        }
        else if (mat.hasReflective){
          rays[id].r = reflect(r,hit);
        }
        else if (mat.hasRefractive){
          float n1 = 1.000293;
          float n2 = mat.indexOfRefraction;

          rays[id].r = refract(r,hit,n1,n2);
          // pixelColor = pixelColor*mat.color;
        }
        else{ 

          //Add direct lighting contribution for specularity (Cook-Torrance microfacet model)
          if(mat.specularExponent>0.0){
            for (int i=0; i<nLights; i++){
              glm::vec3 lightPos = getRandomPointOnCube(lights[i], time*id*(rayDepth)/2.0f);
              float ct = cookTorrance(cam.position, hit, lightPos, mat, r);
              rays[id].color+=ct*mat.specularColor;
            }
          }

          ray out;
          // reflect ray in random hemisphere direction
          glm::vec3 seed = generateRandomNumberFromThread(time*id*float(rayDepth)/2.0f);
          out.direction = calculateRandomDirectionInHemisphere(hit.normal, seed.x, seed.y);
          // out.direction = getRandomDirectionInSphere(seed.x, seed.y);
          // if (glm::dot(out.direction, hit.normal)<0){
            // out.direction*=-1.0f;
          // }
          out.origin = hit.intersectionPoint + 0.001f*out.direction;
          rays[id].r = out;
        }
      }
    }
    else{
      rays[id].alive=false;
      in[id]=0;
      rays[id].color=glm::vec3(0,0,0);
    }
}
}

void cudaInit(uchar4* pos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms,
                    int numberOfGeoms){
  
  //send image to GPU
  // = NULL;

  std::cout<<"Initializing memory"<<std::endl;

  cudaimage=NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
  geomList = new staticGeom[numberOfGeoms];
  nLights=0;

  nGeoms = 2*numberOfGeoms;
  meshList = new glm::vec3*[nGeoms];

  for(int i=0; i<numberOfGeoms; i++){
    if (materials[geoms[i].materialid].emittance>0){
      nLights++;
    }
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;

    glm::vec3 translation = geoms[i].translations[frame];
    glm::vec3 rotation = geoms[i].rotations[frame];
    glm::vec3 scale = geoms[i].scales[frame];
    cudaMat4 transform = geoms[i].transforms[frame];
    cudaMat4 inverseTransform = geoms[i].inverseTransforms[frame];

    if (frame>0 && frame<renderCam->frames){ //middle frame
      float t1 = float(rand())/RAND_MAX;
      float t2 = float(rand())/RAND_MAX;
      float t3 = float(rand())/RAND_MAX;

      glm::vec3 translation1 = t1*geoms[i].translations[frame-1] + (1-t1)*geoms[i].translations[frame];
      glm::vec3 translation2 = t2*geoms[i].translations[frame] + (1-t2)*geoms[i].translations[frame+1];

      translation = t3*translation1 + (1-t3)*translation2;

      glm::vec3 scale1 = t1*geoms[i].scales[frame-1] + (1-t1)*geoms[i].scales[frame];
      glm::vec3 scale2 = t2*geoms[i].scales[frame] + (1-t2)*geoms[i].scales[frame+1];

      scale = t3*scale1 + (1-t3)*scale2;

      glm::vec3 rotation1 = t1*geoms[i].rotations[frame-1] + (1-t1)*geoms[i].rotations[frame];
      glm::vec3 rotation2 = t2*geoms[i].rotations[frame] + (1-t2)*geoms[i].rotations[frame+1];

      rotation = t3*rotation1 + (1-t3)*rotation2;

      glm::mat4 temp = utilityCore::buildTransformationMatrix(translation, rotation, scale);
      transform = utilityCore::glmMat4ToCudaMat4(temp);
      inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(temp));

    }

    newStaticGeom.translation = translation;
    newStaticGeom.rotation = rotation;
    newStaticGeom.scale = scale;
    newStaticGeom.transform = transform;
    newStaticGeom.inverseTransform = inverseTransform;
    
    if (newStaticGeom.type==MESH){
      newStaticGeom.obj.numTris = geoms[i].obj.numTris;
      newStaticGeom.obj.numVerts = geoms[i].obj.numVerts;
      newStaticGeom.obj.tris = geoms[i].obj.tris;
      newStaticGeom.obj.verts = geoms[i].obj.verts;

      //package meshes
      cudaMalloc((void**)&cudameshtris, newStaticGeom.obj.numTris*sizeof(glm::vec3));
      cudaMemcpy( cudameshtris, newStaticGeom.obj.tris, newStaticGeom.obj.numTris*sizeof(glm::vec3), cudaMemcpyHostToDevice);

      cudaMalloc((void**)&cudameshverts, newStaticGeom.obj.numVerts*sizeof(glm::vec3));
      cudaMemcpy( cudameshverts, newStaticGeom.obj.verts, newStaticGeom.obj.numVerts*sizeof(glm::vec3), cudaMemcpyHostToDevice);

      meshList[i*2] = cudameshtris;
      meshList[i*2+1] = cudameshverts;
    }
    else{
      meshList[i*2] = 0;
      meshList[i*2+1] = 0;
    }
    newStaticGeom.meshid = geoms[i].meshid;
    geomList[i] = newStaticGeom;
  }
  
  //package geom
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&cudamesh, nGeoms*sizeof(glm::vec3*));
  cudaMemcpy( cudamesh, meshList, nGeoms*sizeof(glm::vec3*), cudaMemcpyHostToDevice);

  //package materials
  cudaMalloc((void**)&cudaMats, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudaMats, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package lights
  lightList = new staticGeom[nLights];
  int lightIndex = 0;
  for (int i=0; i<numberOfGeoms; i++){
  if (materials[geomList[i].materialid].emittance>0){
      lightList[lightIndex] = geomList[i];
      lightIndex++;
    }
  }

  cudaMalloc((void**)&cudaLights, nLights*sizeof(staticGeom));
  cudaMemcpy(cudaLights, lightList, nLights*sizeof(staticGeom), cudaMemcpyHostToDevice);

  //package camera
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];

  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.dof = renderCam->dof;
  cam.focusDistance = renderCam->focusDistance;

  int numRays = renderCam->resolution.x*renderCam->resolution.y;
  cpuRays = new rayBounce[numRays];
  cpuIndices = new int[numRays];

  for (int i=0; i<numRays; i++){
    cpuIndices[i] = 1;
  }

  cudaMalloc((void**)&raysA, numRays*sizeof(rayBounce));
  cudaMemcpy(raysA, cpuRays, numRays*sizeof(rayBounce), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&raysB, numRays*sizeof(rayBounce));
  cudaMemcpy(raysB, cpuRays, numRays*sizeof(rayBounce), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&indicesA, numRays*sizeof(int));
  cudaMemcpy(indicesA, cpuIndices, numRays*sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&indicesB, numRays*sizeof(int));
  cudaMemcpy(indicesB, cpuIndices, numRays*sizeof(int), cudaMemcpyHostToDevice);

  std::cout<<"Done allocationg"<<std::endl;
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCoreStream(uchar4* PBOpos, camera* renderCam, int frame, int iterations , int numberOfMaterials, geom* geoms, int numberOfGeoms){

  int traceDepth = 1; //determines how many bounces the raytracer traces

  for (int i=0; i<numberOfGeoms; i++){
    glm::vec3 translation = geoms[i].translations[frame];
    glm::vec3 rotation = geoms[i].rotations[frame];
    glm::vec3 scale = geoms[i].scales[frame];
    cudaMat4 transform = geoms[i].transforms[frame];
    cudaMat4 inverseTransform = geoms[i].inverseTransforms[frame];

    if (frame>0 && frame<renderCam->frames){ //middle frame
      float t1 = float(rand())/RAND_MAX;
      float t2 = float(rand())/RAND_MAX;
      float t3 = float(rand())/RAND_MAX;

      glm::vec3 translation1 = t1*geoms[i].translations[frame-1] + (1-t1)*geoms[i].translations[frame];
      glm::vec3 translation2 = t2*geoms[i].translations[frame] + (1-t2)*geoms[i].translations[frame+1];

      translation = t3*translation1 + (1-t3)*translation2;

      glm::vec3 scale1 = t1*geoms[i].scales[frame-1] + (1-t1)*geoms[i].scales[frame];
      glm::vec3 scale2 = t2*geoms[i].scales[frame] + (1-t2)*geoms[i].scales[frame+1];

      scale = t3*scale1 + (1-t3)*scale2;

      glm::vec3 rotation1 = t1*geoms[i].rotations[frame-1] + (1-t1)*geoms[i].rotations[frame];
      glm::vec3 rotation2 = t2*geoms[i].rotations[frame] + (1-t2)*geoms[i].rotations[frame+1];

      rotation = t3*rotation1 + (1-t3)*rotation2;

      glm::mat4 temp = utilityCore::buildTransformationMatrix(translation, rotation, scale);
      transform = utilityCore::glmMat4ToCudaMat4(temp);
      inverseTransform = utilityCore::glmMat4ToCudaMat4(glm::inverse(temp));

      geomList[i].translation = translation;
      geomList[i].rotation = rotation;
      geomList[i].scale = scale;
      geomList[i].transform = transform;
      geomList[i].inverseTransform = inverseTransform;
    }
  }
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  int numRays = int(renderCam->resolution.x*renderCam->resolution.y);

  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  // cudaMemcpy(indicesA, cpuIndices, numRays*sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(raysA, cpuRays, numRays*sizeof(rayBounce), cudaMemcpyHostToDevice);

  rayBounce* streamA = raysA;
  rayBounce* streamB = raysB;

  firstBounce<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, 
                                               numberOfGeoms, cudaMats, numberOfMaterials, cudaLights, nLights, cudamesh,
                                               raysA, indicesA);

  dim3 threadsPerBlockL(64);
  dim3 fullBlocksPerGridL(int(ceil(float(numRays)/64.0f)));

  while(traceDepth<100){

    traceDepth++;

    //scan algorithm
    for (int d=1; d<=ceil(log(float(numRays))/log(2.0f)); d++){
      sum<<<fullBlocksPerGridL, threadsPerBlockL>>>(indicesA, indicesB, numRays, powf(2.0f, d-1));
      int* temp = indicesA;
      indicesA = indicesB;
      indicesB = temp;
    }

    //Stream compation from A into B, then save back into A
    streamCompaction<<<fullBlocksPerGridL, threadsPerBlockL>>>(streamA, indicesA, streamB, numRays);
    rayBounce* temp = streamA;
    streamA = streamB;
    streamB = streamA;

    //update numrays
    cudaMemcpy(&numRays, &indicesA[numRays-1], sizeof(int), cudaMemcpyDeviceToHost);
    fullBlocksPerGridL = dim3(int(ceil(float(numRays)/64.0f)));

    //subsequent bounces...
    if (numRays>0){
      bounce<<<fullBlocksPerGridL, threadsPerBlockL>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, 
                                               numberOfGeoms, cudaMats, numberOfMaterials, cudaLights, nLights, cudamesh,
                                               streamA, indicesA, numRays);
    }
    else{
      break;
    }

  }
  // // std::cout<<traceDepth<<std::endl;

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);
  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  // make certain the kernel has completed
  cudaThreadSynchronize();
  checkCUDAError("kernel failed!");
}
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, 
                    geom* geoms, int numberOfGeoms){

  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudaMats, numberOfMaterials, cudaLights, nLights, cudamesh);
  // pathtraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudaMats, numberOfMaterials, cudaLights, nLights, cudamesh);
  
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // make certain the kernel has completed
  cudaThreadSynchronize();
  checkCUDAError("kernel failed!");
}

void testStream(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, 
                    geom* geoms, int numberOfGeoms){

  int traceDepth = 1; //determines how many bounces the raytracer traces

  int numRays = int(renderCam->resolution.x*renderCam->resolution.y);
  numRays=1500;

  rayBounce* testRays = new rayBounce[numRays];

  for (int i=0; i<numRays; i++){
    rayBounce tr(i);
    testRays[i] = tr;
  }

  testRays[1].alive=false;
  testRays[4].alive=false;
  testRays[5].alive=false;
  testRays[7].alive=false;
  testRays[9].alive=false;
  testRays[12].alive=false;
  testRays[19].alive=false;
  testRays[14].alive=false;

  rayBounce* cudaRaysA;
  rayBounce* cudaRaysB;

  cudaMalloc((void**)&cudaRaysA, numRays*sizeof(rayBounce));

  cudaMalloc((void**)&cudaRaysB, numRays*sizeof(rayBounce));

  int* testin;
  int* testout;
  int* cputest = new int[numRays];

  for (int i=0; i<numRays; i++){
    if (testRays[i].alive){
      cputest[i]=1;
    }
    else{
      cputest[i]=0;
    }
  }

  cudaMalloc((void**)&testin, numRays*sizeof(int));
  cudaMalloc((void**)&testout, numRays*sizeof(int));


  cudaMemcpy(cudaRaysA, testRays, numRays*sizeof(rayBounce), cudaMemcpyHostToDevice);  
  cudaMemcpy(cudaRaysB, testRays, numRays*sizeof(rayBounce), cudaMemcpyHostToDevice);  
  cudaMemcpy(testin, cputest, numRays*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(testout, cputest, numRays*sizeof(int), cudaMemcpyHostToDevice);


  //start loop
  while(numRays>0 && traceDepth<10000){
    
    for (int i=0; i<numRays; i++){
      std::cout<<testRays[i].index<<", "<<cputest[i]<<std::endl;
    }

    dim3 threadsPerBlock(64);
    dim3 fullBlocksPerGrid(int(ceil(float(numRays)/64.0f)));

    //scan
    for (int d=1; d<=ceil(log(float(numRays))/log(2.0f))+1; d++){
      sum<<<fullBlocksPerGrid, threadsPerBlock>>>(testin, testout, numRays, int(pow(2.0f,d-1)));
      cudaThreadSynchronize();
      cudaMemcpy(cputest, testout, numRays*sizeof(int), cudaMemcpyDeviceToHost);

      // std::cout<<"sum at depth: "<<d<<" (using "<<testout<<")"<<std::endl;
      // for (int i=0; i<numRays; i++){
      //   std::cout<<cputest[i]<<std::endl;
      // }

      int* temp = testin;
      testin=testout;
      testout=temp;
    }
    //Compact
    streamCompaction<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaRaysA, testin, cudaRaysB, numRays);
    cudaRaysA = cudaRaysB;
    cudaThreadSynchronize();

    cudaMemcpy(&numRays, &testin[numRays-1], 1*sizeof(int), cudaMemcpyDeviceToHost);


    std::cout<<"number of rays left: "<<numRays<<std::endl;

    if (numRays==0) break;

    // for (int i=0; i<numRays; i++){
    //   std::cout<<cputest[i]<<std::endl;
    // }    

    cudaMemcpy(cputest, testin, numRays*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(testRays, cudaRaysA, numRays*sizeof(rayBounce), cudaMemcpyDeviceToHost);


    for (int i=0; i<numRays; i++){
      std::cout<<testRays[i].index<<std::endl;
    }
    std::cout<<"___________________________________"<<std::endl;

    //kill some amount
    if (numRays>0) testRays[0].alive=false;
    if (numRays>3) testRays[3].alive=false;
    if (numRays>10) testRays[10].alive=false;
    if (numRays>13) testRays[13].alive=false;
    if (numRays>30) testRays[30].alive=false;
    cudaMemcpy(cputest, testin, numRays*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0; i<numRays; i++){
      if (testRays[i].alive){
        cputest[i]=1;
      }
      else{
        cputest[i]=0;
      }
    }
    cudaMemcpy(cudaRaysA, testRays, numRays*sizeof(rayBounce), cudaMemcpyHostToDevice);  
    cudaMemcpy(testin, cputest, numRays*sizeof(int), cudaMemcpyHostToDevice);

    traceDepth++;

    //end loop
  }

  delete [] cputest;
  cudaFree(testin);
  cudaFree(testout);

  delete [] testRays;
  cudaFree(cudaRaysA);
  cudaFree(cudaRaysB);
  
  // make certain the kernel has completed
  cudaThreadSynchronize();
  checkCUDAError("kernel failed!");
}

void cudaFreeCPU(){
  //free up stuff, or else we'll leak memory like a madman
  std::cout<<"Freeing cuda memory"<<std::endl;
  cudaFree( cudaimage  );
  cudaFree( cudageoms  );
  cudaFree( cudaMats   );
  cudaFree( cudaLights );
  cudaFree( indicesA );
  cudaFree( indicesB );
  cudaFree( raysA );
  cudaFree( raysB );
  for (int i=0; i<nGeoms; i++){
    if (meshList[i]!=0) {
      cudaFree( meshList[i] );
    }
  }
  cudaFree( cudamesh );
  delete [] meshList;
  delete [] geomList;
  delete [] lightList;
  delete [] cpuRays;
  delete [] cpuIndices;
}