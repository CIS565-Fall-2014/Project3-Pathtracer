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

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

// TODO: IMPLEMENT THIS FUNCTION
// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye;
  r.direction = view;
  //suppose the distance from eye to screen is 1
  float fovRad = fov.y / 180 * PI;
  float pixelHeight = view.length() * tan(fovRad /2) / (resolution.y /2);
  glm::vec2 screenCenterIdx(resolution.x / 2.0f, resolution.y / 2.0f);
  glm::vec3 screenCenter = getPointOnRay(r, 1);
  glm::vec3 right = glm::cross(view, up);
  glm::vec2 screenPointIdx((float)x - screenCenterIdx.x,(float)y - screenCenterIdx.y);
  glm::vec3 screenPoint = screenPointIdx.x * pixelHeight *right - screenPointIdx.y * pixelHeight * up + screenCenter;
  r.direction = glm::normalize(screenPoint - eye);
  return r;
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

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
__device__ float intersectionTest(staticGeom *geoms, int numberOfGeoms, ray currentRay, staticGeom *&intersectionGeom, glm::vec3 &intersectionPoint, glm::vec3 &intersectionNormal){
	float Tnear = 10000;
	for (size_t i = 0; i < numberOfGeoms; i++)
	{
		glm::vec3 tempPoint, tempNormal;
		float distance;
		if (geoms[i].type == CUBE)
		{
			distance = boxIntersectionTest(geoms[i], currentRay, tempPoint, tempNormal);
		}
		else if (geoms[i].type == SPHERE)
		{
			distance = sphereIntersectionTest(geoms[i], currentRay, tempPoint, tempNormal);
		}
		if (distance < Tnear && distance > 0)
		{
			Tnear = distance;
			intersectionPoint = tempPoint;
			intersectionNormal = tempNormal;
			intersectionGeom = &geoms[i];
		}
	}
	return Tnear;
}

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms,material* materials, int numberOfMaterials){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int arraySize = resolution.x *resolution.y;
  int bounceCount = 0;
  ray currentRay;
  staticGeom *intersectionGeom = NULL;
  glm::vec3 intersectionPoint(0), intersectionNormal(0);
  glm::vec3  color(0);
  currentRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);

  if((x<resolution.x && y<resolution.y))
  {
	  intersectionTest(geoms, numberOfGeoms, currentRay, intersectionGeom, intersectionPoint, intersectionNormal);
	  if (intersectionGeom != NULL)
	  {
			color = materials[intersectionGeom->materialid].color;
	  }
	  //got the nearest intersection distance Tnear and intersection points normals in the array;
	  while (++bounceCount < rayDepth&& intersectionGeom != NULL)
	  {
		  intersectionGeom = NULL;

		  currentRay.origin = intersectionPoint;
		  thrust::default_random_engine rng(hash(index*time));
		  thrust::uniform_real_distribution<float> u01(0, 1);
		  thrust::uniform_real_distribution<float> u02(0, 1);
		  //currentRay.direction = calculateRandomDirectionInHemisphere(intersectionNormal,u01(rng), u02(rng));
		  currentRay.direction = getRandomDirectionInSphere(index*time, index*time);
		  //currentRay.direction = intersectionNormal;
		  if (glm::dot(currentRay.direction, intersectionNormal) < 0 )
		  {
			  currentRay.direction *= -1;
		  }
		  float distance = intersectionTest(geoms, numberOfGeoms, currentRay, intersectionGeom, intersectionPoint, intersectionNormal);
		
		  if (intersectionGeom != NULL)
		  {
			  //calculate diffuse color
			  if (materials[intersectionGeom->materialid].emittance > 0)	//light
			  {
				  color *= materials[intersectionGeom->materialid].emittance * materials[intersectionGeom->materialid].color;
				  break;
			  }
			  if (bounceCount == rayDepth -1)
			  {
				  color *= 0;
				  break;
			  }
			  color *= materials[intersectionGeom->materialid].color;
		  }
		  else
		  {
			  color *= 0;
			  break;
		  } 
	  }
	  colors[index] = colors[index] / (time + 1) * time + color / (time + 1);
   }
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 5; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 16;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  // send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  material *materialList = new material[numberOfMaterials];
  for (size_t i = 0; i < numberOfMaterials; i++)
  {
	  material newMaterial;
	  newMaterial.color = materials[i].color;
	  newMaterial.emittance = materials[i].emittance;
	  newMaterial.absorptionCoefficient = materials[i].absorptionCoefficient;
	  newMaterial.specularColor = materials[i].specularColor;
	  newMaterial.hasReflective = materials[i].hasReflective;
	  newMaterial.hasRefractive = materials[i].hasRefractive;
	  newMaterial.hasScatter = materials[i].hasScatter;
	  newMaterial.indexOfRefraction = materials[i].indexOfRefraction;
	  newMaterial.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	  newMaterial.specularExponent = materials[i].specularExponent;
	  materialList[i] = newMaterial;
  }
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  material *dev_material = NULL;
  cudaMalloc((void**)&dev_material, numberOfMaterials * sizeof(material));
  cudaMemcpy(dev_material, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,dev_material,numberOfMaterials);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete[] geomList;
  delete[] materialList;
  cudaFree(dev_material);
  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
