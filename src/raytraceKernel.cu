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
#include "thrust\device_ptr.h"
#include "thrust\remove.h"
#include "thrust\count.h"

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
	
	glm::vec3 midPoint = eye + view;

	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);

	glm::vec3 axisV = B * (float)(glm::length(view) * tan(fov[1] * PI / 180.0f) / glm::length(B));	//fov[0]?
	glm::vec3 axisH = A * (glm::length(axisV) * resolution[0] / resolution[1] / glm::length(A));

	glm::vec3 h = axisH * (2 * x / (float) (resolution[0] - 1) - 1);
	glm::vec3 v = axisV * (2 * y / (float) (resolution[1] - 1) - 1);
	glm::vec3 p = midPoint + h + v;

	ray r;
	r.origin = midPoint;
	r.direction = glm::normalize(p - eye);
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
  int PBOindex = x + ((resolution.y-1-y) * resolution.x);
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
      PBOpos[PBOindex].w = 0;
      PBOpos[PBOindex].x = color.x;
      PBOpos[PBOindex].y = color.y;
      PBOpos[PBOindex].z = color.z;
  }
}

// pixel parallelization
__global__ void raytraceRayPP(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* cudaMaterials, int numberOfMaterials){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y)){

		glm::vec3 random = generateRandomNumberFromThread(cam.resolution, time, x, y);

		ray r = raycastFromCameraKernel(resolution, time, x + (2.0 * random.x - 1.0), y + (2.0 * random.y - 1.0), cam.position, cam.view, cam.up, cam.fov);

		int depth = 0;
		glm::vec3 color(1.0f, 1.0f, 1.0f);

		while(depth++ < rayDepth) {
			glm::vec3  intersect, normal;
			int geomId = -1;
			float nearestDist = intersectionTest(r, geoms, numberOfGeoms, intersect, normal, geomId);

			if (geomId == -1)
			{
				color = glm::vec3(0.0);
				break;
			}

			material curMaterial = cudaMaterials[geoms[geomId].materialid];

			if (curMaterial.emittance > 0.0f)  //light
			{
				color *= curMaterial.emittance * curMaterial.color;
				break;
			}

			glm::vec3 random2 = generateRandomNumberFromThread(cam.resolution, time+depth, x, y);
			if (curMaterial.hasReflective > EPSILON && curMaterial.hasRefractive > EPSILON)  //fresel
			{
				glm::vec3 reflDir, refrDir;
				Fresnel fresnel = calculateFresnel(normal, r.direction, 1.0f, curMaterial.indexOfRefraction, reflDir, refrDir);
				
				r.direction = (random2.x < fresnel.reflectionCoefficient) ? reflDir : refrDir;
			}
			else if (curMaterial.hasReflective > EPSILON)  //reflection
			{
				r.direction = calculateReflectionDirection(normal, r.direction);	//normalized
				color *= curMaterial.color;
			}
			else if (curMaterial.hasRefractive > EPSILON)  //refraction
			{
				float inOrOut = glm::dot(r.direction, normal);

				glm::vec3 refractedDirection = calculateTransmissionDirection(normal, r.direction, 1.0, curMaterial.indexOfRefraction);
				r.direction = refractedDirection;
				r.origin = intersect + 0.01f * r.direction;
				continue;
			}
			else		//diffuse
			{
				r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, random2.x, random2.y));
				color *= curMaterial.color;
			}
			r.origin = intersect + 0.001f * r.direction;
		}

		if (depth == rayDepth)
			color = glm::vec3(0.0);

		colors[index] = (colors[index] * time + color) / (float)(time+1);
		//colors[index] += color;
	}
}

__global__ void rayPixelInitialize(glm::vec2 resolution, float time, cameraData cam, rayPixel* rayPixels)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
  
	if(x<=resolution.x && y<=resolution.y){
		rayPixels[index].x = x;
		rayPixels[index].y = y;
		rayPixels[index].color = glm::vec3(1.0, 1.0, 1.0);
		rayPixels[index].isTerminated = false;

		glm::vec3 random = generateRandomNumberFromThread(cam.resolution, time, x, y);
		rayPixels[index].r = raycastFromCameraKernel(resolution, time, x + (2.0 * random.x - 1.0), y + (2.0 * random.y - 1.0), cam.position, cam.view, cam.up, cam.fov);
	}
}

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel, ray parallelization
__global__ void raytraceRay(int numRays, float time, cameraData cam, int depth, int rayDepth, rayPixel* rayPixels,
                            staticGeom* geoms, int numberOfGeoms, material* cudaMaterials, int numberOfMaterials){

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if((index < numRays)){

		rayPixel rayP = rayPixels[index];

		glm::vec3  intersect, normal;
		int geomId = -1;

		float nearestDist = intersectionTest(rayP.r, geoms, numberOfGeoms, intersect, normal, geomId);

		if (geomId == -1)
		{
			rayP.color = glm::vec3(0.0);
			rayP.isTerminated = true;
			rayPixels[index] = rayP;
			return;
		}

		material curMaterial = cudaMaterials[geoms[geomId].materialid];

		if (curMaterial.emittance > 0.0f)  //light
		{
			rayP.color *= (curMaterial.emittance * curMaterial.color);
			rayP.isTerminated = true;
			rayPixels[index] = rayP;
			return;
		}

		if (depth == rayDepth)
		{
			rayP.color = glm::vec3(0.0);
			rayP.isTerminated = true;
			rayPixels[index] = rayP;
			return;
		}

		glm::vec3 random2 = generateRandomNumberFromThread(cam.resolution, time+depth, rayP.x, rayP.y); 
		if (curMaterial.hasReflective > EPSILON && curMaterial.hasRefractive > EPSILON)  //fresel
		{
			glm::vec3 reflDir, refrDir;
			Fresnel fresnel = calculateFresnel(normal, rayP.r.direction, 1.0f, curMaterial.indexOfRefraction, reflDir, refrDir);
				
			rayP.r.direction = (random2.x < fresnel.reflectionCoefficient) ? reflDir : refrDir;
		}
		else if (curMaterial.hasReflective > EPSILON)  //reflection
		{
			rayP.r.direction = calculateReflectionDirection(normal, rayP.r.direction);	//normalized
		}
		else if (curMaterial.hasRefractive > EPSILON)  //refraction
		{
			rayP.r.direction = calculateTransmissionDirection(normal, rayP.r.direction, 1.0f, curMaterial.indexOfRefraction);
		}
		else		//diffuse
		{
			rayP.r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, random2.x, random2.y));
		}
		
		rayP.r.origin = intersect + 0.001f * rayP.r.direction;
		rayP.color *= curMaterial.color;
		rayPixels[index] = rayP;
	}
}

__global__ void colorAccumulator(int numRays, rayPixel* rayPixels, glm::vec2 resolution, glm::vec3* colors, float time)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numRays)
	{
		rayPixel rayP = rayPixels[index];
		if (rayP.isTerminated)
		{
			int colorId = rayP.x + rayP.y * resolution.x;
			colors[colorId] = (colors[colorId] * time + rayP.color) / (float)(time+1);
		}
	}
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,
						bool rayParallel){
  
  int traceDepth = 8; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
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
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // copy material
  material* cudaMaterials = NULL;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  cudaEvent_t startTime, stopTime;
  cudaEventCreate(&startTime);
  cudaEventCreate(&stopTime);
  cudaEventRecord(startTime, 0);

  // kernel launches
  if (rayParallel)
  {
	  rayPixel* rayPixels = NULL;
	  cudaMalloc((void**)&rayPixels, renderCam->resolution.x * renderCam->resolution.y * sizeof(rayPixel));
	  rayPixelInitialize<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, rayPixels);
	  cudaThreadSynchronize();

	  int depth = 0;
	  int numRays = renderCam->resolution.x * renderCam->resolution.y;
	  threadsPerBlock = dim3(tileSize*tileSize);

	  while(depth++ < traceDepth && numRays > 0) {
		  
		  //std::cout << depth-1 << ": " << numRays << std::endl;

		  fullBlocksPerGrid = dim3((int)ceil(numRays / float(tileSize*tileSize)));

		  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(numRays, (float)iterations, cam, depth, traceDepth, rayPixels, cudageoms, numberOfGeoms, cudaMaterials, numberOfMaterials);
		  colorAccumulator<<<fullBlocksPerGrid, threadsPerBlock>>>(numRays, rayPixels, renderCam->resolution, cudaimage, (float)iterations);
		  
		  //stream compaction
		  if (depth == traceDepth)
			  break;

		  thrust::device_ptr<rayPixel> beginItr(rayPixels);
		  thrust::device_ptr<rayPixel> endItr = beginItr + numRays;
		  endItr = thrust::remove_if(beginItr, endItr, isTerminated());
		  numRays = (int)(endItr - beginItr);

		  float duration = 0.0f;
		  cudaEventElapsedTime(&duration, startTime, stopTime);
		  std::cout << iterations << ": " << duration << std::endl;
	  }

	  dim3 threadsPerBlock(tileSize, tileSize);
	  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
	  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  }
  else
  {
	  //pixel parallel
	  raytraceRayPP<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudaMaterials, numberOfMaterials);
	  
	  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);
  }

  cudaThreadSynchronize();
  cudaEventRecord(stopTime, 0);
  cudaEventSynchronize(stopTime);

  float duration = 0.0f;
  cudaEventElapsedTime(&duration, startTime, stopTime);

  std::cout << iterations << ": " << duration << std::endl;
  
  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree(cudaMaterials);

  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
