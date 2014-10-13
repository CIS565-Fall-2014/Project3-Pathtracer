// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <time.h>  
#include <math.h> 

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

#define RAY_DEPTH 10
#define TILE_SIZE 8
//#define DOF 11.2
//#define MOTION_BLUR 1

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 
struct  pixelRayUnit
{
	ray pixelRay;
	int index,x,y;
	bool isFinished;
	bool currentDepth;
};
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
	float Tnear = FLT_MAX;
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
		if (distance < Tnear && distance != -1.0f)
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
  glm::vec3  color(1,1,1);
  currentRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
#ifdef DOF
  float focalLength = DOF;
  glm::vec3 aimingPosition = currentRay.origin + focalLength * currentRay.direction;
  glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time * 3, x, y);
  glm::vec3 camPosition = glm::vec3(cam.position.x + (float)rand.x, cam.position.y + (float)rand.y, cam.position.z + (float)rand.z);
  currentRay.origin = camPosition;
  currentRay.direction = glm::normalize(aimingPosition - camPosition);
#endif
  if((x<resolution.x && y<resolution.y))
  {
	  while (++bounceCount < rayDepth)
	  {
		  staticGeom *bounceGeom = NULL;
		  glm::vec3 bouncePoint, bounceNormal;
		  intersectionTest(geoms, numberOfGeoms, currentRay, bounceGeom, bouncePoint, bounceNormal);
		  glm::vec3 rnd = generateRandomNumberFromThread(cam.resolution, time + (bounceCount + 1), x, y);
		 
		  if (bounceGeom != NULL)
		  {
			
			  //calculate diffuse color
			  if (materials[bounceGeom->materialid].emittance > 0)	//light
			  {
				  color *= materials[bounceGeom->materialid].emittance * materials[bounceGeom->materialid].color;
				  break;
			  }
			  if (bounceCount == rayDepth -1)
			  {
				  color *= 0;
				  break;
			  }

			  glm::vec3 nextRayDir = calculateRandomDirectionInHemisphere(bounceNormal,rnd.x,rnd.y);

			  glm::vec3 H = glm::normalize(nextRayDir - currentRay.direction);
			  float NdotH = glm::dot(H, bounceNormal);
			  if (materials[bounceGeom->materialid].hasRefractive <= 0)
			  {
				 
				  color *= materials[bounceGeom->materialid].color;
			  }
			  	  //*************has reflection********************
			  if (materials[bounceGeom->materialid].hasReflective && rnd.z < materials[bounceGeom->materialid].hasReflective)
			  {
				  Fresnel fresnel;
				  glm::vec3 reflectionDir = calculateReflectionDirection(bounceNormal, currentRay.direction);
				  float io = glm::dot(currentRay.direction, bounceNormal);
				  currentRay.origin = bouncePoint + 0.001f * reflectionDir;
				  currentRay.direction = reflectionDir;

				  continue;
			  }
			  //********************Refractive***********************
			  else if (materials[bounceGeom->materialid].hasRefractive)
			  {
				  Fresnel fresnel;
				  glm::vec3 refractionDir;
				  float io = glm::dot(currentRay.direction, bounceNormal);
				
				  if (rnd.z < 0.9f)
				  {
					  if (io < 0)
					  {
						  refractionDir = calculateTransmissionDirection(bounceNormal, currentRay.direction, 1.0f, materials[bounceGeom->materialid].indexOfRefraction);
					  }
					  else
					  {
						  refractionDir = calculateTransmissionDirection(-bounceNormal, currentRay.direction, materials[bounceGeom->materialid].indexOfRefraction, 1.0);
					  }
					  currentRay.origin = bouncePoint + 0.001f * refractionDir;
					  currentRay.direction = refractionDir;
					
					  continue;
				  }
				  else
				  {
					  glm::vec3 reflectionDir = calculateReflectionDirection(bounceNormal, currentRay.direction);
					  currentRay.direction = reflectionDir;
					  currentRay.origin = bouncePoint + 0.001f * reflectionDir;
					 
					  continue;
				  }
			  }
			  //**********************Diffuse*********************
			  else
			  {
				  currentRay.direction = nextRayDir;
				  currentRay.origin = bouncePoint + 0.001f * nextRayDir;
			  }
		
		  }
		  else
		  {
			  color *= 0;
			  break;
		  } 
	  }

	  colors[index] = colors[index] / (time + 1)*time  + color / (time + 1);
	 //colors[index] = color;
   }
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
	clock_t  clockBegin, clockEnd;
	clockBegin = clock();
  int traceDepth = RAY_DEPTH; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = TILE_SIZE;
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

  //******************Motion blur*****************************
#ifdef MOTION_BLUR
  int motionIndex = 5;
  float t = (float)(iterations / 50.0f);
  float trans = sin(0.5*t);
  geoms[motionIndex].translations[0].x = trans;
  glm::mat4 transform = utilityCore::buildTransformationMatrix(geoms[motionIndex].translations[0], geoms[motionIndex].rotations[0], geoms[motionIndex].scales[0]);
  geoms[motionIndex].transforms[0] = utilityCore::glmMat4ToCudaMat4(transform);
  geoms[motionIndex].inverseTransforms[0] = utilityCore::glmMat4ToCudaMat4(glm::inverse(transform));
#endif

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

  clockEnd = clock();
  printf("one iteration completed in %d ms\n", clockEnd - clockBegin);
}

__global__ void raytraceRaySC(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
	staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, pixelRayUnit *rayPool)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int index = rayPool[idx].index;
	glm::vec3 color(1, 1, 1);
	if (rayPool[idx].currentDepth > 0)
	{
		color = colors[index];
	}
	rayPool[idx].currentDepth++;

	if (rayPool[idx].currentDepth >= rayDepth)
	{
		color *= 0;
		rayPool[idx].isFinished = true;
		colors[index] = colors[index] / (time + 1)*time + color / (time + 1);
		return;
	}
#ifdef DOF
	float focalLength = DOF;
	glm::vec3 aimingPosition = rayPool[idx].pixelRay.origin + focalLength *  rayPool[idx].pixelRay.direction;
	glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time * 3, rayPool[idx].x, rayPool[idx].y);
	glm::vec3 camPosition = glm::vec3(cam.position.x + (float)rand.x, cam.position.y + (float)rand.y, cam.position.z + (float)rand.z);
	rayPool[idx].pixelRay.origin = camPosition;
	rayPool[idx].pixelRay.direction = glm::normalize(aimingPosition - camPosition);
#endif
	if (rayPool[idx].currentDepth <= rayDepth && !rayPool[idx].isFinished)
	{
		staticGeom *bounceGeom = NULL;
		glm::vec3 bouncePoint, bounceNormal;
		intersectionTest(geoms, numberOfGeoms, rayPool[idx].pixelRay, bounceGeom, bouncePoint, bounceNormal);
		glm::vec3 rnd = generateRandomNumberFromThread(cam.resolution, time + (rayPool[idx].currentDepth + 1), rayPool[idx].x, rayPool[idx].y);
		
		if (bounceGeom != NULL)
		{
			
			//calculate diffuse color
			if (materials[bounceGeom->materialid].emittance > 0)	//light
			{
				color *= materials[bounceGeom->materialid].emittance * materials[bounceGeom->materialid].color;
				rayPool[idx].isFinished = true;
				colors[index] = colors[index] / (time + 1)*time + color / (time + 1);
				return;
			}
		

			glm::vec3 nextRayDir = calculateRandomDirectionInHemisphere(bounceNormal, rnd.x, rnd.y);

			glm::vec3 H = glm::normalize(nextRayDir - rayPool[idx].pixelRay.direction);
			float NdotH = glm::dot(H, bounceNormal);
			if (materials[bounceGeom->materialid].hasRefractive <= 0)
			{
				color *= materials[bounceGeom->materialid].color;
			}
			//*************has reflection********************
			if (materials[bounceGeom->materialid].hasReflective && rnd.z < materials[bounceGeom->materialid].hasReflective)
			{
				
				glm::vec3 reflectionDir = calculateReflectionDirection(bounceNormal, rayPool[idx].pixelRay.direction);
				float io = glm::dot(rayPool[idx].pixelRay.direction, bounceNormal);
				rayPool[idx].pixelRay.origin = bouncePoint + 0.001f * reflectionDir;
				rayPool[idx].pixelRay.direction = reflectionDir;

			}
			//********************Refractive***********************
			else if (materials[bounceGeom->materialid].hasRefractive)
			{
				float io = glm::dot(rayPool[idx].pixelRay.direction, bounceNormal);
				glm::vec3 refractionDir;
			
				if (rnd.z < 0.9f)
				{
					if (io < 0)
					{
						refractionDir = calculateTransmissionDirection(bounceNormal, rayPool[idx].pixelRay.direction, 1.0f, materials[bounceGeom->materialid].indexOfRefraction);
					}
					else
					{
						refractionDir = calculateTransmissionDirection(-bounceNormal, rayPool[idx].pixelRay.direction, materials[bounceGeom->materialid].indexOfRefraction, 1.0);
					}
					rayPool[idx].pixelRay.origin = bouncePoint + 0.001f * refractionDir;
					rayPool[idx].pixelRay.direction = refractionDir;
				
					
				}
				else
				{
					glm::vec3 reflectionDir = calculateReflectionDirection(bounceNormal, rayPool[idx].pixelRay.direction);
					rayPool[idx].pixelRay.direction = reflectionDir;
					rayPool[idx].pixelRay.origin = bouncePoint + 0.001f * reflectionDir;
					
					
				}
			}
			//**********************Diffuse*********************
			else
			{
				rayPool[idx].pixelRay.direction = nextRayDir;
				rayPool[idx].pixelRay.origin = bouncePoint + 0.001f * nextRayDir;
			}

		}
		else
		{
			color *= 0;
			rayPool[idx].isFinished = true;
			colors[index] = colors[index] / (time + 1)*time + color / (time + 1);
			return;
		}
	}
	colors[index] = colors[index] / (time + 1)*time + color / (time + 1);
}
__global__ void initRayPool( glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
	staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, pixelRayUnit *rayPool)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if (x < resolution.x && y < resolution.y)
	{
		rayPool[index].pixelRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		rayPool[index].isFinished = false;
		rayPool[index].index = index;
		rayPool[index].x = x;
		rayPool[index].y = y;
		rayPool[index].currentDepth = 0;
	}
}
struct is_finished
{
	__host__ __device__
	bool operator()(const pixelRayUnit x)
	{
		return x.isFinished;
	}
};
void cudaRaytraceCoreSC(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
	clock_t  clockBegin, clockEnd;
	clockBegin = clock();
	int traceDepth = RAY_DEPTH; //determines how many bounces the raytracer traces
	// set up crucial magic
	int tileSize = TILE_SIZE;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x) / float(tileSize)), (int)ceil(float(renderCam->resolution.y) / float(tileSize)));

	// send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy(cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	// package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for (int i = 0; i<numberOfGeoms; i++){
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
	cudaMemcpy(cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
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
	//stream compaction
	pixelRayUnit *rayPool = NULL;
	int rayPoolSize = cam.resolution.x * cam.resolution.y;
	cudaMalloc((void**)&rayPool, (int)renderCam->resolution.x*(int)renderCam->resolution.y * sizeof(pixelRayUnit));
	initRayPool << <fullBlocksPerGrid, threadsPerBlock >> >(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, dev_material, numberOfMaterials,rayPool);
	int count = 0;
	tileSize = 16;

	while (rayPoolSize > 0 && count < traceDepth)
	{
		fullBlocksPerGrid = (int)ceil(float(rayPoolSize) / float(tileSize));

		raytraceRaySC << <fullBlocksPerGrid, tileSize >> >(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, dev_material, numberOfMaterials, rayPool);

		count++;
		thrust::device_ptr<pixelRayUnit> iteratorStart(rayPool);
		thrust::device_ptr<pixelRayUnit> iteratorEnd = iteratorStart + rayPoolSize;
		iteratorEnd = thrust::remove_if(iteratorStart, iteratorEnd, is_finished());
		rayPoolSize = (int)(iteratorEnd - iteratorStart);
	}
	tileSize = TILE_SIZE;
	threadsPerBlock = dim3(tileSize, tileSize);
	fullBlocksPerGrid = dim3((int)ceil(float(renderCam->resolution.x) / float(tileSize)), (int)ceil(float(renderCam->resolution.y) / float(tileSize)));

	sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> >(PBOpos, renderCam->resolution, cudaimage);

	// retrieve image from GPU
	cudaMemcpy(renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree(cudaimage);
	cudaFree(cudageoms);
	delete[] geomList;
	delete[] materialList;
	cudaFree(dev_material);
	cudaFree(rayPool);
	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");

	clockEnd = clock();
	printf("one iteration completed in %d ms\n", clockEnd - clockBegin);
}