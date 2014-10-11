// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

#define ANTI_ALIAS 1
#define MAX_DEPTH 8

#define DOFLENGTH	8
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 
struct isDead{
	__host__ __device__ 
	bool operator()(const ray r)
	{
		return r.active == false;
	}
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
	r.active = true;

	int index = y * resolution.x + x;
    float phi = glm::radians(fov.y);
	float theta = glm::radians(fov.x);
	glm::vec3 A = glm::normalize(glm::cross(view, up));
	glm::vec3 B = glm::normalize(glm::cross(A, view));
	glm::vec3 M = eye + view;
	glm::vec3 V = B * glm::length(view) * tan(phi);
	glm::vec3 H = A * glm::length(view) * tan(theta);

	// super sampling for anti-aliasing
	thrust::default_random_engine rng(hash(time*index));
	thrust::uniform_real_distribution<float> u01(0, 1);
	float fx = x + (float)u01(rng);
	float fy = y + (float)u01(rng);

	glm::vec3 P = M + (2*fx/(resolution.x-1)-1) * H + (2*(1-fy/(resolution.y-1))-1) * V;
	r.direction = glm::normalize(P-eye);
	//depth of field
	//thrust::uniform_real_distribution<float> u02(-0.3,0.3);
	//glm::vec3 aimPoint = r.origin + (float)DOFLENGTH * r.direction;
	//r.origin += glm::vec3(u02(rng),u02(rng),u02(rng));
	//r.direction = aimPoint - r.origin;
	//r.direction = glm::normalize(r.direction);

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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float i){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0/i;
      color.y = image[index].y*255.0/i;
      color.z = image[index].z*255.0/i;

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

__global__ void generateRay(cameraData cam, float time, ray* raypool) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if((x<=cam.resolution.x && y<=cam.resolution.y)){
		raypool[index] = raycastFromCameraKernel(cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		raypool[index].index = index;
		raypool[index].color = glm::vec3(1, 1, 1);
	}
}
__host__ __device__ glm::vec3 getReflectedRay(glm::vec3 d, glm::vec3 n) {
	glm::vec3 VR; // reflected ray direction
	if (glm::length(-d - n) < EPSILON) {
		VR = n;
	}
	else if (abs(glm::dot(-d, n)) < EPSILON) {
		VR = d;
	}
	else {
		VR = glm::normalize(d - 2.0f * glm::dot(d, n) * n);
	}
	return VR;
}

// Get the refracted ray direction from ray direction, normal and index of refraction (IOR)
__host__ __device__ glm::vec3 getRefractedRay(glm::vec3 d, glm::vec3 n, float IOR) {
	glm::vec3 VT; // refracted ray direction
	float t = 1 / IOR;
	float base = 1 - t * t * (1 - pow(glm::dot(n, d), 2));
	if (base < 0) {
		 VT = glm::vec3(0, 0, 0);
	}
	else {
		VT = (-t * glm::dot(n, d) - sqrt(base)) * n + t * d; // refracted ray
		VT = glm::normalize(VT);
	}
	return VT;
}
__host__ __device__ bool notDiffuseRay(float randomSeed, float hasReflect) {
	// determine if ray is reflected according to the proportion
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) > hasReflect) {
		return true;
	}
	else {
		return false;
	}
}

// Determine if the randomly generated ray is a refracted ray or a reflected ray
__host__ __device__  bool isRefractedRay(float randomSeed, float IOR, glm::vec3 d, glm::vec3 n, glm::vec3 t) {
	float rpar = (IOR * glm::dot(n, d) - glm::dot(n, t)) / (IOR * glm::dot(n, d) + glm::dot(n, t));
	float rperp = (glm::dot(n, d) - IOR * glm::dot(n, t)) / (glm::dot(n, d) + IOR * glm::dot(n, t));

	// compute proportion of the light reflected
	float fr = 0.5 * (rpar * rpar + rperp * rperp);

	// determine if ray is reflected according to the proportion
	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) <= fr) {
		return false;
	}
	else {
		return true;
	}
}

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material *materials, ray* rays, int numberOfRays){

  /*int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);*/
  int rayIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(rayIdx < numberOfRays){
		int pixelIdx = rays[rayIdx].index;
		float seed = time * rayIdx * (rayDepth+1);
		if(rays[rayIdx].active){
			int matIdx = -1;
			glm::vec3 interPoint, normal;				
			IntersectionTest(geoms, rays[rayIdx], interPoint, normal, matIdx, numberOfGeoms);

			if(matIdx != -1){
				material mat1 = materials[matIdx];
				if(mat1.emittance > EPSILON){
					glm::vec3 color = rays[rayIdx].color * mat1.color * mat1.emittance;
					colors[pixelIdx] += color;
					rays[rayIdx].active = false;
					//colors[pixelIdx] = normal;
				}
				else{
					
					
					//if(mat1.hasReflective > EPSILON || mat1.hasRefractive > EPSILON){
					//	//if (notDiffuseRay(seed, mat1.hasReflective)) {
					//		float IOR = mat1.indexOfRefraction;//Index of Refraction
					//		if (glm::dot(rays[rayIdx].direction, normal) > 0) { // reverse normal and index of refraction if ray inside the object
					//			normal *= -1;
					//			IOR = 1/(IOR + EPSILON);
					//		}
					//		if (mat1.hasRefractive > EPSILON) { // if the surface has refraction
					//			glm::vec3 dir = getRefractedRay(rays[rayIdx].direction, normal, IOR);
					//			if (glm::length(dir) > EPSILON && (mat1.hasReflective < EPSILON|| isRefractedRay(seed, IOR, rays[rayIdx].direction, normal, dir))) {
					//				rays[rayIdx].direction = dir;
					//				rays[rayIdx].origin = interPoint + dir * (float)EPSILON;
					//				rays[rayIdx].color *= mat1.color;
					//				return;
					//			}
					//		}
					//		// if the surface only has reflection
					//		glm::vec3 dir2 = getReflectedRay(rays[rayIdx].direction, normal);
					//		rays[rayIdx].origin = interPoint + dir2 * (float)EPSILON;
					//		rays[rayIdx].direction = dir2;
					//		rays[rayIdx].color *= mat1.color;
					//		return;
					//	 //}
					//}
					if (glm::dot(rays[rayIdx].direction, normal) > 0) { // reverse normal if we are inside the object
						normal *= -1;
					}
					//diffuse
					thrust::default_random_engine rng(hash(seed));
					thrust::uniform_real_distribution<float> u01(0, 1);

					rays[rayIdx].direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng)));
					rays[rayIdx].origin = interPoint + rays[rayIdx].direction * (float)EPSILON;
					rays[rayIdx].color = rays[rayIdx].color * mat1.color;
					
				}
			}
			else{
				//rays[rayIdx].color = glm::vec3(0,0,0);
				rays[rayIdx].active = false;
			}
		}
		
   }
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
	
	
	int traceDepth; //determines how many bounces the raytracer traces

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
	int numberOfRays = (int)renderCam->resolution.x*(int)renderCam->resolution.y;
	ray *raypool1;
	cudaMalloc((void**)&raypool1, numberOfRays * sizeof(ray));
	
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
	//cache material
	material* cudamtls;
	cudaMalloc((void**)&cudamtls, numberOfMaterials*sizeof(material));
	cudaMemcpy(cudamtls, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	generateRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, (float)iterations, raypool1);
	// kernel launches
	int threadPerBlock = 64;//TODO tweak
    int blockPerGrid = (int)ceil((float)numberOfRays/threadPerBlock);
	for(traceDepth = 0; traceDepth < MAX_DEPTH; traceDepth++){
		raytraceRay<<<blockPerGrid, threadPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamtls, raypool1, numberOfRays);
		cudaDeviceSynchronize();
		thrust::device_ptr<ray> rayPoolStart = thrust::device_pointer_cast(raypool1);
	    thrust::device_ptr<ray> rayPoolEnd = thrust::remove_if(rayPoolStart,rayPoolStart+numberOfRays,isDead());
	    numberOfRays = (int)( rayPoolEnd - rayPoolStart);
		if(numberOfRays <= 0) break;
	}
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( raypool1 );
	cudaFree( cudamtls );
	delete[] geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
