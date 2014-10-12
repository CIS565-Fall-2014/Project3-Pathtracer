// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>

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
// IDID: Implemented camera raycast, random scatter included.
// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 planeRight, glm::vec3 planeDown){
  ray r;
  r.origin = glm::vec3(eye);

  glm::vec3 dir = view + (2 * x / resolution.x - 1) * planeRight + (2 * y / resolution.y - 1) * planeDown;

  glm::vec3 rand = generateRandomNumberFromThread(resolution, time, x, y) * (glm::length(planeRight) * 2 / resolution.x);
  r.direction = glm::normalize(dir + rand);

  return r;
}

// TODO: IMPLEMENT THIS FUNCTION
// IDID: Implemented camera raycast, random scatter included.
// Function that fills the initial raycast from camera, for stream compation optimization
__global__ void initRaycastFromCamera(glm::vec2 resolution, float time, glm::vec3 eye, glm::vec3 view, glm::vec3 planeRight, glm::vec3 planeDown, ray* rays){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if(x<=resolution.x && y<=resolution.y){
		int index = x + (y * resolution.x);
		ray r;
		r.origin = glm::vec3(eye);

		glm::vec3 dir = view + (2 * x / resolution.x - 1) * planeRight + (2 * y / resolution.y - 1) * planeDown;

		glm::vec3 rand = generateRandomNumberFromThread(resolution, time, x, y) * (glm::length(planeRight) * 2 / resolution.x);
		r.direction = glm::normalize(dir + rand);
		r.index = index;
		r.ended = false;
		r.init = false;
		r.color = glm::vec3(1.0f);

		rays[index] = r;
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

//Kernel that blacks out a given color buffer
__global__ void clearColors(glm::vec2 resolution, glm::vec3* colors){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      colors[index] = glm::vec3(0,0,0);
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

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials){
	
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int index = x + (y * resolution.x);
	if((x<=resolution.x && y<=resolution.y)){
		glm::vec3 newCol;
		ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.planeRight, cam.planeDown);
		for (int i = 0; i < rayDepth; i++) {
			glm::vec3 point, normal;
			float minT = FLT_MAX;
			float t;
			int minIndex = -1;
			for (int j = 0; j < numberOfGeoms; j++) {
				if (geoms[j].type == SPHERE) {
					t = sphereIntersectionTest(geoms[j], r, point, normal);
				} else if (geoms[j].type == CUBE) {
					t = boxIntersectionTest(geoms[j], r, point, normal);
				}
				if (t < minT && t != -1) {
					minT = t;
					minIndex = j;
				}
			}
			if (minIndex != -1) {
				// Simple normal debugging:
				//colors[index] += normal;
				// Flat colors
				newCol += materials[geoms[minIndex].materialid].color;
				glm::vec3 rand = generateRandomNumberFromThread(resolution, time, x, y);
				// Include a small epsilon * normal term in the new origin to prevent self-intersection.
				r.origin += r.direction * (float)minIndex + (float)EPSILON * normal;
				r.direction = calculateRandomDirectionInHemisphere(normal, rand.x, rand.y);
			}
			else {
				break;
			}
		}
		colors[index] = (colors[index] * time + newCol) / (time + 1);
	}
}

// IDID: added this function
// Core raytracer kernel for stream compation optimization
__global__ void raytraceRays(glm::vec2 resolution, float time, int traceDepth, ray* rays, int numberOfRays,
                            staticGeom* geoms, int numberOfGeoms, material* materials){
	
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	ray r = rays[index];
	if(index < numberOfRays && !r.ended){
		glm::vec3 point, normal;
		float minT = FLT_MAX;
		float t;
		int minIndex = -1;
		for (int j = 0; j < numberOfGeoms; j++) {
			if (geoms[j].type == SPHERE) {
				t = sphereIntersectionTest(geoms[j], r, point, normal);
			} else if (geoms[j].type == CUBE) {
				t = boxIntersectionTest(geoms[j], r, point, normal);
			}
			if (t < minT && t != -1) {
				minT = t;
				minIndex = j;
			}
		}
		if (minIndex != -1) {
			r.init = true;
			// Flat colors
			material mat =  materials[geoms[minIndex].materialid];
			if (mat.emittance > ZERO_ABSORPTION_EPSILON) {
				r.color = r.color * mat.color * mat.emittance;
				r.ended = true;
			} else {
				calculateBSDF(r, point, normal, mat, index*time);
				r.color = r.color * mat.color;
				/*// Set up new R
				thrust::default_random_engine rng(hash());
				thrust::uniform_real_distribution<float> u01(0,1);

				// Include a small epsilon * normal term in the new origin to prevent self-intersection.
				r.origin += r.direction * (float)minIndex + (float)EPSILON * normal;
				r.direction = calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng));*/
			}
			// Simple normal debugging:
			//r.color = normal;
		}
		else {
			if (!r.init) {
				r.color = glm::vec3(0.f);
			}
			r.ended = true;
		}
		if (traceDepth == 0) {
			r.ended = true;
			r.color = glm::vec3(0.f);
		}
		rays[index] = r;
	}
}

// Accumulates the colors of ended rays on this iteration
__global__ void raysToColors(glm::vec2 resolution, ray* rays, int numberOfRays, glm::vec3* colors) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int index = x + (y * resolution.x);
	// TODO: change this to a single-dimensional grid?
	if((x<=resolution.x && y<=resolution.y) && index < numberOfRays){
		ray r = rays[index];
		if (r.ended) {
			colors[r.index] = r.color;
			if(r.color == glm::vec3(0.f)) {
				r.color = glm::vec3(1.f);
			}
		}
	}
}

// Adds the colors accumulated from the current raytrace iteration to the image
__global__ void colorsToImage(glm::vec2 resolution, float time, glm::vec3* colors, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int index = x + (y * resolution.x);
	if((x<=resolution.x && y<=resolution.y)){
		image[index] = (image[index] * time + colors[index]) / (time + 1);
	}
}

// Predicate that marks a ray as "ended" if its 'ended' value is true.
struct isRayEnded {
	__host__ __device__
	bool operator() (const ray r) {
		return r.ended;
	}
};

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	int traceDepth = 4; //determines how many bounces the raytracer traces

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
	// calculate the vectors for screen coordinates here first.
	glm::vec3 right = glm::normalize(glm::cross(renderCam->views[frame], glm::normalize(renderCam->ups[frame])));
	glm::vec3 up = glm::cross(right, renderCam->views[frame]);
	// Image plane should have (0,0) at upper-left, (res.x, rex.y) at bottom-right.
	// TODO: Figure out why this is flipped horizontally
	cam.planeRight = renderCam->planeRight;
	cam.planeDown = renderCam->planeDown;

	// package materials
	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy(cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	// package lights --CURRENTLY NOT USED--
	int numberOfLights = 0;
	int * lightIds = new int[numberOfGeoms];
	for (int i = 0; i < numberOfGeoms; i++) {
		if (materials[geoms[i].materialid].emittance > ZERO_ABSORPTION_EPSILON) {
			lightIds[numberOfLights] = i;
			numberOfLights++;
		}
	}
	int* cudalightids = NULL;
	cudaMalloc((void**)&cudalightids, numberOfLights*sizeof(int));
	cudaMemcpy(cudalightids, lightIds, numberOfLights*sizeof(int), cudaMemcpyHostToDevice);
	delete [] lightIds;

	// raytrace core
	int numberOfRays = renderCam->resolution.x * renderCam->resolution.y;
	ray* cudarays = NULL;
	cudaMalloc((void**)&cudarays, numberOfRays*sizeof(ray));
	initRaycastFromCamera<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam.position, cam.view, cam.planeRight, cam.planeDown, cudarays);

	// compile the colors from this iteration
	glm::vec3* cudacolors = NULL;
	cudaMalloc((void**)&cudacolors, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));

	clearColors<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudacolors);
	while (traceDepth > 0 && numberOfRays > 0) {
		traceDepth--;
		raytraceRays<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, traceDepth, cudarays, numberOfRays, cudageoms, numberOfGeoms, cudamaterials);
		raysToColors<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudarays, numberOfRays, cudacolors);
		// Using thrust stream compaction
		thrust::device_ptr<ray> cudaraysStart(cudarays);

		float numRemoved = thrust::count_if(cudaraysStart, cudaraysStart+numberOfRays, isRayEnded());
		thrust::remove_if(cudaraysStart, cudaraysStart+numberOfRays, isRayEnded());
		numberOfRays -= numRemoved;

	}
	colorsToImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudacolors, cudaimage);

	// kernel launches
	//raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudaMaterials);

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudarays );
	cudaFree( cudalightids );
	cudaFree( cudacolors );

	delete geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
