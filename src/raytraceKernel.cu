// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

//Render Settings
#define TRACE_DEPTH 5
#define RAY_STREAM_COMPACTION_ON 0
#define ENABLE_ANTIALIASING 1

//report kernel failure
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//predicate function for thrust copy_if
struct rayIsActive
{
	__host__ __device__ bool operator()(const ray r)
	{
		return r.isActive;
	}
};

//compact rays, remove inactive rays
int streamCompactRays(ray * in, ray * out, int N)
{
	thrust::device_ptr<ray> input(in);
	thrust::device_ptr<ray> output(out);
	int ret = thrust::count_if(input,input + N,rayIsActive());
	thrust::copy_if(input,input+N,output,rayIsActive());
	return ret;
}

// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

// Function that does the initial raycast from the camera with DOF and AA capability
//ASSUMING VIEW AND UP vector are all normalized AND FOV ARE IN RADIAN
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, float focalLen, float aperture){

	float xx = x;
	float yy = y;

	ray r;
	r.origin = eye;

	float halfResX = (float)resolution.x / 2.0f;
	float halfResY = (float)resolution.y / 2.0f;

	glm::vec3 Pcenter = eye + view;
	glm::vec3 right = glm::cross(view,up);

	glm::vec3 Vy = tan(fov.y) * up;
	glm::vec3 Vx = tan(fov.x) * right;
	//if non-pin hole camera
	if(aperture > EPSILON)
	{
		thrust::default_random_engine rng(hash(time+1.0f));
		thrust::uniform_real_distribution<float> un11(-1,1);
		r.origin += un11(rng) * aperture * up;
		r.origin += un11(rng) * aperture * right;

		Pcenter = eye + focalLen * view;
		Vy = tan(fov.y) * up * focalLen;
		Vx = tan(fov.x) * right * focalLen;
	}

	//if aa, jitter pixel position
#if(ENABLE_ANTIALIASING)
	{
		thrust::default_random_engine rng(hash((time+1.0f)* xx * yy));
		thrust::uniform_real_distribution<float> u01(0.0f,1.0f);
		xx += u01(rng);
		yy += u01(rng);
	}
#endif


	glm::vec2 normalizedPos = glm::vec2((xx -halfResX)/halfResX,(halfResY - yy)/halfResY);
	glm::vec3 posOnImagePlane = Pcenter + normalizedPos.y * Vy + normalizedPos.x * Vx;

	r.direction = glm::normalize(posOnImagePlane - r.origin);

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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
	  color.x = image[index].x*255.0/iterations;
      color.y = image[index].y*255.0/iterations;
      color.z = image[index].z*255.0/iterations;

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

__global__ void pathtraceRays(ray * raypool,glm::vec3* colors, int N, float iterations, int depth, staticGeom* geoms,  int numOfGeoms, material * materials, int numOfMats)
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < N)
	{
		if(!raypool[index].isActive) return;  //isActive can be removed when stream compaction is done

		float randSeed = ((float)iterations/10.0f + 1.0f) * ((float) index + 1.0f) * ((float)depth + 1.0f);

		//gather hit info
		glm::vec3 intersectionPoint;
		glm::vec3 normal;
		int hitMatID,hitObjID;
		float hitDistance;
		hitDistance = intersectionTest(geoms,numOfGeoms, raypool[index], intersectionPoint, normal, hitMatID,hitObjID);

		//if hit nothing
		if(hitDistance < 0.0f || hitDistance >= FAR_CLIPPING_DISTANCE) 
		{
			raypool[index].isActive = false;
			return;
		}
		
		material hitMaterial = materials[hitMatID];

		//if hit light
		if(hitMaterial.emittance > EPSILON)
		{
			//colors[raypool[index].pixelIndex]  = colors[raypool[index].pixelIndex] + hitMaterial.emittance * hitMaterial.color*raypool[index].color /((float)iterations);
			colors[raypool[index].pixelIndex]  += hitMaterial.emittance * hitMaterial.color*raypool[index].color;
			raypool[index].isActive = false;
			return;
		}
		
		else
		{
			//BSDF handles ray interaction with surface
			calculateBSDF(raypool[index],geoms,randSeed, hitObjID, intersectionPoint, normal, hitMaterial);
		}

	}
}

//generate rays from camera
__global__ void generateInitialCamRays(ray * pool,glm::vec2 resolution, float iter, cameraData cam)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	ray R = raycastFromCameraKernel(resolution, iter, x, y,cam.position, cam.view, cam.up, cam.fov * ((float)PI / 180.0f), cam.focalLen, cam.aperture);
	R.color = glm::vec3(1.0f);
	R.isActive = true;
	R.pixelIndex = x + (resolution.x * y);
	pool[index] = R;
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos,  glm::vec3 * cudaimage,camera* renderCam, ray * rayPoolA, ray * rayPoolB,int poolSize, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
	
	//MEMORY MANAGEMENT////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.m_triangle = geoms[i].m_triangle;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[i] = newStaticGeom;
	}

	//send materials to device
	material * cudamaterials;
	cudaMalloc((void**) & cudamaterials, numberOfMaterials*sizeof(material));
 	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

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
	cam.focalLen = renderCam->focalLen;
	cam.aperture = renderCam->aperture;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	
	//kernel config 
	int tileSize = 32;
	int blockSize = 128;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
	
	//flood raypool with init cam rays
	generateInitialCamRays<<<fullBlocksPerGrid,threadsPerBlock>>>(rayPoolA,renderCam->resolution, float(iterations), cam);


	//trace rays
#if RAY_STREAM_COMPACTION_ON 
	int rayCount = poolSize;
	for(int i = 0;i < TRACE_DEPTH; i++)
	{
		if(rayCount< 1) break;
		if(i % 2 == 0)
		{
			pathtraceRays<<<ceil((float)rayCount/(float)blockSize),blockSize>>>(rayPoolA,cudaimage,rayCount, iterations, i, cudageoms, numberOfGeoms,cudamaterials, numberOfMaterials);
			rayCount = streamCompactRays(rayPoolA,rayPoolB,rayCount);
		}
		else
		{
			pathtraceRays<<<ceil((float)rayCount/(float)blockSize),blockSize>>>(rayPoolB,cudaimage,rayCount, iterations, i, cudageoms, numberOfGeoms,cudamaterials, numberOfMaterials);
			rayCount = streamCompactRays(rayPoolB,rayPoolA,rayCount);
		}

	}


#else
	for(int i = 0;i < TRACE_DEPTH; i++)
	{
		pathtraceRays<<<ceil((float)poolSize/(float)blockSize),blockSize>>>(rayPoolA,cudaimage,poolSize, iterations, i, cudageoms, numberOfGeoms,cudamaterials, numberOfMaterials);
	}
#endif



	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,(float) iterations);




	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudageoms );
	cudaFree( cudamaterials);
	delete geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
 
}


