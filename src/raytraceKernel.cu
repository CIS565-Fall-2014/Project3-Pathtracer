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

#define TRACE_DEPTH 10

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

// Function that does the initial raycast from the camera
//ASSUMING VIEW AND UP vector are all normalized AND FOV ARE IN RADIAN
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	ray r;
	r.origin = eye;

	float halfResX = (float)resolution.x / 2.0f;
	float halfResY = (float)resolution.y / 2.0f;

	glm::vec3 Pcenter = eye + view;

	glm::vec3 right = glm::cross(view,up);

	glm::vec3 Vy = tan(fov.y) * up;
	glm::vec3 Vx = tan(fov.x) * right;

	glm::vec2 normalizedPos = glm::vec2((x-halfResX)/halfResX,(halfResY - y)/halfResY);
	glm::vec3 posOnImagePlane = Pcenter + normalizedPos.y * Vy + normalizedPos.x * Vx;

	r.direction = glm::normalize(posOnImagePlane - eye);

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

		float randSeed = (iterations + 1.0f) * ((float) index + 1.0f) * (float) (depth + 1.0f);

		//gather hit info
		glm::vec3 intersectionPoint;
		glm::vec3 normal;
		int hitMatID;
		float hitDistance;
		hitDistance = intersectionTest(geoms,numOfGeoms, raypool[index], intersectionPoint, normal, hitMatID);

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
			calculateBSDF(raypool[index],randSeed, intersectionPoint, normal, hitMaterial);
		}

	}
}

//generate rays from camera
__global__ void generateInitialCamRays(ray * pool,glm::vec2 resolution, float iter, cameraData cam)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	ray R = raycastFromCameraKernel(resolution, iter, x, y,cam.position, cam.view, cam.up, cam.fov * ((float)PI / 180.0f));
	R.color = glm::vec3(1.0f);
	R.isActive = true;
	R.pixelIndex = x + (resolution.x * y);
	pool[index] = R;
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos,  glm::vec3 * cudaimage,camera* renderCam, ray * rayPool, int poolSize, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
	
	//MEMORY MANAGEMENT////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	
	//kernel config 
	int tileSize = 32;
	int blockSize = 128;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
	
	//flood raypool with init cam rays
	generateInitialCamRays<<<fullBlocksPerGrid,threadsPerBlock>>>(rayPool,renderCam->resolution, float(iterations), cam);

	//trace rays
	for(int i = 0;i < TRACE_DEPTH; i++)
	{
		pathtraceRays<<<ceil((float)poolSize/(float)blockSize),blockSize>>>(rayPool,cudaimage,poolSize, iterations, i, cudageoms, numberOfGeoms,cudamaterials, numberOfMaterials);
	}



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


// TODO: IMPLEMENT THIS FUNCTION
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){

	  ray R =  raycastFromCameraKernel(resolution, time, x, y,cam.position, cam.view, cam.up, cam.fov * ((float)PI / 180.0f));
	  
	  //hit info
	  float hitDist(FAR_CLIPPING_DISTANCE);
	  glm::vec3 hitPos, hitNorm;
	  int hitMatID;
	
	  //loop through all geometries
	  for(int i=0;i<numberOfGeoms;i++)
	  {
		  glm::vec3 interPt(0.0f);
		  glm::vec3 interNorm(0.0f);
		  float d(0.0f);
		  if(geoms[i].type == SPHERE) d = sphereIntersectionTest(geoms[i], R,interPt, interNorm);
		  else if(geoms[i].type == CUBE) d = boxIntersectionTest(geoms[i], R,interPt, interNorm);
		  //when hitting a surface that's closer than previous hit
		  if(d > 0.0f && d < hitDist)
		  {
			  hitDist = d;
			  hitPos = interPt;
			  hitNorm = interNorm;
			  hitMatID = geoms[i].materialid;
		  }
	  }
	  colors[index] = (hitDist < 0.0f || hitDist >= FAR_CLIPPING_DISTANCE) ? glm::vec3(0.0f,0.0f,1.0f) : glm::vec3(0.8f*glm::dot(hitNorm,-R.direction));
	  //colors[index] = (hitDist >= FAR_CLIPPING_DISTANCE) ? glm::vec3(0.0f) : hitNorm;
	  //colors[index] = (hitDist >= FAR_CLIPPING_DISTANCE) ? glm::vec3(0.0f) : glm::vec3(1.0f,0.0f,0.0f);
	  //colors[index] = R.direction;
   }
}
