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
#include "../src/cuPrintf.cu"  

#define len(x) sqrtf(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])

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
glm::vec3 norm(glm::vec3 in)
{
	 glm::vec3 ret = in / (float)len(in);
	 return ret;
}
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){

  glm::vec3 A = glm::cross(view, up);
  glm::vec3 B = glm::cross(A,view);
  glm::vec3 M = view + eye;
  glm::vec3 V = B * (float)view.length() * tanf(float(fov.y * PI / 180.0)) / (float)B.length();
  glm::vec3 H = A * (float)view.length() * tanf(float(fov.x * PI / 180.0)) / (float)A.length(); 
  glm::vec3 P = M + (float)((2.0*x)/(resolution.x-1.0)-1.0) * H +  (float)(2.0*(resolution.y - y - 1.0)/(resolution.y-1.0)-1.0) * V;
   
  ray r;
  r.origin = P;
  r.direction = glm::normalize(P-eye);
 
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

__host__ __device__ int checkIntersections(ray r, staticGeom* geoms, int numberOfGeoms, glm::vec3 intersectionPoint, glm::vec3 normal)
{
	int closestGeo = -1;
	float t = 99999;

	for(int i = 0; i < numberOfGeoms; ++i)
	{
		float tmp;
		if(geoms[i].type == SPHERE)
			tmp = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == CUBE)
			tmp = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);

		if(tmp != -1 && tmp < t)
		{
			t = tmp;
			closestGeo = i;
		}
	}

	if( closestGeo != -1)
	{
		if(geoms[closestGeo].type == SPHERE)
			sphereIntersectionTest(geoms[closestGeo], r, intersectionPoint, normal);
		else if(geoms[closestGeo].type == CUBE)
			boxIntersectionTest(geoms[closestGeo], r, intersectionPoint, normal);
		return closestGeo;
	}
	else
		return -1;

}
__host__ __device__ void iterativeRayTrace(ray r, int rayDepth, float time, staticGeom* geoms, material* materials,
	                                       glm::vec3& color, int x, int y)
{
	if(rayDepth > 2)
		return;

	
}
__global__ void genCameraRayBatch(glm::vec2 resolution, cameraData cam,  ray * rays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y)
	{
		rays[index] = raycastFromCameraKernel(resolution, x, y, cam.position, cam.view, cam.up, cam.fov);

	}
}
// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material * cudaMat, ray * rays){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  ray r;
  r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
  //cuPrintf("Ray Postion: %f %f %f  Direction: %f %f %f\n", r.origin.x , r.origin.y,r.origin.z,r.direction.x,r.direction.y,r.direction.z );
  
  if((x<=resolution.x && y<=resolution.y))
  {
	 glm::vec3 intersectionPoint, normal;
	 int geoIndex = checkIntersections(r, geoms, numberOfGeoms, intersectionPoint, normal);
	 colors[index] = cudaMat[geoms[geoIndex].materialid].color;
	 if(geoIndex!=-1) // hit something, shoot ray again
	 {
	
	 }
   } 
    //colors[index] = generateRandomNumberFromThread(resolution, time, x, y); 
  
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
	int traceDepth = 2; //determines how many bounces the raytracer traces

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
  
	material* cudaMat = NULL;
	cudaMalloc((void**)&cudaMat, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudaMat, materials, numberOfGeoms*sizeof(material), cudaMemcpyHostToDevice);

	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	// package light


	// kernel launches
	//cudaPrintfInit();
	ray * rays;
	cudaMalloc((void**)&rays, cam.resolution.x * cam.resolution.y * sizeof(material));
	genCameraRayBatch(cam.resolution, cam,  rays);
	for( int i = 0; i < traceDepth; ++i)
		raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudaMat);
	//cudaPrintfDisplay(stdout, false);

	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	//cudaPrintfEnd();
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	delete geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
