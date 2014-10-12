// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

// SEE HANDWRITTEN NOTES TO UNDERSTAND WHAT TO DO!

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"




#define TRACE_DEPTH_LIMIT 6 // this means 5 bounces in my convoluted code




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

// HINT: look at the first homework from CIS560! Start at line 250 and go down through the for loop at 328: all
//       the mathematics I need is already there!
//       "time" seems to refer to iteration number. Probably useful for a depth-of-field effect or something, but I will ignore it for now.
//       These rays could easily be "saved" to provide further optimization (so they aren't recalculated with each iteration), but in case
//       a DOF effect is implemented later on and "jittering" is required, I'll overlook this potential optimization.
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){

	// the HORIZONAL direction of the viewing plane, calculated with the "up" vector
	glm::vec3 A = glm::cross(view, up);
	// the VERTICAL direction of the viewing plane
	glm::vec3 B = glm::cross(A, view);

	// central point on the image plane that vectors from the eye are being drawn towards
	glm::vec3 M = eye + view;

	float phi = fov.y;
	float theta = fov.x;

	//rescaled HORIZONTAL
	glm::vec3 H = A * glm::length(view) * tan(theta) / glm::length(A);
	//rescaled VERTICAL
	glm::vec3 V = B * glm::length(view) * tan(phi) / glm::length(B);

	float sx = (float) x / (float) (resolution.x - 1);
	float sy = (float) y / (float) (resolution.y - 1);

	glm::vec3 screenPoint = M + (2*sx-1) * H + (2*sy-1) * V;

	ray r;
	r.direction = glm::normalize(screenPoint - eye);
	r.origin = eye;
	r.active = true;
	r.sourceindex = x + (y * resolution.x);
	r.color = glm::vec3(1,1,1);
	//r.intensityMultiplier = 1.f;

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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, float time){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

	  // output needs to be normalized against number of iterations for EACH frame drawn to the screen
	  color.r /= time;
	  color.g /= time;
	  color.b /= time;

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

// NOTE: I believe I just need to ADD an argument or so for materials/lights (lights are just materials with emittance)

// NOTE: this kernel represents "tracing ONE bounce" 
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, ray* rays,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  // REMEMBER:
  /*
       - do a colors[] +=, not a colors[] = ... I need to ACCUMULATE colors and then divide (see PBO function above)
	   - use the pooled array map to ensure you're +='ing to the proper colors[] entry!
	   - ** ADD TO COLORS[] UPON DEACTIVATION
  */
  if((x<=resolution.x && y<=resolution.y)){

	  if (!rays[index].active) {
		  //the ray has been deactivated, so do nothing
		  // if/when I implement stream compaction, this will be taken care of elsewhere, so there should be no issue
		  return;
	  }

	  if (rayDepth == TRACE_DEPTH_LIMIT - 1) {
		  rays[index].active = false;
		  colors[index] += glm::vec3(0,0,0); // THIS OCCURS TWICE (this line does nothing... it was originally used for debug)
	  }// too many bounces causes deactivation

		//glm::vec3 colorOfMinDistance(0, 0, 0);
		float minDistance = -1.f;
		glm::vec3 minIsectPt(0,0,0);
		glm::vec3 minNormal(0,0,0);
		material minMaterial;

		glm::vec3 i(0,0,0);
		glm::vec3 n(0,0,0);

		for (int g = 0; g < numberOfGeoms; g++) {
			if (geoms[g].type == CUBE) {
				float test = boxIntersectionTest(geoms[g], rays[index], i, n);
				if (test > -.5) { // so, positive
					if (minDistance < 0 || test < minDistance) { // if unset or new min
						minDistance = test;
						////light debug
						//if (materials[geoms[g].materialid].emittance > 0) {
							// colorOfMinDistance = glm::vec3(1,1,0);
						//}
						//else {
							//colorOfMinDistance = materials[geoms[g].materialid].color;
						//}
						minIsectPt = i;
						minNormal = n;
						minMaterial = materials[geoms[g].materialid];
					}

						//if ray isect any cube, add something
					colors[index] += materials[geoms[g].materialid].color;
						rays[index].active = false;
						return;
						////

				}
			}
			else if (geoms[g].type == SPHERE) {
				float test = sphereIntersectionTest(geoms[g], rays[index], i, n);
				if (test > -.5) { // so, positive
					if (minDistance < 0 || test < minDistance) { // if unset or new min
						//colorOfMinDistance = materials[geoms[g].materialid].color;
						minDistance = test;
						minIsectPt = i;
						minNormal = n;
						minMaterial = materials[geoms[g].materialid];
					}
				}
			}
		}

		// nothing was hit
		if (minDistance == -1.f) {
			rays[index].active = false;
			colors[index] += glm::vec3(0,0,0); // THIS OCCURS TWICE (also does nothing)
			return;
		}

		if (calculateBSDF(rays[index], rayDepth, minMaterial, minIsectPt, minNormal, (int)time) == 2) {
			//light has been hit, so terminate the ray here and add its color
			rays[index].active = false;
			colors[index] += rays[index].color;
		}
		// if light wasn't hit, BSDF takes care of other stuff as needed

  }
}

__global__ void initiateCameraRays(cameraData cam, ray* rays, float time) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if((x<=cam.resolution.x && y<=cam.resolution.y)){
		rays[index] = raycastFromCameraKernel(glm::vec2(cam.resolution.x, cam.resolution.y), time, x, y, cam.position, cam.view, cam.up, cam.fov);
	}
}

// TODO: FINISH THIS FUNCTION ("Support passing materials and lights to CUDA")
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management

/*
	 - This function manages an array of "pooled" rays which provides stream compaction optimization.
*/
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 0; //determines how many bounces the raytracer traces

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
  
  // package materials/lights and send to CUDA
  // I assume "materials" already contains everything I need, and numberOfMaterials has what I need:

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // send rays array to GPU (very similar to cudaimage)
  ray* cudarays = NULL;
  cudaMalloc((void**)&cudarays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));
  // no memcpy required

  // kernel launches

  initiateCameraRays<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, cudarays, (float)iterations);

  while (traceDepth < TRACE_DEPTH_LIMIT) {
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth,
														cudaimage, cudarays, cudageoms, numberOfGeoms, cudamaterials,
														numberOfMaterials);
	traceDepth++;
  }

  //std::cout << "\nraytraceRay call is done\n" << std::endl;

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, (float)iterations);

  //std::cout << "\nKernel calls are done\n" << std::endl;

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;
  cudaFree( cudamaterials );
  cudaFree( cudarays );

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
