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
	
	int index = x + (y * resolution.x);

	//A = view x up
	glm::vec3 A = glm::cross(view, up);
	//B = A x view
	glm::vec3 B = glm::cross(A, view);
	//M = eye + view
	glm::vec3 M = eye + view;

	float AspectRatio = float(resolution.x)/float(resolution.y);
	float mag_V = glm::length(view) * tan( float( fov.y * PI/(float)180) )/ glm::length(B);

	glm::vec3 V = B * mag_V;
	glm::vec3 H = A * mag_V * AspectRatio; 

	
	//choose point on the image plane based on pixel location
	float Sh = float(x)/float(resolution.x-1);
	float Sv = float(y)/float(resolution.y-1);

	//choose random point on image plane
/*	thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);
	float Sh = (float) u01(rng);
	float Sv = (float) u01(rng);*/

	//sreen coordinates to world coordinates
	glm::vec3 point = M + (2*Sh-1)*H + (2*Sv-1)*V;

	//ray tracing for each pixel
	ray r;
	r.direction = point - eye;
	r.origin = eye;
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

__device__ int findHitGeo(ray r, glm::vec3& Pintersect, glm::vec3& Pnormal, staticGeom* geoms, int numberOfGeoms){
	 float dist_min = -1, dist;
	 int ID = -1;
	//geometry and ray intersect tesing
	for (int g=0; g<numberOfGeoms; g++){
		if(geoms[g].type == SPHERE){
			dist = sphereIntersectionTest(geoms[g], r, Pintersect, Pnormal);
		}
		else if(geoms[g].type == CUBE){
			dist = boxIntersectionTest(geoms[g], r, Pintersect, Pnormal);
			
		}
		else if (geoms[g].type == MESH){

		}

		//overwrite minimum distance if needed
		if( (dist_min<0 && dist>0 ) || ( dist_min > 0 && dist<dist_min && dist>0 ) ){
			dist_min = dist;   //update minimum dist
			ID = g;   //update ID of geometry
		}
	}
	return ID;
}

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* cudamats){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  glm::vec3 Pintersect, Pnormal;
  int hitGeoID = -1;
  if( (x<=resolution.x && y<=resolution.y) ){
		//colors[index] = glm::vec3(0,0,0);
		// colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
		 
		//for(int k=0; k<rayDepth; k++){
			ray r = raycastFromCameraKernel(resolution,time, x, y, cam.position,cam.view, cam.up, cam.fov);
			//colors[index] = glm::vec3( std::abs( r.direction.x), std::abs( r.direction.y), std::abs( r.direction.z) );    //ray casting test
			hitGeoID = findHitGeo(r, Pintersect, Pnormal, geoms, numberOfGeoms);
			if(hitGeoID!=-1){
				
				colors[index] = cudamats[ geoms[hitGeoID].materialid ].color;
			}

			//find intersection point and normal
			if( geoms[hitGeoID].type == SPHERE){
				sphereIntersectionTest(geoms[hitGeoID], r, Pintersect, Pnormal);
			}
			else if(geoms[hitGeoID].type == CUBE){
				boxIntersectionTest(geoms[hitGeoID], r, Pintersect, Pnormal);

			}
			else if (geoms[hitGeoID].type == MESH){

			}
			//colors[index] = glm::vec3( Pnormal.x, Pnormal.y,Pnormal.z) ;
			//pick a random direction from hit point and keep going
			/*ray nr;
			nr.origin = Pintersect;
			nr.direction = calculateRandomDirectionInHemisphere( Pnormal, float xi1, float xi2);
			
			//compute BRDF for this ray
			}*/
		//}
   }
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

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

  // package material
  material* matList = new material[numberOfMaterials];
   for(int i=0; i<numberOfMaterials; i++){
	material curMat;
    curMat.absorptionCoefficient = materials[i].absorptionCoefficient;
	curMat.color = materials[i].color;
	curMat.emittance = materials[i].emittance;
	curMat.hasReflective = materials[i].hasReflective;
	curMat.hasRefractive = materials[i].hasRefractive;
	curMat.hasScatter = materials[i].hasScatter;
	curMat.indexOfRefraction = materials[i].indexOfRefraction;
	curMat.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	curMat.specularColor = materials[i].specularColor;
	curMat.specularExponent = materials[i].specularExponent;
    matList[i] = curMat;
  }
  material* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfMaterials*sizeof(material));
  checkCUDAError("Kernel failed!");
  cudaMemcpy( cudamats, matList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  checkCUDAError("Kernel failed!");

  // kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamats);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  delete geomList;
  delete matList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
