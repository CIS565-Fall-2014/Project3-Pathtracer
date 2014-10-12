// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath> 
#include <stdlib.h>
#include <windows.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
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
  glm::vec3 u = glm::cross(up,view);
  r.direction = glm::normalize(up * (tan(fov.y*3.14159f/180.0f) * (-y / resolution.y + 0.5f)*2.0f) + u * (tan(fov.x*3.14159f/180.0f) * (x / resolution.x - 0.5f)*2.0f) + view);
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(1,1,1);
    }
}

__global__ void clearRay(glm::vec2 resolution, ray* r){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
		r[index].direction = glm::vec3(0,0,0);
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

 __device__ glm::vec3 get_texture_color(bmp_texture* tex, glm::vec2 texcoord,glm::vec3* texData){
	 if(texcoord.x<0)
		 texcoord.x=0;
	 else if (texcoord.x>1)
		 texcoord.x=1;

	 if(texcoord.y<0)
		 texcoord.y=0;
	 else if (texcoord.y>1)
		 texcoord.y=1;

	 int i = (int)(texcoord.y*tex->height);
	 int j = (int)(texcoord.x*tex->width);
	 int index = j*tex->height + i;
	 if(index>tex->height*tex->width-1)
		 index = tex->height*tex->width-1;
	 else if (index<0)
		index = 0;
	 return texData[index];
 }

 __device__ glm::vec3 pathTrace(ray r,float time,int index, int depth, staticGeom* geoms, int numberOfGeoms, material* materials,ray &reflectRay,bool &hitLight,bool &hitAnything,bmp_texture* tex,glm::vec3* texData){
	glm::vec3 intersectionPoint;
	glm::vec3 normal;
	glm::vec3 intersectionPoint_tmp;
	glm::vec3 normal_tmp;
	glm::vec2 texCoord;
	int minIndex = -1;
	float mint = 99999999.0f;
	for(int i=0;i<numberOfGeoms;i++){
		if(geoms[i].type == SPHERE){
			float temp = sphereIntersectionTest(geoms[i],r,intersectionPoint_tmp,normal_tmp);
			if(temp>0&&temp<mint){
				minIndex = i;
				intersectionPoint = intersectionPoint_tmp;
				normal = normal_tmp;
				mint=temp;
			}
		}
		else if(geoms[i].type == CUBE){
			float temp = boxIntersectionTest(geoms[i],r,intersectionPoint_tmp,normal_tmp,texCoord);
			if(temp>0&&temp<mint){
				minIndex = i;
				intersectionPoint = intersectionPoint_tmp;
				normal = normal_tmp;
				mint=temp;
			}
		}
	}
	if(minIndex!=-1){

		material m = materials[geoms[minIndex].materialid];	
		if(m.emittance>0){
			hitLight = true;
			return m.color * m.emittance;
		}

		if(m.indexOfRefraction>0.1f){
			float ndotwo = glm::dot(normal , -r.direction);
			reflectRay.origin=intersectionPoint;
			reflectRay.direction = r.direction + 2.0f * normal * ndotwo; 
			return m.specularColor*m.color;
		}
		else{

			thrust::default_random_engine rng(hash(index*(time+depth+1)));
			thrust::uniform_real_distribution<float> xi1(0,1);
			thrust::uniform_real_distribution<float> xi2(0,1);	

			reflectRay.origin=intersectionPoint;
			reflectRay.direction = calculateRandomDirectionInHemisphere(normal,(float)xi1(rng),(float)xi2(rng));
			float diffuse = glm::dot(reflectRay.direction,normal);
			if(diffuse<0)
				diffuse = 0;

			if(m.texture[0]!='N')
				return get_texture_color(tex,texCoord,texData);
			return diffuse * m.color;
		}

	}
	hitAnything = false;
	return glm::vec3(0,0,0);
}
#define MaxDepth 5
// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, ray* rays, glm::vec3* tmpColors,bmp_texture* tex,glm::vec3* texData){

  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(glm::length(rays[index].direction)<0.01f && rayDepth>1)
	  return;

  if((x<=resolution.x && y<=resolution.y)){
		ray r;
		if(rayDepth == 1)
			r = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov);
		else
			r = rays[index];

		bool hitLight = false;
		bool hitAnything = true;
		ray reflectRay;
		reflectRay.direction = glm::vec3(1,1,1);
		reflectRay.origin = glm::vec3(0,0,0);
		
		tmpColors[index] *= pathTrace(r,time,index,rayDepth,geoms,numberOfGeoms,materials,reflectRay,hitLight,hitAnything,tex,texData);
		if(hitLight){
			colors[index] = (colors[index] * (time-1)) /(time) + tmpColors[index]/time;
			rays[index].direction = glm::vec3(0,0,0);
			return;
		}
		else if(rayDepth==MaxDepth)
			colors[index] = (colors[index] * (time-1)) /(time);
		if(hitAnything&&rayDepth<MaxDepth)
			rays[index] = reflectRay;
		else{
			colors[index] = (colors[index] * (time-1)) /(time);
			rays[index].direction = glm::vec3(0,0,0);
		}		
   }
}


// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,bmp_texture* tex){

  
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
  
  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  bmp_texture* cudatex = NULL;
  cudaMalloc((void**)&cudatex, sizeof(bmp_texture));
  cudaMemcpy( cudatex, tex, sizeof(bmp_texture), cudaMemcpyHostToDevice);
  glm::vec3 *data = NULL; 
  cudaMalloc((void**)&data, tex->height*tex->width*sizeof(glm::vec3));
  cudaMemcpy( data, tex->data, tex->height*tex->width*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // send image to GPU
  ray* cudaRay = NULL;
  cudaMalloc((void**)&cudaRay, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(ray));
  clearRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaRay);
  cudaThreadSynchronize();

  glm::vec3 *tmpColor = NULL; 
  cudaMalloc((void**)&tmpColor, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, tmpColor);
  cudaThreadSynchronize();


  // kernel launches
  for(;traceDepth<=MaxDepth;traceDepth++){
	 raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamaterials,cudaRay,tmpColor,cudatex,data);
	 cudaThreadSynchronize();
  }
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudaRay );
  cudaFree( tmpColor );
  cudaFree( cudamaterials );
  cudaFree( cudatex );
  cudaFree( data );

  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
