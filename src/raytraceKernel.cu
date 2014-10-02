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
	ray r;
	//r.origin = glm::vec3(0,0,0);
	//r.direction = glm::vec3(0,0,-1);

	view = glm::normalize(view);
	glm::vec3 vecA = glm::normalize(glm::cross(view, up));// center to right
	glm::vec3 vecB = glm::normalize(glm::cross(vecA, view));// center to up

	glm::vec3 vecV = vecB * glm::length(view) * (float)tan(fov.y/ 180.0f * PI) * (1.0f - (1.0f + y * 2.0f) /resolution.y);
	glm::vec3 vecH = vecA * glm::length(view) * (float)tan(fov.x/ 180.0f * PI) * (1.0f - (1.0f + x * 2.0f) /resolution.x);

	glm::vec3 rayDir = glm::normalize(view + vecV + vecH);

	r.origin = eye;
	r.direction = rayDir;


	return r;
}

__host__ __device__ glm::vec3 getSpecularColor(ray* light, int lightCount, glm::vec3* lightColor, ray r, glm::vec3 faceNormal, float specularExp){
	//float specularColorR = 0;
	//float specularColorG = 0;
	//float specularColorB = 0;
	glm::vec3 specularColor(0,0,0);

	for(int i = 0; i < lightCount ; i++)
	{
		glm::vec3 reflectLight = -1.0f * glm::normalize( light[i].direction - faceNormal * 2.0f * glm::dot(light[i].direction, faceNormal));
		//reflectLight = -1.0f * reflectLight / sqrt(dot(reflectLight, reflectLight));

		//float alpha = acos(glm::dot(r.direction, reflectLight));
		//float specularTerm = pow(cos(alpha), specularExp);
		float specularTerm = 0.0f;
		float dotProduct = glm::dot(r.direction, reflectLight);
		if(dotProduct < 0)
			specularTerm = 0;
		else{
			specularTerm = pow(glm::dot(r.direction, reflectLight), specularExp);
		}
		//if(specularTerm < 0.0)
		//	specularTerm = 0.0f;
		//else if(specularTerm > 1.0)
		//	specularTerm = 1.0f;

		//specularColorR += specularTerm * lightColor[i].x;
		//specularColorG += specularTerm * lightColor[i].y;
		//specularColorB += specularTerm * lightColor[i].z;

		specularColor += specularTerm * lightColor[i];
	}
	
	//return glm::vec3(specularColorR, specularColorG, specularColorB);
	return specularColor;
}

//__host__ __device__ glm::vec3 getDiffuseColor(ray* light, int lightCount, glm::vec3* lightColor, ray r, glm::vec3 faceNormal){
//	return glm::vec3(0,0,0);
//}

__host__ __device__ glm::vec3 raytraceRecursive(ray r, int depth, /*glm::vec3* lightPos, int lightCount,*/ material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms){

	if(depth <= 0)
		return glm::vec3(0,0,0);

	bool hitCheck = false;
	float shortestDis = -1;
	int hitObjectIndex = -1;
	glm::vec3 intersectionPoint(0,0,0);
	glm::vec3 intersectionNormal(0,0,0);

	for(int i = 0; i < numberOfGeoms; ++i){
		float dis = -1;
		glm::vec3 objIntersectPt(0, 0, 0);
		glm::vec3 objIntersectN(0, 0, 0);
		switch(geoms[i].type){
			case SPHERE:
				dis = sphereIntersectionTest(geoms[i], r, objIntersectPt, objIntersectN);
				break;
			case CUBE:
				dis = boxIntersectionTest(geoms[i], r, objIntersectPt, objIntersectN);
				break;
			case MESH:
				break;
		}

		if((dis != -1 && shortestDis == -1) || (dis != -1 && shortestDis != -1 && dis < shortestDis && dis > 0)){
			hitCheck = true;
			shortestDis = dis;
			intersectionPoint = objIntersectPt;
			intersectionNormal = objIntersectN;
			hitObjectIndex = i;


		}
	}
	if(hitCheck == false){


		return glm::vec3(0,0,0);
	}
	else{
		material mate = materials[hitObjectIndex];

		
		if(mate.emittance != 0){ //hit light, so terminate the ray
			return mate.color * mate.emittance / 5.0f;
		}

		ray newRay;
		newRay.origin = glm::vec3(0,0,0);
		newRay.direction = glm::vec3(0,0,0);



		//glm::vec3 newEyePositionOut = intersectionPoint - r.direction * (float)EPSILON;//給一個epsloon避免ray打進去face裡面
		//glm::vec3 newEyePositionIn = intersectionPoint + r.direction * (float)EPSILON;//給一個epsloon讓ray打進去face裡面

		//glm::vec3* light2HitPtArray = new glm::vec3[lightCount];
		//float* disLight2HitPtArray = new float[lightCount];

		//for(int i = 0 ; i < lightCount ; i++){
		//	light2HitPtArray[i] = glm::normalize(newEyePositionOut - lightPos[i]);
		//	disLight2HitPtArray[i] = glm::length(light2HitPtArray[i]);
		//}


		//Reflect Color
		glm::vec3 reflectColor;
		if(mate.hasReflective != 0)//TODO  Check if really use this attribute
			reflectColor = raytraceRecursive(newRay, --depth, /*lightPos, lightCount,*/ materials, numberOfMaterials, geoms, numberOfGeoms);
		else
			reflectColor = glm::vec3(0, 0, 0);

		//Refract Color


		//Diffuse Color


		//Specular Color

		//getSpecularColor(ray* light, int lightCount, glm::vec3* lightColor, ray r, glm::vec3 faceNormal, float specularExp);

		glm::vec3 currentPtColor = reflectColor + mate.color;
		return currentPtColor;
	}


	return glm::vec3(0,0,0);
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

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms){
	
	//int lightCount = 1;
	//glm::vec3* lightPosArray = new glm::vec3[lightCount];
	//lightPosArray[0] = glm::vec3(0,0,0);

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	glm::vec3 pixelColor(0, 0, 0);
	if((x < resolution.x && y < resolution.y )){
		ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		
			
		glm::vec3 newColorEnergy =raytraceRecursive(r, rayDepth, /*lightPosArray, lightCount,*/ materials, numberOfMaterials, geoms, numberOfGeoms);
		glm::vec3 oldColorEnergy = colors[index] * (time - 1);
		glm::vec3 newColor = (newColorEnergy + oldColorEnergy) / time;
		colors[index] = newColor;

		 //glm::vec3 colorReflect = glm::vec3(0,0,0);;// = raytraceRay(resolution, time, ;

		//colors[index] = colorBRDF + colorReflect;

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
  
  material* cudaMaterials = NULL;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);




  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudaMaterials, numberOfMaterials, cudageoms, numberOfGeoms);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
