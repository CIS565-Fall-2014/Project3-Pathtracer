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

  float a = (float) u01(rng);
  float b = (float) u01(rng);
  float c = (float) u01(rng);
  glm::vec3 returnValue(a,b,c);
  return returnValue;
}

// TODO: IMPLEMENT THIS FUNCTION
// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	ray r;

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

__host__ __device__ glm::vec3 getDirectRadiance(glm::vec3 light, glm::vec3 lightColor, ray r, glm::vec3 faceNormal, glm::vec3 materialColor, glm::vec3 materialSpecularColor, float specularExp){
	glm::vec3 directMaterialColor(0,0,0);

	glm::vec3 reflectLight = -1.0f * glm::normalize( light - faceNormal * 2.0f * glm::dot(light, faceNormal));

	float specularTerm = 0.0f;
	float dotProduct = glm::dot(r.direction, reflectLight);
	if(dotProduct < 0 || specularExp < 1)
		specularTerm = 0;
	else{
		specularTerm = pow(dotProduct, specularExp);
	}

	float diffuseTerm = glm::dot(-1.0f * light, faceNormal);
	if(diffuseTerm < 0)
		diffuseTerm = 0;

	directMaterialColor.x +=( specularTerm * materialSpecularColor.x + diffuseTerm * materialColor.x) * lightColor.x;
	directMaterialColor.y += (specularTerm * materialSpecularColor.y + diffuseTerm * materialColor.y) * lightColor.y;
	directMaterialColor.z += (specularTerm * materialSpecularColor.z + diffuseTerm * materialColor.z) * lightColor.z;


	
	return directMaterialColor;
}

__host__ __device__ glm::vec3 getSpecularColor(glm::vec3 light, glm::vec3 lightColor, ray r, glm::vec3 faceNormal, glm::vec3 materialSpecularColor, float specularExp){

	glm::vec3 specularColor(0,0,0);

	glm::vec3 reflectLight = -1.0f * glm::normalize( light - faceNormal * 2.0f * glm::dot(light, faceNormal));

	float specularTerm = 0.0f;
	float dotProduct = glm::dot(r.direction, reflectLight);
	if(dotProduct < 0)
		specularTerm = 0;
	else{
		specularTerm = pow(glm::dot(r.direction, reflectLight), specularExp);
	}


	specularColor.x += specularTerm * materialSpecularColor.x * lightColor.x;
	specularColor.y += specularTerm * materialSpecularColor.y * lightColor.y;
	specularColor.z += specularTerm * materialSpecularColor.z * lightColor.z;

	//specularColor += specularTerm * lightColor;
	
	return specularColor;
}

__host__ __device__ glm::vec3 getDiffuseColor(glm::vec3 light, glm::vec3 lightColor, glm::vec3 faceNormal, glm::vec3 materialColor){
	glm::vec3 diffuseColor(0,0,0);


	float testR = faceNormal.x;
	if(testR < 0)
		testR = 0;
	float testG = faceNormal.y;
	if(testG < 0)
		testG = 0;
	float testB = faceNormal.z;
	if(testB < 0)
		testB = 0;
	//diffuseColor = glm::vec3(testR, testG, testB);


	float newDiffuseTerm = glm::dot(-1.0f * light, faceNormal);
	if(newDiffuseTerm < 0)
		newDiffuseTerm = 0;


	diffuseColor.x += newDiffuseTerm * materialColor.x * lightColor.x;
	diffuseColor.y += newDiffuseTerm * materialColor.y * lightColor.y;
	diffuseColor.z += newDiffuseTerm * materialColor.z * lightColor.z;


	return diffuseColor;
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
__global__ void raytraceRay(ray* rayLast, glm::vec2 resolution, float time, cameraData cam, int totalDepth, glm::vec3* colors, glm::vec3* radianceBuffer,
                            material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms,
							glm::vec3* lightPos, glm::vec3* lightColor, int numberOfLights, int currentDepth, int* terminateFlag, bool isDOF){


	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	
	ray r; 
	if((x < resolution.x && y < resolution.y && terminateFlag[index] == 0)){

		if(currentDepth == 0){
			r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
			if(isDOF){
				glm::vec3 focalPoint = cam.position + r.direction * cam.focalLength;
				glm::vec3 aperturePoint =	getRandomPointOnAperture(cam.position, cam.view, cam.up, cam.aperture, time*index);
				r.origin = aperturePoint;
				r.direction = glm::normalize(focalPoint - aperturePoint);
			}	
		}
		else
			r = rayLast[index];


		glm::vec3 directRadiance;

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
					dis = sphereIntersectionTest(geoms[i], r, objIntersectPt, objIntersectN);////r
					break;
				case CUBE:
					dis = boxIntersectionTest(geoms[i], r, objIntersectPt, objIntersectN);////r
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


		if(hitCheck == false){ //Terminate the ray
			radianceBuffer[currentDepth + index * totalDepth] = glm::vec3(0,0,0);
			directRadiance = glm::vec3(0,0,0);
			terminateFlag[index] = 1;
		}
		else{
			material mate = materials[hitObjectIndex];
		
			if(mate.emittance != 0){ //Terminate the ray
				radianceBuffer[currentDepth + index * totalDepth] = mate.color * mate.emittance / 5.0f;
				directRadiance = mate.color * mate.emittance / 5.0f;
				terminateFlag[index] = 1;
			}
			else{

				glm::vec3 newEyePositionOut = intersectionPoint - r.direction * (float)RAY_BIAS_AMOUNT;//給一個epsloon避免ray打進去face裡面 ////r
				glm::vec3 newEyePositionIn = intersectionPoint + r.direction * (float)RAY_BIAS_AMOUNT;//給一個epsloon讓ray打進去face裡面 ////r

				//Direct Radiance
				glm::vec3 directRadiance = glm::vec3(0, 0, 0);

				for(int i = 0 ; i < numberOfLights ; i++){
					ray hitPt2LightRay;
					hitPt2LightRay.origin = newEyePositionOut;
					hitPt2LightRay.direction = glm::normalize(lightPos[i] - newEyePositionOut);
					float shortestDisHitPt2Light = glm::length(lightPos[i] - newEyePositionOut);

					bool lightObstructCheck = false;

					for(int i = 0; i < numberOfGeoms; ++i){
						if(materials[i].emittance != 0)//if this is a emitter, then the light will not be obstructed
							continue;

						float disHitPt2Light = -1;
						glm::vec3 objIntersectPt(0, 0, 0);
						glm::vec3 objIntersectN(0, 0, 0);
						switch(geoms[i].type){
							case SPHERE:
								disHitPt2Light = sphereIntersectionTest(geoms[i], hitPt2LightRay, objIntersectPt, objIntersectN);
								break;
							case CUBE:
								disHitPt2Light = boxIntersectionTest(geoms[i], hitPt2LightRay, objIntersectPt, objIntersectN);
								break;
							case MESH:
								break;
						}

						if(disHitPt2Light != -1 &&  disHitPt2Light < shortestDisHitPt2Light && disHitPt2Light > 0){
							lightObstructCheck = true;
							break;
						}
					}

					if(lightObstructCheck == false)
						directRadiance += getDirectRadiance(glm::normalize(newEyePositionOut - lightPos[i]), lightColor[i], r, intersectionNormal, mate.color, mate.specularColor, mate.specularExponent);////r
				}


				//Direct Radiance
				directRadiance /= numberOfLights;
				radianceBuffer[currentDepth + index * totalDepth] = directRadiance;

				++currentDepth;


				//Compute indirect ray
				if(currentDepth < totalDepth){
					thrust::default_random_engine rng(hash(index*time));
					thrust::uniform_real_distribution<float> u01(0,1);
					int restDepth = totalDepth - currentDepth;
					int type = calculateSelfBSDF(r, geoms[hitObjectIndex], newEyePositionIn, newEyePositionOut, intersectionNormal, mate, u01(rng), u01(rng), restDepth); ////r
					if(restDepth == 0)
						terminateFlag[index] = 1;
					else{
						rayLast[index] = r;		
						terminateFlag[index] = 0;
					}
				}
				else{
					terminateFlag[index] = 1;
				}
			}
		}


		glm::vec3 newColor = colors[index];
		if(terminateFlag[index] == 1){
		
			glm::vec3 radiance;
			for(int d = totalDepth; d > 0; d--){
				radiance = radianceBuffer[d - 1 + index * totalDepth] + (float)DEPTH_WEIGHT * radiance;
			}

			if(radiance.x > 1)
				radiance /= radiance.x;
			if(radiance.y > 1)
				radiance /= radiance.y;
			if(radiance.z > 1)
				radiance /= radiance.z;


			glm::vec3 accumulateRadiance = colors[index] * (time - 1);
			newColor = (radiance + accumulateRadiance) / time;
		}
		colors[index] = newColor;
		
	}


}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, bool isDOF){
  
	int traceDepth = 5; //determines how many bounces the raytracer traces
	int numberOfLights = 1;

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
	// send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
	//Create radiance buffer
	glm::vec3* radianceBuffer = new glm::vec3[ traceDepth * (int)renderCam->resolution.x * (int)renderCam->resolution.y];


	glm::vec3* cudaRadianceBuffer = NULL;
	cudaMalloc((void**)&cudaRadianceBuffer, traceDepth * (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3));
	cudaMemcpy( cudaRadianceBuffer, radianceBuffer, traceDepth * (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

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

	glm::vec3* lightSource = new glm::vec3[numberOfLights];
	glm::vec3* lightColor = new glm::vec3[numberOfLights];

	int numberOfLightSource = 0;
	
	for(int i = 0; i < numberOfGeoms; ++i){
		if(materials[i].emittance != 0){
		numberOfLightSource++;
		}
	}

	int totalNumberOfLights = numberOfLights;
	int accumulateIndex = 0;
	bool boolCeil = false;
	int objIndex = 0; 
	while(totalNumberOfLights > 0 && objIndex < numberOfGeoms){
		if(materials[objIndex].emittance == 0){
			++objIndex;
			continue;
		}

		int numberOfLightPerSource;
		if(boolCeil)
			numberOfLightPerSource = (int)ceil((float) totalNumberOfLights / numberOfLightSource);
		else
			numberOfLightPerSource = (int)floor((float) totalNumberOfLights / numberOfLightSource);
		for(int i = 0; i < numberOfLightPerSource; ++i){

			lightSource[accumulateIndex + i] = getRandomPointOnCube(geomList[objIndex], (float)iterations * numberOfLights+ i );
			lightColor[accumulateIndex + i] = materials[objIndex].color / 15.0f * materials[objIndex].emittance; 
		}

		accumulateIndex += numberOfLightPerSource;
		--numberOfLightSource;
		totalNumberOfLights -= numberOfLightPerSource;
		boolCeil = !boolCeil;
		++objIndex;
	}


	glm::vec3* cudaLightPos = NULL;
	cudaMalloc((void**)&cudaLightPos, numberOfLights *sizeof(glm::vec3));
	cudaMemcpy( cudaLightPos, lightSource, numberOfLights * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	glm::vec3* cudaLightColor = NULL;
	cudaMalloc((void**)&cudaLightColor, numberOfLights *sizeof(glm::vec3));
	cudaMemcpy( cudaLightColor, lightColor, numberOfLights * sizeof(glm::vec3), cudaMemcpyHostToDevice);





	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;
	cam.focalLength = renderCam->focalLength;
	cam.aperture = renderCam->aperture;


	ray* initialRay = new ray[1];
	ray newRay;
	newRay.direction = glm::vec3(0,0,0);
	newRay.origin = glm::vec3(0,0,0);
	initialRay[0] = newRay;

	//int* currentDepth = new int[1];
	//currentDepth[0] = 0;


	ray* incidentRay = new ray[(int)renderCam->resolution.x * (int)renderCam->resolution.y];
	ray* cudaIncidentRay = NULL;

	cudaMalloc((void**)&cudaIncidentRay, 1 *sizeof(ray));
	cudaMemcpy( cudaIncidentRay, initialRay, 1 * sizeof(ray), cudaMemcpyHostToDevice);


	cudaMalloc((void**)&cudaIncidentRay, (int)renderCam->resolution.x*(int)renderCam->resolution.y * sizeof(ray));
	cudaMemcpy( cudaIncidentRay, incidentRay, (int)renderCam->resolution.x*(int)renderCam->resolution.y * sizeof(ray), cudaMemcpyHostToDevice);

	int* terminateFlag = new int[(int)renderCam->resolution.x * (int)renderCam->resolution.y];
	for(int i = 0; i < (int)renderCam->resolution.x*(int)renderCam->resolution.y; ++i){
		terminateFlag[i] = 0;
	}
	int* cudaTerminateFlag = NULL;
	cudaMalloc((void**)&cudaTerminateFlag, (int)renderCam->resolution.x*(int)renderCam->resolution.y * sizeof(int));
	cudaMemcpy( cudaTerminateFlag, terminateFlag, (int)renderCam->resolution.x*(int)renderCam->resolution.y * sizeof(int), cudaMemcpyHostToDevice);



	for(int currentDepth = 0; currentDepth < traceDepth; currentDepth++){
		if(currentDepth == 0){
			// kernel launches
			raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaIncidentRay, renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudaRadianceBuffer, 
																cudaMaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaLightPos, cudaLightColor, numberOfLights, currentDepth, cudaTerminateFlag, 0, isDOF);
		}
		else{
			raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaIncidentRay, renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudaRadianceBuffer, 
																cudaMaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaLightPos, cudaLightColor, numberOfLights, currentDepth, cudaTerminateFlag, 0, isDOF);
		}
	}





	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudaRadianceBuffer );
	cudaFree( cudageoms );
	cudaFree( cudaMaterials );
	cudaFree( cudaLightPos );
	cudaFree( cudaLightColor );
	cudaFree( cudaIncidentRay );
	cudaFree( cudaTerminateFlag );

	delete geomList;
	delete lightSource; 
	delete lightColor;
	delete radianceBuffer;
	delete incidentRay;
	delete terminateFlag;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
