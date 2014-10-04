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

__host__ __device__ glm::vec3 raytraceRecursive(ray r, int depth, glm::vec3* lightPos, glm::vec3* lightColor, int numberOfLights,
												material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms){

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

		glm::vec3 newEyePositionOut = intersectionPoint - r.direction * (float)RAY_BIAS_AMOUNT;//給一個epsloon避免ray打進去face裡面
		glm::vec3 newEyePositionIn = intersectionPoint + r.direction * (float)RAY_BIAS_AMOUNT;//給一個epsloon讓ray打進去face裡面
		
		////Create Reflect Ray
		//ray newReflectRay;
		//newReflectRay.origin = newEyePositionOut;
		//newReflectRay.direction = glm::normalize( r.direction - intersectionNormal * 2.0f * glm::dot(r.direction, intersectionNormal));

		////Create random number
		//thrust::default_random_engine rng(hash(depth * intersectionPoint.x));//TODO 
		//thrust::uniform_real_distribution<float> u01(0,1);

   
		//ray newDiffuseRay;
		//newDiffuseRay.origin  = newEyePositionOut;
		//newReflectRay.direction = calculateRandomDirectionInHemisphere(intersectionNormal, (float)u01(rng), (float)u01(rng));


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

			if(lightObstructCheck == false){
				directRadiance += getDirectRadiance(glm::normalize(newEyePositionOut - lightPos[i]), lightColor[i], r, intersectionNormal, mate.color, mate.specularColor, mate.specularExponent);
			}

		}


		directRadiance /= numberOfLights;

		



		//Reflect Color
		glm::vec3 reflectColor;
		//if(mate.hasReflective == 1)
		//	reflectColor = raytraceRecursive(newReflectRay, --depth, lightPos, lightColor, numberOfLights, materials, numberOfMaterials, geoms, numberOfGeoms);
		//else if(mate.hasReflective == 0)
			reflectColor = glm::vec3(0, 0, 0);

		//Refract Color
		glm::vec3 refractColor;
		if(mate.hasRefractive == 1)
			refractColor = glm::vec3(0,0,0);
		else if(mate.hasRefractive == 0)
			refractColor = glm::vec3(0,0,0);


		glm::vec3 currentPtColor = directRadiance;// + reflectColor;
		return currentPtColor;
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
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors, glm::vec3* radianceBuffer,
                            material* materials, int numberOfMaterials, staticGeom* geoms, int numberOfGeoms,
							glm::vec3* lightPos, glm::vec3* lightColor, int numberOfLights ){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	glm::vec3 pixelColor(0, 0, 0);

	int currentDepth = 0;
	if((x < resolution.x && y < resolution.y )){
		ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
		

		while(currentDepth < rayDepth){
		
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


			if(hitCheck == false){ //Terminate the ray
				radianceBuffer[currentDepth + index * rayDepth] = glm::vec3(0,0,0);
				directRadiance = glm::vec3(0,0,0);
				//TODO: save the direct radiance
				break;
			}
			else{
				material mate = materials[hitObjectIndex];
		
				if(mate.emittance != 0){ //hit light, so terminate the ray
					radianceBuffer[currentDepth + index * rayDepth] = mate.color * mate.emittance / 5.0f;
					directRadiance = mate.color * mate.emittance / 5.0f;
					//TODO: save the direct radiance
					break;
				}

				glm::vec3 newEyePositionOut = intersectionPoint - r.direction * (float)RAY_BIAS_AMOUNT;//給一個epsloon避免ray打進去face裡面
				glm::vec3 newEyePositionIn = intersectionPoint + r.direction * (float)RAY_BIAS_AMOUNT;//給一個epsloon讓ray打進去face裡面

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

					if(lightObstructCheck == false){

						directRadiance += getDirectRadiance(glm::normalize(newEyePositionOut - lightPos[i]), lightColor[i], r, intersectionNormal, mate.color, mate.specularColor, mate.specularExponent);
					}

				}


				//Direct Radiance
				directRadiance /= numberOfLights;
				radianceBuffer[currentDepth + index * rayDepth] = directRadiance;

				++currentDepth;

				thrust::default_random_engine rng(hash(index*time));
				thrust::uniform_real_distribution<float> u01(0,1);

				//Compute indirect ray
				int restDepth = rayDepth - currentDepth;
				if(currentDepth < rayDepth){
					int type = calculateSelfBSDF(r, geoms[hitObjectIndex], newEyePositionIn, newEyePositionOut, intersectionNormal, mate, u01(rng), u01(rng), restDepth);
					currentDepth = rayDepth - restDepth;			
				}


			}
		}

		glm::vec3 radiance;
		for(int d = rayDepth; d > 0; d--){

			radiance = radianceBuffer[d - 1 + index * rayDepth] + (float)DEPTH_WEIGHT * radiance;

			//newRadiance = DirectMat1 * L + DEPTH_WEIGHT * indirectMat1 * ( DirectMat2 * L + weigth2 * indirectMat2 * ( DirectMat3 * L + weigth3 * indirectMat3 ) )
			//Radiance = DirectMat1 * L + weigth1 * indirectMat1 * ( DirectMat2 * L + weigth2 * indirectMat2 * ( DirectMat3 * L ) )
		}

		if(radiance.x > 1)
			radiance /= radiance.x;
		if(radiance.y > 1)
			radiance /= radiance.y;
		if(radiance.z > 1)
			radiance /= radiance.z;
		//glm::vec3 newColorEnergy = raytraceRecursive(r, rayDepth, lightPos, lightColor, numberOfLights, materials, numberOfMaterials, geoms, numberOfGeoms);


		glm::vec3 accumulateRadiance = colors[index] * (time - 1);
		glm::vec3 newColor = (radiance + accumulateRadiance) / time;


		colors[index] = newColor;
	}


}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
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
			//lightSource[accumulateIndex + i] = glm::vec3(0,8,0.25);
			lightColor[accumulateIndex + i] = materials[objIndex].color / 15.0f * materials[objIndex].emittance; 
		}

		accumulateIndex += numberOfLightPerSource;
		--numberOfLightSource;
		totalNumberOfLights -= numberOfLightPerSource;
		boolCeil = !boolCeil;
		++objIndex;
	}

	//for(int i = 0; i < numberOfLights; ++i){
	//	lightSource[i] =  getRandomPointOnCube(geomList[8], (float)iterations * i );
	//	lightColor[i] = glm::vec3(1,1,1);
	//}

 


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

	// kernel launches
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudaRadianceBuffer, 
														cudaMaterials, numberOfMaterials, cudageoms, numberOfGeoms, cudaLightPos, cudaLightColor, numberOfLights);

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

	delete geomList;
	delete lightSource; 
	delete lightColor;
	delete radianceBuffer;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
