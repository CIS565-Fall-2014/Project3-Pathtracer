// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <math.h>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

//#define rayTracer 1

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
		exit(EXIT_FAILURE); 
	}
} 

struct is_dead{  
	__host__ __device__  bool operator()(const ray& r)  
	{    
		return r.isDead;  
	}
};

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
	glm::vec3 a = glm::normalize(glm::cross(view, up));
	glm::vec3 b = glm::normalize(glm::cross(view, a));
	glm::vec3 H = a * glm::length(view) * glm::tan(glm::radians(fov.x));
	glm::vec3 V = b * glm::length(view) * glm::tan(glm::radians(fov.y));
	glm::vec3 M = eye + view;
	glm::vec3 rayDes = M + (2*((float)x/(resolution.x-1)) - 1)*H + (2*((float)y/(resolution.y-1)) - 1)*V;
	//get the ray direction from eye to the destination
	glm::vec3 thisRay = rayDes - eye;

	r.direction = glm::normalize(thisRay);
	r.origin = eye;
	r.tempColor = glm::vec3(1.0f);
	r.isDead = false;
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
		color.x = image[index].x*255.0f/iterations;
		color.y = image[index].y*255.0f/iterations;
		color.z = image[index].z*255.0f/iterations;   //weight for each iteration

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


// loop through all geometry to test ray intersection, returns the geoID that corresponds to intersected geometry
__host__ __device__ int intersectTest(ray r, glm::vec3& intersect, glm::vec3& normal, staticGeom* geoms, int numberOfGeoms, triangle * cudatris){

	if(r.isDead) 
		return -1; //didn't hit anything
	float distMin = -2, dist = -1;
	glm::vec3 tempIntersect(0.0f);
	glm::vec3 tempNormal(0.0f);
	int ID = -1;

	for (int g=0; g<numberOfGeoms; g++){
		if(geoms[g].type == SPHERE){
			dist = sphereIntersectionTest(geoms[g], r, tempIntersect, tempNormal);
		}
		else if(geoms[g].type == CUBE ){
			dist = boxIntersectionTest(geoms[g], r, tempIntersect, tempNormal);

		}
		else if (geoms[g].type == MESH){
			dist = polygonIntersectionTest(geoms[g], r, tempIntersect, tempNormal, cudatris);
		}
		
		if( (distMin < 0 && dist > -0.5f ) || ( distMin > -1 && dist < distMin && dist > -0.5f ) ){
			distMin = dist;  
			ID = g;   
			intersect = tempIntersect; 
			normal = tempNormal;
		}
	}
	return ID;
}

//return true if ray directly hit lights
__host__ __device__ bool LightRayTest(ray r, staticGeom* geoms, int numberOfGeoms, material* materials, triangle * cudatris){
	glm::vec3 intersPoint(0.0f);
	glm::vec3 intersNormal(0.0f);

	//printf("shadow ray: [%f,%f,%f], [%f,%f,%f]\n", sr.origin.x,sr.origin.y,sr.origin.z,sr.direction.x,sr.direction.y,sr.direction.z);
	int geoID = intersectTest(r, intersPoint, intersNormal, geoms, numberOfGeoms, cudatris); 
	if( geoID>-1 && materials[geoms[geoID].materialid].emittance > 0){   //hit light soource
		return true;
	}
	else{
		return false;
	}

}

//calculates the direct lighting for a certain hit point and modify color of that hit
__device__ __host__ void directLighting(float seed, glm::vec3& theColor, glm::vec3& theIntersect, glm::vec3& theNormal, int geoID, int* lights, int numOfLights, material* cudamats, staticGeom* geoms, int numOfGeoms, triangle * cudatris){
	ray shadowRay;
	float rayLen;
	float lightArea;
	glm::vec3 lightNormal;

	int chosenLight = lights[0];
	if( numOfLights > 1){
		thrust::default_random_engine rng(hash(seed));
		thrust::uniform_real_distribution<float> u01(0,1); 
		chosenLight = lights[(int)((float)u01(rng) * numOfLights)]; 
	}
	glm::vec3 Plight;  
	if( geoms[chosenLight].type == CUBE ){
		Plight = getRandomPointOnCube( geoms[chosenLight], seed);
	}
	else if( geoms[chosenLight].type == SPHERE ){
		Plight = getRandomPointOnSphere( geoms[chosenLight], seed);
	}

	shadowRay.direction = glm::normalize(Plight - theIntersect);
	shadowRay.origin = theIntersect + (float)EPSILON * shadowRay.direction;
	int lightID = glm::length(Plight - theIntersect);

	material curMat = cudamats[geoms[geoID].materialid];  //material of the hit goemetry
	if(LightRayTest(shadowRay, geoms, numOfGeoms, cudamats, cudatris)){
		float cosTerm = glm::clamp( glm::dot( theNormal, shadowRay.direction ), 0.0f, 1.0f);  //proportion of facing light
		float cosTerm2 = glm::clamp( glm::dot( lightNormal, -shadowRay.direction ), 0.0f, 1.0f);  //proportion of incoming light
		float areaSampling =  lightArea / (float) pow( rayLen, 2.0f) ;   // dA/r^2
		theColor += cudamats[lightID].emittance * curMat.color * cosTerm * cosTerm2 * areaSampling ;
	}
}

#ifdef rayTracer
//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel (recursive)

__host__ __device__ glm::vec3 raytraceRecursive(ray r, int iteration, float currentIndexOfRefraction, int depth, int maximumDepth, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, light* lightSources, int numberOfLights){

	glm::vec3 bgColor(0.0f);
	glm::vec3 ambientColor(1.0f);
	glm::vec3 phongColor(0.0f), reflColor(0.0f), refraColor(0.0f);;
	glm::vec3 returnColor(0.0f);
	float ka = 0.2f;

	if(depth > maximumDepth)
		return bgColor;

	// intersection test	
	glm::vec3 intersectionPoint, intersectionNormal;
	int intersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
	material mat = materials[geoms[intersIndex].materialid];

	if(intersIndex == -1) return bgColor;

	else if(mat.emittance > 0.0f){ // intersected with light source geometry
		returnColor = mat.color;
	}

	else{ // intersected with actual geometry
	
//		returnColor = ka * ambientColor * materials[geoms[intersIndex].materialid].color;

		if(/*iteration == 0 && */materials[geoms[intersIndex].materialid].hasRefractive == 1)
		{
			float nextIndexOfRefraction = 1.0f;
			glm::vec3 refraDir;
			if(abs(currentIndexOfRefraction - 1) < 0.00001f)  // current ray is in air
			{
				refraDir = calculateRefractionDirection(r.direction, intersectionNormal, currentIndexOfRefraction, materials[geoms[intersIndex].materialid].indexOfRefraction, nextIndexOfRefraction);
			}
			else                                              // current ray is in glass
			{
				refraDir = calculateRefractionDirection(r.direction, -intersectionNormal, currentIndexOfRefraction, 1.0f, nextIndexOfRefraction);
			}

			ray refraRay;
			refraRay.origin = intersectionPoint + 0.01f * refraDir;
			refraRay.direction = refraDir;
			refraColor = raytraceRecursive(refraRay, iteration, nextIndexOfRefraction, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
			returnColor += refraColor;
		}

		if(materials[geoms[intersIndex].materialid].hasReflective == 1)
		{
			glm::vec3 reflDir = calculateReflectionDirection(intersectionNormal, r.direction);
			ray reflRay;
			reflRay.origin = intersectionPoint + 0.01f * reflDir;
			reflRay.direction = reflDir;
			reflColor = raytraceRecursive(reflRay, iteration, 1.0f, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
			returnColor += reflColor;
		}


		if(iteration < numberOfLights){
			if(ShadowRayUnblocked(intersectionPoint, lightSources[iteration].position, geoms, numberOfGeoms, materials))
			{
				glm::vec3 L = glm::normalize(lightSources[iteration].position - intersectionPoint);
				float dot1 = glm::clamp(glm::dot(intersectionNormal, L), 0.0f, 1.0f);
				float dot2 = glm::dot(calculateReflectionDirection(intersectionNormal, -L) ,-r.direction);
				glm::vec3 diffuse = lightSources[iteration].color * 0.5f * materials[geoms[intersIndex].materialid].color * dot1;
				glm::vec3 specular;
				if(abs(materials[geoms[intersIndex].materialid].specularExponent) > 1e-6)
					specular = lightSources[iteration].color * 0.1f * pow(glm::max(dot2, 0.0f), materials[geoms[intersIndex].materialid].specularExponent);
				phongColor +=  diffuse + specular;
				
			}
		}

		returnColor += (5.0f / numberOfLights) * (0.1f * (float)numberOfLights * reflColor + (float)numberOfLights * refraColor);
	}
	return returnColor;
}


__global__ void raytracePrimary(glm::vec2 resolution, int time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, int* lightSources, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  //on every thread, get color for any pixel given pixel(x,y) and camera
  if((x<=resolution.x && y<=resolution.y)){
	  int init_depth = 0;
	  ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
	  colors[index] += raytraceRecursive(r, time, 1.0f, init_depth, rayDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
   }

}
#endif

// TODO: IMPLEMENT THIS FUNCTION
// Core path tracer kernel
__global__ void pathtraceRay(ray* rays, float time, int rayDepth, int numOfRays, glm::vec3* colors, staticGeom* geoms, int numberOfGeoms, material* cudamats, int* lights, int numOfLights, cameraData cam, triangle* cudatris){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (int)ceil(sqrt((float)numOfRays))* y;

	if( index < numOfRays ){
		float seed = (float)index * (float)time * ( (float)rayDepth + 1.0f );
		ray r = rays[index];

		glm::vec3 Pintersect(0.0f);
		glm::vec3 Pnormal(0.0f);
		int intersIndex = intersectTest(r, Pintersect, Pnormal, geoms, numberOfGeoms, cudatris);

		if(intersIndex!=-1){
			material curMat = cudamats[geoms[intersIndex].materialid];
			if( curMat.emittance > 0 ){ //ray ends when hit light source
				colors[r.pixelIndex] += r.tempColor * curMat.color * curMat.emittance; 
				r.isDead = true;
			}
			else{ // for reflection and refraction effect
				if(curMat.hasReflective > 0 || curMat.hasRefractive > 0){
					Fresnel Fres;
					float reflectance;
					glm::vec3 reflectDir, transmitDir;
					if(glm::dot(r.direction,Pnormal)<0){ //ray is outside
						Fres = calculateFresnel(Pnormal,r.direction,1.0f, curMat.indexOfRefraction);
						reflectDir = calculateReflectionDirection(Pnormal, r.direction);
						transmitDir = calculateTransmissionDirection(Pnormal, r.direction, 1.0f, curMat.indexOfRefraction);
					}
					else{ //ray is inside
						Fres = calculateFresnel(-Pnormal,r.direction, curMat.indexOfRefraction, 1.0f);
						reflectDir = calculateReflectionDirection(-Pnormal, r.direction);
						transmitDir = calculateTransmissionDirection(-Pnormal, r.direction, curMat.indexOfRefraction, 1.0f);
					}

					if( curMat.hasRefractive  > 0 && curMat.hasReflective > 0){
						thrust::default_random_engine rng( hash( seed ) );
						thrust::uniform_real_distribution<float> u01(0,1);

						if((float) u01(rng) < Fres.reflectionCoefficient ){ //reflected
							r.direction = reflectDir;
						}
						else{ //transmitted
							r.direction = transmitDir;
						}
					}
					else if(curMat.hasReflective > 0){
						r.direction = reflectDir;	
					}
					else if (curMat.hasRefractive  > 0){
						r.direction = transmitDir;
					}
					r.origin = Pintersect + (float)EPSILON * r.direction;
					if(glm::length(curMat.color)>0)
					r.tempColor *= curMat.color ;
				}
	
				else{
					thrust::default_random_engine rng(hash(seed));
					thrust::uniform_real_distribution<float> u01(0,1);
					if((float) u01(rng) < 0.01f ){ //direct light
						directLighting(seed,r.tempColor,Pintersect,Pnormal,intersIndex,lights,numOfLights, cudamats,geoms, numberOfGeoms, cudatris);
					}
					else{  
						//cos weighted 
						r.direction = calculateCosWeightedRandomDirInHemisphere(Pnormal, (float) u01(rng), (float) u01(rng));
						r.origin = Pintersect + (float)EPSILON * r.direction ;
						float diffuseTerm = glm::clamp( glm::dot( Pnormal,r.direction ), 0.0f, 1.0f);
						r.tempColor *=  diffuseTerm * curMat.color;	
					}
				}	
			}
		}
		else{  //if ray hit nothing
			r.isDead = true;
		}
		rays[index] = r;
	}
}

//initialize the ray pool for cudarays
__global__ void generateRaypool(ray * rayPool, cameraData cam, float iterations,glm::vec3 *colors, staticGeom* geoms, int numberOfGeoms, material* cudamats, int * lightIDs, int numberOfLights, triangle * cudatris){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	//ray r = rayPool[index];

	if( x<= cam.resolution.x && y <= cam.resolution.y ){
		ray r = raycastFromCameraKernel( cam.resolution, iterations, x, y, cam.position, cam.view, cam.up, cam.fov );
		r.pixelIndex = index; 

		if(DEPTH_OF_FIELD){
			glm::vec3 focalPoint = r.origin + r.direction * cam.focalLength / glm::dot(cam.view, r.direction);   //L = f/cos(theta)
			thrust::default_random_engine rng(hash((float)index*iterations));
			thrust::uniform_real_distribution<float> u01(0,1);
			float theta = 2.0f * PI * u01(rng);
			float radius = u01(rng) * cam.aperture;
			glm::vec3 eyeOffset(cos(theta)*radius, sin(theta)*radius, 0);
			glm::vec3 newEyePoint = cam.position + eyeOffset; 
			r.origin = newEyePoint;
			r.direction = glm::normalize(focalPoint - newEyePoint);
		}

		glm::vec3 Pintersect(0.0f);
		glm::vec3 Pnormal(0.0f);
		int geoID = intersectTest(r, Pintersect, Pnormal, geoms, numberOfGeoms, cudatris);
		if( geoID > -1){
			directLighting((float)index*iterations, colors[index], Pintersect, Pnormal,geoID, lightIDs, numberOfLights, cudamats, geoms, numberOfGeoms, cudatris);
		}
		rayPool[index] = r;
	}
}


// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaPathTraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	int traceDepth = 10; //determines how many bounces the raytracer traces

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
	int meshID = -1;
	triangle* cudatris = NULL;

	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		if(geoms[i].type == MESH){
			meshID = i;   // my code now only handles one obj load (unfortunately as I am not able to handle list of triangles well)
			newStaticGeom.boundingBoxMax = geoms[i].boundingBoxMax;  //bBox is in local coordinates, dont change over frames.
			newStaticGeom.boundingBoxMin = geoms[i].boundingBoxMin;
			newStaticGeom.numOfTris = geoms[i].numOfTris;
			cudaMalloc((void**)&cudatris, geoms[meshID].numOfTris*sizeof(triangle));
			cudaMemcpy( cudatris, geoms[meshID].tris, geoms[meshID].numOfTris *sizeof(triangle), cudaMemcpyHostToDevice);
		}
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
	cam.aperture = renderCam->aperture;
	cam.focalLength = renderCam->focalLength;

	// material setup
	material* cudamats = NULL;
	cudaMalloc((void**)&cudamats, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamats, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	//lights setup
	int numberOfLights = 0;
	for(int i = 0; i < numberOfGeoms; ++i){
		if(materials[geoms[i].materialid].emittance > 0){
			numberOfLights ++ ;
		}
	}

	int *lightIDs = new int[numberOfLights];
	int k = 0;
	for(int i = 0; i < numberOfGeoms; ++i){
		if(materials[geoms[i].materialid].emittance > 0){
			lightIDs[k] = i;
			k++;
		}
	}
	int* cudalightIDs = NULL;
	cudaMalloc((void**)&cudalightIDs, numberOfLights*sizeof(int));
	cudaMemcpy( cudalightIDs, lightIDs, numberOfLights*sizeof(int), cudaMemcpyHostToDevice);


	//set up ray pool on device
	ray* cudarays = NULL;
	int numOfRays = cam.resolution.x * cam.resolution.y;
	cudaMalloc((void**)&cudarays, numOfRays*sizeof(ray));
	generateRaypool<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays, cam, (float)iterations, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightIDs, numberOfLights, cudatris);

	for(int cur_depth=0; cur_depth<traceDepth && numOfRays>0; cur_depth++){

		thrust::device_ptr<ray> raypoolStart = thrust::device_pointer_cast(cudarays);  //coverts cuda pointer to thrust pointer
		thrust::device_ptr<ray> raypoolEnd = thrust::remove_if(raypoolStart, raypoolStart + numOfRays, is_dead());
		numOfRays = (int)(raypoolEnd-raypoolStart);

		//xBlocks * yBlocks = numOfRays / (tileSize*tileSize)
		int xBlocks = (int) ceil( sqrt((float)numOfRays)/(float)(tileSize) );
		int yBlocks = (int) ceil( sqrt((float)numOfRays)/(float)(tileSize) );
		dim3 newBlocksPerGrid(xBlocks,yBlocks);

		pathtraceRay<<<newBlocksPerGrid, threadsPerBlock>>>(cudarays, (float)iterations, cur_depth, (int)numOfRays, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightIDs, numberOfLights, cam, cudatris);
	}

	//raytraceRay<<<newBlocksPerGrid, threadsPerBlock>>>(cudarays, (float)iterations, cur_depth, (int)numOfRays, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightIDs, numberOfLights, cam, cudatris);
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,(float)iterations);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamats );
	cudaFree( cudarays );
	cudaFree( cudalightIDs );
	if(meshID>-1){
		cudaFree( cudatris );
	}

	delete geomList;
	delete lightIDs;
	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
