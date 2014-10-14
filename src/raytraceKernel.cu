// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>

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
	r.origin = glm::vec3(0,0,0);
	r.direction = glm::vec3(0,0,-1);
	return r;
}

__global__ void raycastFromCamera(cameraData cam,glm::vec3 ScreenH, glm::vec3 ScreenV, pathray* viewray){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	if((x<=cam.resolution.x && y<=cam.resolution.y)){
		//ray curray;
		//curray.origin=cam.position;
		//curray.direction=glm::normalize(cam.view+(2.0f*x/(float)cam.resolution.x-1)*ScreenH+(2.0f*y/(float)cam.resolution.y-1)*ScreenV);
		viewray[index]=pathray(x,y,cam.position,
			glm::normalize(cam.view+(2.0f*x/(float)cam.resolution.x-1)*ScreenH+(2.0f*y/(float)cam.resolution.y-1)*ScreenV));
		//colors[index] = t<0?glm::vec3(0.0f):glm::vec3(.80f);//generateRandomNumberFromThread(resolution, time, x, y);
	}
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image,float val=0.0f){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y){
		image[index] = glm::vec3(val);
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image,int iter){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;
		color.x = image[index].x*255.0/(float)iter;
		color.y = image[index].y*255.0/(float)iter;
		color.z = image[index].z*255.0/(float)iter;

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


__device__ glm::vec3 TracePath(pathray r, staticGeom* geoms,int numberOfGeoms,material* mats,float time, cameraData cam, int x, int y,glm::vec3& IntClr){
	IntClr=glm::vec3(1.0f);
	if(r.depth>=traceDepth) {
			r.isDead=true;
			IntClr=glm::vec3(0.0f);
			return glm::vec3(0.0f);
	}
	//glm::vec3 IntClr(1.0f);
	for(int i=0;i<traceDepth&&!r.isDead;i++){
		glm::vec3 IntSecPt;
		glm::vec3 normal;
		int geoId;
		float t=-1.0f,p=-2.0f;

		geoId=-1;
		IntSecPt=glm::vec3(0.0f);
		normal=glm::vec3(0.0f);
		for(int i=0;i<numberOfGeoms;i++){

			switch(geoms[i].type){
			case GEOMTYPE::SPHERE:
				p=sphereIntersectionTest(geoms[i],r.curray,IntSecPt,normal);
				t=(t<0||(p>=0&&t>p))?p:t;
				//geoId=(t<0||(p>=0&&t>p))?i:geoId;
				//if(t<0) colors[index]=glm::vec3(0.0f);
				if(epsilonCheck(p,t)) geoId=i;
				break;
			case GEOMTYPE::CUBE:
				p=boxIntersectionTest(geoms[i],r.curray,IntSecPt,normal);
				t=(t<0||(p>=0&&t>p))?p:t;
				//geoId=(t<0||(p>=0&&t>p))?i:geoId;
				//if(t<0) colors[index]=glm::vec3(0.0f);
				if(epsilonCheck(p,t)) geoId=i;
				break;
			case GEOMTYPE::MESH:
				break;
			default:
				break;
			}
		}

		if(geoId<0) {
			//r.state=-1;
			r.isDead=true;
			IntClr=glm::vec3(0.0f);
			return glm::vec3(0.0f);
		}
		else{
			glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time + (i+1), x, y);
			material curmat=mats[geoms[geoId].materialid];
			//r.pushback(geoms[geoId].materialid,ray(IntSecPt,getRandomPointOnSphere(hash(time)),normal),normal);
			if(curmat.emittance>0){
				IntClr*=curmat.emittance*curmat.color;
				r.isDead=true;
				return glm::vec3(0.0f);
			}
			else{
				r.curray.direction=glm::normalize(calculateRandomDirectionInHemisphere(normal,rand.x,rand.y));
				r.curray.origin=IntSecPt+0.001f*r.curray.direction;
			}
			//return mats[geoms[geoId].materialid].color;
			//return (normal+glm::vec3(1.0f))/2.0f;
			//r.depth++;
			//r.state=geoId;
			//return geoms[geoId].materialid;
			IntClr*=curmat.color;
		}
	}
	IntClr=glm::vec3(0.0f);
	return IntClr;
	//__syncthreads();
}
__host__ __device__ bool intersection(staticGeom* geoms, int numberOfGeoms, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, int& geomIndex){
	float t = FLT_MAX;
	//int geomIndex = 0;
	for(int i = numberOfGeoms - 1; i >= 0; --i)
	{	
		float temp;
		if(geoms[i].type == SPHERE)
			temp = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == CUBE)
			temp = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == MESH){
			//printf("hahah tri \n");
			//printf("after haah tri \n");
		}

		if(temp != -1.0f && temp < t)
		{
			t = temp;
			geomIndex = i;		
		}
	}	

	if(t != FLT_MAX){
		//get the intersection point and normal
		if(geoms[geomIndex].type == SPHERE)
			sphereIntersectionTest(geoms[geomIndex], r, intersectionPoint, normal);
		else if(geoms[geomIndex].type == CUBE)
			boxIntersectionTest(geoms[geomIndex], r, intersectionPoint, normal);
		else if(geoms[geomIndex].type == MESH){
			//printf("have tri ");
		}
		return true;
	}
	else
	{
		return false;
	}
}
__host__ __device__ bool diffuseScatter(staticGeom* geoms, int numberOfGeoms, glm::vec3& intersectionPoint, glm::vec3 inNormal, material material,
										glm::vec3 rand, int geomIndex, Fresnel fresnel, float &diffuseScatterCoeff)
{
	glm::vec3 dir = glm::normalize(calculateRandomDirectionInHemisphere(inNormal, rand.x, rand.y));
	glm::vec3 pSample = intersectionPoint + 1/material.reducedScatterCoefficient * dir;
	ray r;
	r.origin = pSample;
	r.direction = -inNormal;
	glm::vec3 newIntersectionPoint;
	glm::vec3 newNormal;
	int newGeomIndex;
	if(intersection(geoms, numberOfGeoms, r, newIntersectionPoint, newNormal, newGeomIndex))
	{
		if(newGeomIndex == geomIndex)
		{
			float r = glm::distance(intersectionPoint, newIntersectionPoint);
			float zr = 1.0f/material.reducedScatterCoefficient;
			float zv = zr + 4 * (1.0f/(3.f * material.reducedScatterCoefficient) * (1.f + fresnel.reflectionCoefficient / 1.f - fresnel.reflectionCoefficient));
			float dr = glm::sqrt(r*r + zr*zr);
			float dv = glm::sqrt(r*r + zv*zv);
			float tr = glm::sqrt(3 * material.absorptionCoefficient.x * material.reducedScatterCoefficient);
			float tr_dr = tr * dr;
			float tr_dv = tr * dv;

			float rd = (tr_dr + 1.0) * glm::exp(-tr_dr)/* * zr *// (glm::pow(dr, 3.f))
				+ (tr_dv + 1.0) * glm::exp(-tr_dv) * zv / (glm::pow(dv,3.0f));

			diffuseScatterCoeff = ((glm::dot(dir ,inNormal) * rd) / (material.reducedScatterCoefficient * material.reducedScatterCoefficient * glm::exp(-material.reducedScatterCoefficient * r))) / 3 * PI;
			intersectionPoint = newIntersectionPoint;
			return true;
		}
	}
	return false;
}



__host__ __device__ void pathTrace(pathray &pr, staticGeom* geoms, material* mats, int geomsNum,cameraData cam, float time,int x, int y, glm::vec3& clr,int lPos){
	//glm::vec3 acol = glm::vec3(1,1,1);
	clr = glm::vec3(1,1,1);
	ray r=pr.curray;
	for(int i = 0; i < traceDepth; i++)
	{
		glm::vec3 intSecPt, normal;
		int geoId;
		if(!intersection(geoms, geomsNum, r, intSecPt, normal, geoId))
		{
			clr = glm::vec3(0,0,0);
			return;
		}

		material curmat = mats[geoms[geoId].materialid];		
		glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time + (i+1), x, y);

		if(curmat.emittance>0){
				clr *= curmat.emittance * curmat.color;
				return;
		}

		//printf("ahhaha\n");
		if(curmat.hasReflective>0.0f&&curmat.hasReflective>rand.z){
			//reflection parts
			glm::vec3 rfDir=calculateReflectionDirection(normal, r.direction);	
			r.origin= intSecPt+0.001f*rfDir;
			r.direction=rfDir;

			Fresnel fsl;
			bool IsOut=glm::dot(rfDir,normal)>0;
			glm::vec3 rfClr,rfractClr;
			if(!IsOut){
				fsl = calculateFresnel(normal, r.direction, 1.0f, curmat.indexOfRefraction, rfClr, rfractClr);
			}
			else{
				fsl=calculateFresnel(-normal, r.direction, curmat.indexOfRefraction,1.0f, rfClr, rfractClr);
			}
			clr*=curmat.color*fsl.reflectionCoefficient;
		}
		else if(curmat.specularExponent>0){
			glm::vec3 newr(glm::normalize(calculateRandomDirectionInHemisphere(normal, rand.x, rand.y)));
			clr*=curmat.color*curmat.specularColor*glm::pow(glm::dot(normal,glm::normalize(newr-r.direction)),curmat.specularExponent);
			r.direction=newr;
			r.origin=intSecPt+0.001f*r.direction;
		}
		else if(curmat.hasScatter>0.0f&&curmat.hasScatter>rand.z){
			//subsurface scattering parts
			Fresnel fsl;
			bool IsOut = glm::dot(r.direction, normal)>0;
			glm::vec3 rfclr, rfractClr;
			if(!IsOut)
				fsl = calculateFresnel(normal, r.direction, 1.0f, curmat.indexOfRefraction, rfclr, rfractClr);
			else
				fsl = calculateFresnel(-normal, r.direction, curmat.indexOfRefraction, 1.0f, rfclr, rfractClr);			
			
			glm::vec3 diffDir;
			float singleScatterCoeff;
			float diffuseScatterCoeff;
			if(diffuseScatter(geoms, geomsNum, intSecPt, normal, curmat, rand, geoId, fsl, diffuseScatterCoeff))
			{				
				glm::vec3 light = getRandomPointOnCube(geoms[lPos], rand.x);
				diffDir  = glm::normalize(light - intSecPt);
				r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rand.x, rand.y));//diffuseDirection;
				r.origin = intSecPt + 0.001f * r.direction;
				clr *= diffuseScatterCoeff;
			}			
		}
		else if(curmat.hasRefractive>0.0f){
			//refraction parts
			glm::vec3 rfDir;//=calculateTransmissionDirection(normal, r.direction);	
			//r.origin= intSecPt+0.001f*rfDir;
			//r.direction=rfDir;
			bool IsOut=glm::dot(r.direction,normal)>0;
			Fresnel fsl;
			glm::vec3 rfClr,rfractClr;
			if(!IsOut){
				fsl = calculateFresnel(normal, r.direction, 1.0f, curmat.indexOfRefraction, rfClr, rfractClr);
			}
			else{
				fsl=calculateFresnel(-normal, r.direction, curmat.indexOfRefraction,1.0f, rfClr, rfractClr);
			}

			float rfcoef=curmat.indexOfRefraction;
			//glm::vec3 rfClr,rfractClr;

			if(rand.z>.5f){
				if(!IsOut)
					rfDir = calculateTransmissionDirection(normal, r.direction, 1.0f, rfcoef);
				else
					rfDir = calculateTransmissionDirection(-normal, r.direction, rfcoef,1.0f);

				r.origin= intSecPt+0.001f*rfDir;
				r.direction=rfDir;
				clr *= fsl.transmissionCoefficient;
			}
			else{
				glm::vec3 rflectDir=calculateReflectionDirection(normal, r.direction);	
				r.origin= intSecPt+0.001f*rflectDir;
				r.direction=rflectDir;
				clr*=fsl.reflectionCoefficient;
			}
		}
		else{
			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rand.x, rand.y));
			r.origin = intSecPt + 0.001f * r.direction;

			clr *= curmat.color;
		}

		
	}
	clr = glm::vec3(0,0,0);
	return;
} 

__host__ __device__ void pathTraceSC(pathray& pr, staticGeom* geoms, material* mats, int geomsNum,cameraData cam, float time,int x, int y, glm::vec3& clr,int lPos){
	//glm::vec3 acol = glm::vec3(1,1,1);
	//clr = glm::vec3(1,1,1);
	ray r=pr.curray;
	
	//for(int i = 0; i < traceDepth; i++)
	{
		glm::vec3 intSecPt, normal;
		int geoId;
		if(!intersection(geoms, geomsNum, r, intSecPt, normal, geoId))
		{
			clr = glm::vec3(0,0,0);
			pr.isDead=true;
			return;
		}

		material curmat = mats[geoms[geoId].materialid];		
		glm::vec3 rand = generateRandomNumberFromThread(cam.resolution, time + (pr.depth+1), x, y);

		if(curmat.emittance>0){
				clr *= curmat.emittance * curmat.color;
				pr.isDead=true;
				return;
		}

		//printf("ahhaha\n");
		if(curmat.hasReflective>0.0f&&curmat.hasReflective>rand.z){
			//reflection parts
			glm::vec3 rfDir=calculateReflectionDirection(normal, r.direction);	
			r.origin= intSecPt+0.001f*rfDir;
			r.direction=rfDir;

			Fresnel fsl;
			bool IsOut=glm::dot(rfDir,normal)>0;
			glm::vec3 rfClr,rfractClr;
			if(!IsOut){
				fsl = calculateFresnel(normal, r.direction, 1.0f, curmat.indexOfRefraction, rfClr, rfractClr);
			}
			else{
				fsl=calculateFresnel(-normal, r.direction, curmat.indexOfRefraction,1.0f, rfClr, rfractClr);
			}
			clr*=curmat.color*fsl.reflectionCoefficient;
		}
		else if(curmat.specularExponent>0){
			glm::vec3 newr(glm::normalize(calculateRandomDirectionInHemisphere(normal, rand.x, rand.y)));
			clr*=curmat.color*curmat.specularColor*glm::pow(glm::dot(normal,glm::normalize(newr-r.direction)),curmat.specularExponent);
			r.direction=newr;
			r.origin=intSecPt+0.001f*r.direction;
		}
		else if(curmat.hasScatter>0.0f&&curmat.hasScatter>rand.z){
			//subsurface scattering parts
			Fresnel fsl;
			bool IsOut = glm::dot(r.direction, normal)>0;
			glm::vec3 rfclr, rfractClr;
			if(!IsOut)
				fsl = calculateFresnel(normal, r.direction, 1.0f, curmat.indexOfRefraction, rfclr, rfractClr);
			else
				fsl = calculateFresnel(-normal, r.direction, curmat.indexOfRefraction, 1.0f, rfclr, rfractClr);			
			
			glm::vec3 diffDir;
			float singleScatterCoeff;
			float diffuseScatterCoeff;
			if(diffuseScatter(geoms, geomsNum, intSecPt, normal, curmat, rand, geoId, fsl, diffuseScatterCoeff))
			{				
				glm::vec3 light = getRandomPointOnCube(geoms[lPos], rand.x);
				diffDir  = glm::normalize(light - intSecPt);
				r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rand.x, rand.y));//diffuseDirection;
				r.origin = intSecPt + 0.001f * r.direction;
				clr *= diffuseScatterCoeff;
			}			
		}
		else if(curmat.hasRefractive>0.0f){
			//refraction parts
			glm::vec3 rfDir;//=calculateTransmissionDirection(normal, r.direction);	
			//r.origin= intSecPt+0.001f*rfDir;
			//r.direction=rfDir;
			bool IsOut=glm::dot(r.direction,normal)>0;
			Fresnel fsl;
			glm::vec3 rfClr,rfractClr;
			if(!IsOut){
				fsl = calculateFresnel(normal, r.direction, 1.0f, curmat.indexOfRefraction, rfClr, rfractClr);
			}
			else{
				fsl=calculateFresnel(-normal, r.direction, curmat.indexOfRefraction,1.0f, rfClr, rfractClr);
			}

			float rfcoef=curmat.indexOfRefraction;
			//glm::vec3 rfClr,rfractClr;

			if(rand.z>.5f){
				if(!IsOut)
					rfDir = calculateTransmissionDirection(normal, r.direction, 1.0f, rfcoef);
				else
					rfDir = calculateTransmissionDirection(-normal, r.direction, rfcoef,1.0f);

				r.origin= intSecPt+0.001f*rfDir;
				r.direction=rfDir;
				clr *= fsl.transmissionCoefficient;
			}
			else{
				glm::vec3 rflectDir=calculateReflectionDirection(normal, r.direction);	
				r.origin= intSecPt+0.001f*rflectDir;
				r.direction=rflectDir;
				clr*=fsl.reflectionCoefficient;
			}
		}
		else{
			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rand.x, rand.y));
			r.origin = intSecPt + 0.001f * r.direction;

			clr *= curmat.color;
		}

		pr.curray=r;
	}
	//clr = glm::vec3(0,0,0);
	pr.depth++;
	if(pr.depth>=traceDepth) clr=glm::vec3(0.0f);
	return;
} 


// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(float time, cameraData cam,staticGeom* geoms, material* mats,int numberOfGeoms, pathray* rays,glm::vec3* img,int lPos){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if(x<cam.resolution.x&&y<cam.resolution.y/*&&!rays[index].isDead*/){	
		int onscreen=rays[index].onscreen.x+rays[index].onscreen.y*cam.resolution.x;
		
		glm::vec3 clr(1.0f);
		pathTrace(rays[index],geoms,mats, numberOfGeoms,cam,time,x,y,clr,lPos);
		img[onscreen]+=clr;


	}
}

__global__ void raytraceRaySC(int poolSize,float time, cameraData cam,staticGeom* geoms, material* mats,int numberOfGeoms, pathray* rays,glm::vec3* img,int lPos){
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(index<poolSize){	
		int onscreen=rays[index].onscreen.x+rays[index].onscreen.y*cam.resolution.x;
		
		glm::vec3 clr=img[onscreen];
		pathTraceSC(rays[index],geoms,mats, numberOfGeoms,cam,time,rays[index].onscreen.x,rays[index].onscreen.y,clr,lPos);
		img[onscreen]=clr;


	}
}

__global__ void imageAdd(glm::vec3* dst,glm::vec3* src, cameraData cam){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);
	if(x<cam.resolution.x&&y<cam.resolution.y/*&&!rays[index].isDead*/){	
		dst[index]+=src[index];
	}
}
// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){

	//determines how many bounces the raytracer traces
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0);
	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	// send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

	if(iterations==1){
		clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution,cudaimage);
	}
	// package geometry and materials and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	int lPos;
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
		if(materials[newStaticGeom.materialid].emittance>0)
			lPos=i;
	}

	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	material* cudaMaterials=NULL;
	cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy(cudaMaterials,materials,numberOfMaterials*sizeof(material),cudaMemcpyHostToDevice);
	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	glm::vec3 A=glm::normalize(glm::cross(cam.up,cam.view));
	glm::vec3 ScreenH=A*(float)tan(cam.fov.x*PI/180.0f)*glm::length(cam.view);
	glm::vec3 ScreenV=glm::normalize(glm::cross(A,cam.view))*(float)tan(cam.fov.y*PI/180.0f)*glm::length(cam.view);

	pathray* rays=NULL;
	cudaMalloc((void**)&rays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(pathray));
	//initial rays from camera;
	raycastFromCamera<<<fullBlocksPerGrid, threadsPerBlock>>>(cam,ScreenH,ScreenV,rays);
	// kernel launches
	//while(!rays[0].isDead)
	//for(int i=0;i<traceDepth;i++){

#ifndef _STREAM_COMPACT
	raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>((float)iterations, cam, cudageoms, cudaMaterials, numberOfGeoms,rays,cudaimage,lPos);
#else
	//pathray* pool=rays;
	//cudaMalloc((void**)&rays, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(pathray));
	
	glm::vec3* cacheimage = NULL;
	cudaMalloc((void**)&cacheimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	//if(iterations==1){
	clearImage<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution,cacheimage,1.0f);
	//}
	int poolSize=(int)renderCam->resolution.x*(int)renderCam->resolution.y;
	
	int curDepth=0;
	while(poolSize>0&&curDepth<traceDepth){
		//printf("poolSize=%d\n",poolSize);
		curDepth++;
		dim3 threadsPerBlock(2*tileSize*tileSize);
		fullBlocksPerGrid=dim3((int)ceil(float(poolSize)/float(tileSize*tileSize)));
		raytraceRaySC<<<fullBlocksPerGrid, threadsPerBlock>>>(poolSize,(float)iterations, cam, cudageoms, cudaMaterials, numberOfGeoms,rays,cacheimage,lPos);
		cudaThreadSynchronize();
		thrust::device_ptr<pathray> iteratorStart(rays);
		thrust::device_ptr<pathray> iteratorEnd=iteratorStart+poolSize;
		iteratorEnd=thrust::remove_if(iteratorStart, iteratorEnd,IsDead());
		poolSize=(int)(iteratorEnd-iteratorStart);
		
	}
	//printf("curDepth=%d\n",curDepth);
	fullBlocksPerGrid=dim3((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
	//thrust::copy_if(rays,rays+poolSize,pool,NotDead());
	
	imageAdd<<<fullBlocksPerGrid, threadsPerBlock>>>(cudaimage,cacheimage,cam);
	cudaFree(cacheimage);
	//if(poolSize>0) cudaFree(pool);
#endif
	
	
	
	//cudaThreadSynchronize();
	//}
	
	//compose<<<fullBlocksPerGrid, threadsPerBlock,2*tileSize*tileSize*sizeof(glm::vec3)>>>(cudaimage,rays,cudaMaterials,cam);
		
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,iterations);
	
	//cudaThreadSynchronize();
	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	//checkCUDAError("11 Kernel failed!");
	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( rays );
	delete geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();
	cudaEventRecord( stop, 0);
	cudaEventSynchronize( stop );

	float seconds = 0.0f;
	cudaEventElapsedTime( &seconds, start, stop);
  
	printf("One Loop time:  %f ms\n", seconds);
	checkCUDAError("Kernel failed!");
}
