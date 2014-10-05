// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

extern bool streamcompact_b;
extern bool texturemap_b;

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
	glm::vec3 A = glm::cross(view,up);
	glm::vec3 B = glm::cross(A,view);
	glm::vec3 M = eye + view;
	float angley = fov.y;
	float anglex= fov.x;
	glm::vec3 H =  glm::normalize(A) * glm::length(view) * tan(glm::radians(anglex));
	glm::vec3 V =  glm::normalize(B) * glm::length(view) * tan(glm::radians(angley));

	float sx = ((float)x)/(resolution.x - 1);
	float sy = ((float)y)/(resolution.y - 1);

	glm::vec3 P = M + (2.0f*sx-1)* H + (1-2.0f*sy) * V; //The picture begins
	glm::vec3 D = glm::normalize(P - eye);
	ray r;
	r.origin = eye;
	r.direction = D;
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image,float iterations){

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;
		color.x = image[index].x*255.0;
		color.y = image[index].y*255.0;
		color.z = image[index].z*255.0;
		color /= iterations;

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
__global__ void PathTraceColor(ray* remainrays,int raysnum,int currdepth,int maxdepth,
	staticGeom* geoms, int numberOfGeoms, int* lightIndex, 
	int lightNum,material* materials,float time,uint3* tcolors,int* tnums,bool textureb)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index<=raysnum)
	{
		ray r = remainrays[index];
		if(!r.exist) return;

		//clear all ray if currdepth == maxdepth
		if(currdepth==maxdepth)
		{
			r.exist = false;
			remainrays[index] = r;		
			return;
		}

		bool Intersect = false;
		glm::vec3 InterSectP,InterSectN;
		int IntersectgeomId = -1;
		Intersect = Intersecttest(r,InterSectP,InterSectN,geoms,numberOfGeoms,IntersectgeomId);

		//if the ray intersect with nothing, give it black/backgroundcolor
		if(Intersect==false)
		{
			r.raycolor = glm::vec3(0,0,0);
			r.exist = false;
			remainrays[index] = r;		
			return;
		}


		material currMaterial = materials[geoms[IntersectgeomId].materialid];	
		if(textureb)
			textureMap(geoms,IntersectgeomId,currMaterial,InterSectN,InterSectP,tcolors,tnums);

		bool IsLight = false;
		for(int i=0;i<lightNum;i++)
		{
			if(IntersectgeomId==lightIndex[i])
				IsLight = true;
		}

		if(IsLight)
		{
			r.raycolor = r.raycolor * currMaterial.color * currMaterial.emittance;
			r.exist = false;
		}
		else
		{
			int seed = (index+1) * (time/2 + currdepth);
		    int BSDF = calculateBSDF(r,InterSectP,InterSectN,currMaterial,seed,currdepth);	
			if(BSDF==0)
				r.raycolor = r.raycolor * currMaterial.color;
		}
		

		remainrays[index] = r;		
	}
}

//Changed
__global__ void AddColor(glm::vec3* colors, ray* remainrays,int raysnum)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index<=raysnum){
		ray r = remainrays[index];
		if(r.exist==false)
		   colors[r.initindex] += r.raycolor;
	}
}

__global__ void InitRays(ray* rays, glm::vec2 resolution, cameraData cam, float time)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);

	if((x<=resolution.x && y<=resolution.y))
	{
		//anti-aliasing
		thrust::default_random_engine rng(hash(index*time));
		thrust::uniform_real_distribution<float> u01(0, 1);
		ray r = raycastFromCameraKernel(resolution,0.0f, x + float(u01(rng)) -0.5f, y+float(u01(rng))-0.5f,cam.position,cam.view,cam.up,cam.fov);
		r.exist = true;
		r.initindex = index;
		r.raycolor = glm::vec3(1,1,1);
		r.IOR = 1.0f;
		rays[index] = r;
	}
}

struct Is_Exist
{
	__host__ __device__
	bool operator()(const ray x)
	{
		if(x.exist) return true;
		else return false;
	}
};

struct Is_Not_Exist
{
	__host__ __device__
	bool operator()(const ray x)
	{
		if(!x.exist) return true;
		else return false;
	}
};

//StreamCompact
void ThrustStreamCompact(thrust::device_ptr<ray> origin,int &N)
{
	//Count how many rays still exist
	int finallength = thrust::count_if(origin, origin+N,Is_Exist());
	thrust::remove_if(origin, origin+N,Is_Not_Exist());
	N = finallength;
	return;
}


// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials,
	int numberOfMaterials, geom* geoms, int numberOfGeoms,std::vector<uint3> mapcolors,std::vector<int> maplastnums){

	int traceDepth = 8; //determines how many bounces the raytracer traces

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
		newStaticGeom.transinverseTransform = geoms[i].transinverseTransforms[frame];
		newStaticGeom.tri = geoms[i].tri;
		newStaticGeom.trinum = geoms[i].trinum;
		newStaticGeom.texindex = geoms[i].texindex;
		newStaticGeom.theight = geoms[i].theight;
		newStaticGeom.twidth = geoms[i].twidth;
		geomList[i] = newStaticGeom;
	}

	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	//materials
	material* materialList = new material[numberOfMaterials];
	for(int i=0; i<numberOfMaterials; i++){
		material newMaterial;
		newMaterial.color = materials[i].color;

		//specular is useless as the highlight area color is decided by light
		newMaterial.specularExponent = materials[i].specularExponent;
		newMaterial.specularColor = materials[i].specularColor;  
		newMaterial.hasReflective = materials[i].hasReflective;
		newMaterial.hasRefractive = materials[i].hasRefractive;
		newMaterial.indexOfRefraction = materials[i].indexOfRefraction;
		newMaterial.hasScatter = materials[i].hasScatter;
		newMaterial.absorptionCoefficient = materials[i].absorptionCoefficient;
		newMaterial.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
		newMaterial.emittance = materials[i].emittance;
		materialList[i] = newMaterial;
	}

	material* cudamaterials = NULL;
	cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	//light
	int lcount = 0;
	for(int i=0; i<numberOfGeoms; i++)
	{
		if(materials[geomList[i].materialid].emittance>0)
			lcount++;
	}

	int *lightIds = new int[lcount];
	lcount = 0;
	for(int i=0; i<numberOfGeoms; i++)
	{
		if(materials[geomList[i].materialid].emittance>0)
		{
			lightIds[lcount] = i;
			lcount++;
		}
	}

	int *cudalightIds=NULL;
	cudaMalloc((void**)&cudalightIds,lcount * sizeof(int));
	cudaMemcpy( cudalightIds, lightIds,lcount * sizeof(int), cudaMemcpyHostToDevice);

	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;

	//Transfer Texture Map
	uint3* cudacolors = NULL;
	int* cudalastnums = NULL;
	if(texturemap_b)
	{
		if(iterations==1  && (maplastnums.size()==0||mapcolors.size()==0))
		{
			std::cout<<"No Texture Map Set!"<<std::endl;
			texturemap_b = false;
		}
		uint3 *allcolors = new uint3[(int)mapcolors.size()];
	    int *alllastnum = new int[(int)maplastnums.size()];
        for(int i=0;i<(int)mapcolors.size();i++)
		    allcolors[i] = mapcolors[i];

	    for(int i=0;i<(int)maplastnums.size();i++)
		    alllastnum[i] = maplastnums[i];


	    cudaMalloc((void**)&cudacolors, (int)mapcolors.size()*sizeof(uint3));
	    cudaMemcpy( cudacolors, allcolors, (int)mapcolors.size()*sizeof(uint3), cudaMemcpyHostToDevice);
		
	    cudaMalloc((void**)&cudalastnums, (int)maplastnums.size()*sizeof(int));
	    cudaMemcpy( cudalastnums, alllastnum, (int)maplastnums.size()*sizeof(int), cudaMemcpyHostToDevice);
		delete allcolors;
		delete alllastnum;
	}
	


	//set up init rays
	int numberOfInitrays = renderCam->resolution.x*renderCam->resolution.y;
	ray* cudarays = NULL;
	cudaMalloc((void**)&cudarays, numberOfInitrays*sizeof(ray));
	InitRays<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays,renderCam->resolution,cam,(float)iterations);

	//set path trace dim
	int raythreadsPerBlock = (int)(tileSize*tileSize);
	int rayblocksPerGrid = ceil((float)numberOfInitrays/(float)raythreadsPerBlock);

	// kernel launches
	for(int i=0;i<=traceDepth;i++)
	{
		if(numberOfInitrays>0)
		{
			PathTraceColor<<<rayblocksPerGrid, raythreadsPerBlock>>>(cudarays,numberOfInitrays,i,traceDepth,cudageoms,
				numberOfGeoms,cudalightIds,lcount,cudamaterials,(float)iterations,cudacolors,cudalastnums,texturemap_b);
			AddColor<<<rayblocksPerGrid, raythreadsPerBlock>>>(cudaimage, cudarays,numberOfInitrays);
			if(streamcompact_b)
			{
				thrust::device_ptr<ray> rayStart(cudarays);
			    ThrustStreamCompact(rayStart,numberOfInitrays);
			}			
		}	
	}


	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,(float)iterations);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	//Added
	cudaFree( cudalightIds );
	cudaFree( cudamaterials );
	cudaFree( cudarays );
	cudaFree( cudacolors );
	cudaFree( cudalastnums );

	delete geomList;
	//Added
	delete materialList;
	delete lightIds;
	

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
