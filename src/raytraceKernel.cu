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
#include "thrust/copy.h"
#include "SOIL/SOIL.h"

#define THRESHOLD 0.005f
#define len(x) sqrtf(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
#define FRESNEL 1
#define DEPTH_FIELD_MODE 0
#define AMBIENT_OCCLUSION_MODE 0
#define NORMAL_MODE 0

glm::vec3 * cudaTextures;

// a simple loadBMP function ### not work 
glm::vec3 * loadBMP(const char * imagepath)
{
	// Data read from the header of the BMP file
	unsigned char header[54]; // Each BMP file begins by a 54-bytes header
	unsigned int dataPos;     // Position in the file where the actual data begins
	unsigned int width, height;
	unsigned int imageSize;   // = width*height*3
	// Actual RGB data
	unsigned char * data;

	// Open the file
	FILE * file = fopen(imagepath, "rb");
	if(!file)
	{
		printf("image could not be opened\n");
		exit(0);
	}

	if( fread(header,1,54,file)!=54){
		printf("Incorrect BMP file! \n");
		exit(0);
	}
	
	if( header[0] != 'B' || header[1] != 'M')
	{
		printf("Incorrect BMP file! \n");
		exit(0);
	}

	dataPos    = *(int*)&(header[0x0A]);
	imageSize  = *(int*)&(header[0x22]);
	width      = *(int*)&(header[0x12]);
	height     = *(int*)&(header[0x16]);
	// Some BMP files are misformatted, guess missing information
	if (imageSize==0)    imageSize=width*height*3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos==0)      dataPos=54; // The BMP header is done that way

	// Create a buffer
	//printf("width = %d, height = %d, image size : %d \n", width,height,imageSize);
	data = new unsigned char [imageSize];
 
	// Read the actual data from the file into the buffer
	fread(data,1,imageSize,file);
	//printf("BMP file loaded! \n");
	fclose(file);

	int i = 0;
	glm::vec3 * texture = new glm::vec3[height * width];
	for( int y = 0; y < height; ++y)
	{
		for(int x = 0; x < width; ++x)
		{ 
			texture[x + y * width] = glm::vec3((float)data[i]/255.0f, (float)data[i+1]/255.0f,(float)data[i+2]/255.0f);
			i=i+3;
		}
 
	}
		//printf("R: %f, G: %f, B: %f \n", (float)data[i],(float)data[i+1],(float)data[i+2]);
	return texture;
}

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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, 
	                                           float focl, float aptr, float time){

  glm::vec3 A = glm::cross(view, up);
  glm::vec3 B = glm::cross(A,view);
  glm::vec3 M = view + eye;
  glm::vec3 V = B * (float)view.length() * tanf(float(fov.y * PI / 180.0)) / (float)B.length();
  glm::vec3 H = A * (float)view.length() * tanf(float(fov.x * PI / 180.0)) / (float)A.length(); 

  thrust::default_random_engine rng(hash((time+1)*(x+1)*(y+1)));
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

  float i = x + (float)u01(rng);
  float j = y + (float)u01(rng);
  glm::vec3 P = M + (float)((2.0*i)/(resolution.x-1.0)-1.0) * H +  (float)(2.0*(resolution.y - j - 1.0)/(resolution.y-1.0)-1.0) * V;
   
  ray r;
  r.origin = P;
  r.direction = glm::normalize(P-eye);

  if(DEPTH_FIELD_MODE)
  {
	  glm::vec3 focalPoint = eye + r.direction *focl;
	  		
	thrust::uniform_real_distribution<float> u02(-aptr/2, aptr/2);
	r.origin = eye + A * u02(rng) + B * u02(rng);
	r.direction = glm::normalize(focalPoint - r.origin);
  }

 
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
      color.x = image[index].x*255.0/time;
      color.y = image[index].y*255.0/time;
      color.z = image[index].z*255.0/time;

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

__host__ __device__ int checkIntersections(ray r, staticGeom* geoms, int numberOfGeoms, glm::vec3 & intersectionPoint, glm::vec3 & normal)
{
	int closestGeo = -1;
	float t = 99999;
	glm::vec3 tmpN, tmpIntersection;
	for(int i = 0; i < numberOfGeoms; ++i)
	{
		float tmp;
		if(geoms[i].type == SPHERE)
			tmp = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == CUBE)
			tmp = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);
		else if(geoms[i].type == MESH)
		{
			  tmp =  meshIntersectionTest(geoms[i], r,  intersectionPoint, normal);
			  /*tmpN = normal;
			  tmpIntersection = intersectionPoint;*/
		}

		if(tmp >= 0 && tmp < t)
		{
			t = tmp;
			closestGeo = i;
		}
	}

	if( closestGeo >= 0 )
	{
		if(geoms[closestGeo].type == SPHERE)
			sphereIntersectionTest(geoms[closestGeo], r, intersectionPoint, normal);
		else if(geoms[closestGeo].type == CUBE)
			boxIntersectionTest(geoms[closestGeo], r, intersectionPoint, normal);
		else if(geoms[closestGeo].type == MESH)
			meshIntersectionTest(geoms[closestGeo], r, intersectionPoint, normal);
		return closestGeo;
	}
	else
		return -1;

}

__global__ void genCameraRayBatch(glm::vec2 resolution, cameraData cam,  ray * rays, float time)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);
	if(x<=resolution.x && y<=resolution.y)
	{
		rays[index] = raycastFromCameraKernel(resolution, x, y, cam.position, cam.view, cam.up, cam.fov, cam.focl, cam.aperture, time);
		rays[index].id = index;
		rays[index].rayColor = glm::vec3(1.0f, 1.0f, 1.0f);
	}
}

__global__ void buildDirectionLightMap(glm::vec2 resolution, cameraData cam,  ray * rays, float time)
{
}

// smoothing kernel
__global__ void averagePixelColor(glm::vec2 resolution,  glm::vec3* colors, float iterations)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
	
    const int n = 5;
	int avgIndex[n*n*4];

	if((x<resolution.x-n && y<resolution.y-n) && x>n && y >n)
    {
			for(int i = -n; i <n; ++i)
			{
				for(int j = -n; j <n; ++j)
					avgIndex[j+n + (i+n)*2 *n]  = (x+i) + ((y+j) * resolution.x);
			}

			glm::vec3 newColor(0.0f,0.0f,0.0f);
			for(int i = 0; i < n*n*4; ++i)
				newColor += colors[avgIndex[i]];
			colors[index] =  newColor/(float)(n*n*4);
	}
}

__host__ __device__ bool isDiffuse(float seed, float diffuseRate)
{
	thrust::default_random_engine rng(hash(seed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) <= diffuseRate) 
		return true;
	else 
		return false;

}

__host__ __device__ bool isReflected(float seed, float IOR, float reflectRate, float refractRate, glm::vec3 normal,
									glm::vec3 idir, glm::vec3 tdir)
{
	float R;
	if(FRESNEL)
	{
		float rs = (IOR * glm::dot(normal, idir) - glm::dot(normal, tdir)) / (IOR * glm::dot(normal, idir) + glm::dot(normal, tdir));
		float rp = (glm::dot(normal, idir) - IOR * glm::dot(normal, tdir)) / (glm::dot(normal, idir) + IOR * glm::dot(normal, tdir));
		R = 0.5f * (rs * rs + rp * rp);
	}
	else
	{
		glm::vec2 r(reflectRate,refractRate);
		r = glm::normalize(r);
		R = r.x;
	}

	thrust::default_random_engine rng(hash(seed));
	thrust::uniform_real_distribution<float> u01(0,1);
	if (u01(rng) <= R) 
		return true;
	else 
		return false;
}

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, glm::vec3* colors, int rayDepth,
                            staticGeom* geoms, int numberOfGeoms, material * cudaMat, ray * rays, glm::vec3 * cudaTextures){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //cuPrintf("number of rays: %d", numRays );
  
  if((x<=resolution.x && y<=resolution.y))
  {
	  if(rays[index].id >= 0)
	  {
		 glm::vec3 intersectionPoint;
		 glm::vec3 normal;
		 int geoIndex = checkIntersections(rays[index], geoms, numberOfGeoms, intersectionPoint, normal);
		 //colors[index] = cudaMat[geoms[geoIndex].materialid].color;
		 if(geoIndex < 0 ) 
		 {
			 rays[index].id = -1;
		 }
		 else 
		 {
			 material mat = cudaMat[geoms[geoIndex].materialid];
			 if(mat.emittance > 0.0001f)
			 {
				 colors[index] += rays[index].rayColor * mat.color * mat.emittance;
				 rays[index].id = -1;
			 }			 
			 else 
			 {
				 float seed= (time/10.0f+1.0f) * (index+1.0f) * (rayDepth+1.0f) ;
				 if((mat.hasReflective > 0.0f || mat.hasRefractive > 0.0f) && !isDiffuse(seed, mat.hasScatter))
				 {
					 float IOR = mat.indexOfRefraction;
					 if(glm::dot(normal, rays[index].direction) > 0)
					 {
						 normal = -normal;
						 IOR = 1.0f/(IOR+THRESHOLD);
					 }

					 glm::vec3 transmittedRay = glm::refract(rays[index].direction, normal, 1.0f/(IOR+THRESHOLD));
					 if(!isReflected(seed,IOR, mat.hasReflective, mat.hasRefractive, normal,rays[index].direction,transmittedRay) && mat.hasRefractive > 0.0f)
					 {
						 if(glm::length(transmittedRay) > THRESHOLD)
						 {
							 rays[index].direction = glm::normalize(transmittedRay);
							 rays[index].origin = intersectionPoint - normal * THRESHOLD;
							 rays[index].rayColor =  rays[index].rayColor * mat.color;
						 }	 
					 }
					 else
					 {
						 glm::vec3 reflectedRay = glm::reflect(rays[index].direction, normal);
						 rays[index].direction = glm::normalize(reflectedRay);
						 rays[index].origin = intersectionPoint + normal * THRESHOLD;
						 rays[index].rayColor =  rays[index].rayColor * mat.color;
					 }
					 return;
				 }

				if(glm::dot(rays[index].direction, normal) > 0)
				{
					normal = -normal;
				}
				thrust::default_random_engine rng(hash(seed));
				thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
				//colors[index] = mat.color;
				//colors[index] += intersectionPoint;
				//glm::vec3 toLight = geoms[7].translation - intersectionPoint;
				rays[index].direction = calculateRandomDirectionInHemisphere(normal,  u01(rng), u01(rng));
				rays[index].origin = intersectionPoint + normal * THRESHOLD;
				if(mat.isTextured > 0)
				{
					glm::vec3 p = multiplyMV(geoms[geoIndex].inverseTransform, glm::vec4(intersectionPoint,1.0f));
					float r = geoms[geoIndex].scale.x * 0.5f;
					int s = 256.0f * fabs(cosf((float)p.z/0.5f)/PI) ;
					int t = 256.0f * fabs(cosf((float)p.x/(float)(r * sinf(PI * s)))/(2.0f * PI));
		
					colors[index] += cudaTextures[s + t * 256];
					//cuPrintf("[s,t] = [%d, %d] with color = (%f, %f, %f) \n", s,t,cudaTextures[s + t * 256].x,cudaTextures[s + t * 256].y,cudaTextures[s + t * 256].z );
					rays[index].id = -1;
				}
				else
					rays[index].rayColor =  rays[index].rayColor * mat.color;
		      }
		   }
	   }
   } 
    //colors[index] = generateRandomNumberFromThread(resolution, time, x, y); 
  
}

__host__ __device__ bool isTerminated(ray r)
{
	if(r.id==-1)
		return false;
	else 
		return true;
}
// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
	int traceDepth = 5; //determines how many bounces the raytracer traces

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

	if(geoms[i].type ==MESH)
	{
		newStaticGeom.faceNum = geoms[i].faceNum;

		for(int k = 0; k < newStaticGeom.faceNum*3; ++k)
		{
			newStaticGeom.faces[k] = geoms[i].faces[k];
			if(k < newStaticGeom.faceNum)
				newStaticGeom.normals[k] = geoms[i].normals[k];
		}
		//std::cout << newStaticGeom.faceNum << std::endl;
		/*for(int j =0; j<newStaticGeom.faceNum; ++j)
		{
			std::cout <<newStaticGeom.faces[3*j].x << " " <<  newStaticGeom.faces[3*j].y << " " << newStaticGeom.faces[3*j].z <<  " | ";
			std::cout << newStaticGeom.faces[3*j+1].x << " " <<  newStaticGeom.faces[3*j+1].y << " " << newStaticGeom.faces[3*j+1].z << " | ";
			std::cout << newStaticGeom.faces[3*j+2].x << " " <<  newStaticGeom.faces[3*j+2].y << " " << newStaticGeom.faces[3*j+2].z << std::endl;
		}*/
	}
	geomList[i] = newStaticGeom;
	}
  
	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
	material* cudaMat = NULL;
	cudaMalloc((void**)&cudaMat, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudaMat, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
	
	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;
	cam.focl = renderCam->focl;
	cam.aperture = renderCam->aperture;

	// copy texture
	glm::vec3 * texture = loadBMP("metal1b.bmp");
	glm::vec3 texture2[256*256];
	for(int i = 0; i < 256 * 256; ++i)
		texture2[i] = texture[i];
	cudaMalloc((void**) &cudaTextures, 256 * 256 * sizeof(glm::vec3));
	cudaMemcpy(cudaTextures, texture2, 256 * 256 * sizeof(glm::vec3), cudaMemcpyHostToDevice);


	// kernel launches
	
	ray * cudaRays;
	cudaMalloc((void**)&cudaRays, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(ray));
	
	
	genCameraRayBatch<<<fullBlocksPerGrid, threadsPerBlock>>>(cam.resolution, cam,  cudaRays, iterations);
	//cudaPrintfInit();
	
	for( int i = 0; i < traceDepth; ++i)
	{	
		raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cudaimage, i, cudageoms, numberOfGeoms, cudaMat,cudaRays, cudaTextures);
	}
	
	//cudaPrintfDisplay(stdout, false);
	//if(iterations == 9999)
	//	averagePixelColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution,cudaimage,iterations);
	
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);
	//
	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	//cudaPrintfEnd();
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudaRays);
	cudaFree( cudaMat);
	cudaFree( cudaTextures);
	delete [] geomList;
	delete [] texture;

	// make certain the kernel has completed
	cudaThreadSynchronize();
	
	checkCUDAError("Kernel failed!");
}
