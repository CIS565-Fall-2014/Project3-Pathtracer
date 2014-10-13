// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/count.h>
#include <thrust\device_vector.h>
#include <thrust\remove.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"


#define	 MAX_TRAVEL_DIST	9999999.99f
#define  ENABLE_AA			1
#define  ENABLE_MOTION_BLUR 0
#define  ENABLE_DOF			1
#define  APERTURE_RADIUS	0.1f
#define  FOCALLEN_LENGTH	1.1f

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 


// TODO: IMPLEMENT THIS FUNCTION
// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye;
  glm::vec3 image_x_direction=glm::cross(view,up);
  glm::vec3 image_y_direction=-up;
  glm::vec3 image_center=eye+view;
  float px=float(x);
  float py=float(y);

  if(ENABLE_DOF)
  {
	  thrust::default_random_engine rng(hash(time+1.0f));
	  thrust::uniform_real_distribution<float> u01(-1.0f,1.0f);
	  r.origin=r.origin+u01(rng)*image_x_direction*APERTURE_RADIUS;
	  r.origin=r.origin+u01(rng)*image_y_direction*APERTURE_RADIUS;
	  image_center=eye+FOCALLEN_LENGTH*view;
  }


  //http://en.wikipedia.org/wiki/Supersampling for Anti Aliasing
   if(ENABLE_AA)
  {
	  thrust::default_random_engine rng(hash((time+1.0f)*(px+2.0f)*(py+3.0f)));
	  thrust::uniform_real_distribution<float> u01(-1.5f,1.5f);
	  px=px+u01(rng);
	  py=py+u01(rng);
  }
  float image_x=((float)px-(float)resolution.x/2)/((float)resolution.x/2);
  float image_y=((float)py-(float)resolution.y/2)/((float)resolution.y/2);
 
 


  float angle_x=fov.x;
  float angle_y=fov.y;
  glm::vec3 image_pos=image_center+image_x*glm::length(view)*tan(angle_x)*glm::normalize(image_x_direction)+image_y*glm::length(view)*tan(angle_y)*glm::normalize(image_y_direction);
  glm::vec3 ray_direction=glm::normalize(image_pos-eye);
  r.direction=ray_direction;
  r.travel_dist=0;
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








// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}




__global__ void InitRays(ray* activeRays, glm::vec2 resolution,float time, cameraData cam)
{
	int x=blockIdx.x*blockDim.x+threadIdx.x;
	int y=blockIdx.y*blockDim.y+threadIdx.y;
	int index=x+y*resolution.x;
	if(x<=resolution.x && y<=resolution.y)
	{
	ray newRay=raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov*(float)PI/180.0f);
	newRay.color=glm::vec3(1.0f);
	newRay.is_Active=true;
	newRay.index=index;
	activeRays[index]=newRay;
	}
}

__global__ void average_image(glm::vec2 resolution,float time,glm::vec3* current_image,glm::vec3* final_image)
{
	int x=blockIdx.x*blockDim.x+threadIdx.x;
	int y=blockIdx.y*blockDim.y+threadIdx.y;
	int index=x+y*resolution.x;
	if(x<=resolution.x && y<=resolution.y)
	{
		//final_image[index]=current_image[index]/(float)time+final_image[index]*(time-1)/(float)time;
		
		final_image[index]=current_image[index]/(float)time+final_image[index]*(time-1)/(float)time;
		glm::clamp(final_image[index],0.0f,1.0f);
		
	}


}




// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(ray* activeRays,int N,int current_depth,glm::vec2 resolution, float time, cameraData cam, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials){
  int index = blockIdx.x*blockDim.x+threadIdx.x;                                                                                       

  if(index<N){
	//test for direction
	//ray newRay = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov*(float)PI/180.0f);
	//colors[index]=newRay.direction;
	  
	  if(activeRays[index].is_Active)
	  {
		  glm::vec3 intersectionPoint, normal;
		  glm::vec3 temp_intersectionPoint,temp_normal;
		  float travelDist(MAX_TRAVEL_DIST);
		  float d;
		  int MaterialID,ObjectID;
		  for(int i=0;i<numberOfGeoms;i++)
		  {
			  if(geoms[i].type==SPHERE)
				  d=sphereIntersectionTest(geoms[i],activeRays[index],temp_intersectionPoint,temp_normal);
			  else if(geoms[i].type==CUBE)
				  d=boxIntersectionTest(geoms[i],activeRays[index],temp_intersectionPoint,temp_normal);
			  if(d>0.0f && d<travelDist)
			  {
				  travelDist=d;
				  intersectionPoint=temp_intersectionPoint;
				  normal=temp_normal;
				  MaterialID=geoms[i].materialid;
				  ObjectID=i;
			  }
		  }
		  if(travelDist<0.0f||travelDist>=MAX_TRAVEL_DIST)
		  {
			  activeRays[index].is_Active=false;
			  return;
		  }
		  material M=materials[MaterialID];
		  activeRays[index].travel_dist+=travelDist;
		  if(M.emittance>0.001f)
		  {
			  colors[activeRays[index].index]=exp(-0.05f*activeRays[index].travel_dist)*M.emittance*M.color*activeRays[index].color;
			  activeRays[index].is_Active=false;
			  return;
		  }
		  else
		  {
			  float randSeed=((float)time+1.0f)*((float)index+2.0f)*((float)current_depth+3.0f);
			  //int flag;
			  //flag=calculateBSDF(randSeed, activeRays[index], geoms,ObjectID,intersectionPoint,normal,M);
			  /*if(flag==0)
			  {
				  activeRays[index].color=glm::vec3(1.0f,0.0f,0.0f);
			  }
			  else if(flag==1)
			  {
				  activeRays[index].color=glm::vec3(0.0f,1.0f,0.0f);
			  }
			  else
			  {
				  activeRays[index].color=glm::vec3(0.0f,0.0f,1.0f);
			  }*/
			  calculateBSDF(randSeed, activeRays[index], geoms,ObjectID,intersectionPoint,normal,M);
			  return;
		  }

	  }
	  else
	  {
		  return;
	  }
   }
}



//helper function for stream compact
struct ray_isActive
{
	__host__ __device__ bool operator()(const ray Ray)
	{
		return !Ray.is_Active;
	}
};


// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  // send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  //send current image to GPU
  glm::vec3* current_cudaimage = NULL;
  cudaMalloc((void**)&current_cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( current_cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);



  //send rays to GPU
  ray* activeRays=NULL;
  int Num_rays=renderCam->resolution.x*renderCam->resolution.y;
  cudaMalloc((void**)&activeRays,Num_rays*sizeof(ray));


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
	 //motion blur
  if(ENABLE_MOTION_BLUR)
  {
	  if(i==6)
	  {
		  newStaticGeom.translation.x-=(float)iterations/3000;
		  newStaticGeom.translation.y+=(float)iterations/3000;
		  glm::mat4 new_transform=utilityCore::buildTransformationMatrix(newStaticGeom.translation,newStaticGeom.rotation,newStaticGeom.scale);
		  newStaticGeom.transform=utilityCore::glmMat4ToCudaMat4(new_transform);
		  newStaticGeom.inverseTransform=utilityCore::glmMat4ToCudaMat4(glm::inverse(new_transform));
	  }
  }
    geomList[i] = newStaticGeom;
  }
  //send geometry
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  //send materials
  material* cudamaterials=NULL;
  cudaMalloc((void**)&cudamaterials,numberOfMaterials*sizeof(material));
  cudaMemcpy(cudamaterials,materials,numberOfMaterials*sizeof(material),cudaMemcpyHostToDevice);

  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
 


 




  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  bool stream_compact=false;
  int traceDepth=10;
  // set up crucial magic
  int tileSize = 16;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  InitRays<<<fullBlocksPerGrid,threadsPerBlock>>>(activeRays, renderCam->resolution,(float)iterations,cam);

  // kernel launches
  int blockSize=64;
  for(int i=0;i<traceDepth;i++)
  {
	if(stream_compact)
	{
	thrust::device_ptr<ray> current_rays(activeRays);
	thrust::device_ptr<ray> new_rays=thrust::remove_if(current_rays,current_rays+Num_rays,ray_isActive());
	Num_rays=new_rays.get()-current_rays.get();
	//printf("%d\n",Num_rays);
	if(Num_rays<1.0f)
		break;
	}
	raytraceRay<<<ceil((float)Num_rays/blockSize),blockSize>>>(activeRays,Num_rays,i,renderCam->resolution, (float)iterations, cam, current_cudaimage, cudageoms, numberOfGeoms,cudamaterials,numberOfMaterials);
  }


  average_image<<<fullBlocksPerGrid,threadsPerBlock>>>(renderCam->resolution,(float)iterations,current_cudaimage,cudaimage);


  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree(current_cudaimage);
  cudaFree(activeRays);
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
