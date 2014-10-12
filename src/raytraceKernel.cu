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

#include <thrust/device_ptr.h> 
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//////////////// Stream Compaction //////////////
struct is_ray_dead
  {
    __host__ __device__
    bool operator()(const ray r)
    {
		return r.isDead;
    }
  };

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

bool nearlyEqual(float a, float b)
{
        const float absA = fabs(a);
        const float absB = fabs(b);
        const float diff = fabs(a - b);

        if (a == b) { // shortcut
            return true;
        } else if (a * b == 0) { // a or b or both are zero
            // relative error is not meaningful here
            return diff < (EPSILON * EPSILON);
        } else { // use relative error
            return diff / (absA + absB) < EPSILON;
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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, int index){
  ray r;
 // r.origin = glm::vec3(0,0,0);
 // r.direction = glm::vec3(0,0,-1);
  r.origin = eye;

  glm::vec3 E = eye;
  glm::vec3 C = view;
  glm::vec3 U = up;

  glm::vec3 A = glm::cross(C, U);
  glm::vec3 B = glm::cross(A, C);
  glm::vec3 M = E + C;  

  float tan_phi = tan((float)PI*fov.x/180);
  float tan_theta = tan((float)PI*fov.y/180); 
  glm::vec3 tmp1 = A * glm::length(C) * tan_phi;
  glm::vec3 tmp2 = B * glm::length(C) * tan_theta;
  glm::vec3 H = tmp1 / glm::length(A);
  glm::vec3 V = tmp2 / glm::length(B);

  float r1 = generateRandomNumberFromThread(resolution,time,x,y)[0] - 0.5;
  float randx = x + r1;
  float randy = y + r1;
  float sx = randx / (resolution.x-1);
  float sy = randy / (resolution.y-1);

  glm::vec3 P = M + (2.0f*sx - 1) * H + (1 - 2.0f*sy) * V;
  r.direction = glm::normalize(P-E);

   //Depth of field  
  //http://ray-tracer-concept.blogspot.com/2011/12/depth-of-field.html
  
  //pointAimed is the position of pixel on focal plane in specified ray
  float length = 13.0f;
  float r2 = generateRandomNumberFromThread(resolution,time,x,y)[0]/2.5;
  glm::vec3 pointAimed = r.origin + length * r.direction;
  r.origin += glm::vec3(r2,r2,r2);
  r.direction = pointAimed - r.origin;
  r.direction = glm::normalize(r.direction);

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

__device__ ray calcReflectRay(ray Ri, glm::vec3 normal, glm::vec3 interP)
{
	ray Rr;
	Rr.origin = interP;
	Rr.direction = glm::normalize(Ri.direction) - 2.0f * glm::dot(normal, Ri.direction) * normal; 
	Rr.direction = glm::normalize(Rr.direction);
	Rr.color = Ri.color;

	return Rr;
}

//http://ray-tracer-concept.blogspot.com/2011/12/refraction.html
__device__ ray calcRefractRay(ray Ri, glm::vec3 normal, glm::vec3 interP, float indexOfRefraction)
{
	ray Rr = calcReflectRay(Ri, normal, interP);
	ray Rrr;
	Rrr.origin = interP;

	float nr; glm::vec3 tmpnorm = normal;
	if(glm::dot(normal, Ri.direction) < 0)
	{ // going into the sphere
		 nr = (float)1/indexOfRefraction;
    }
	 else { // going out of sphere
	   nr = indexOfRefraction;
	   tmpnorm = -1.0f * normal;
	}
   float rootContent = sqrtf(1 - nr * nr * (1 - (glm::dot(tmpnorm, -1.0f * Ri.direction)) * glm::dot(tmpnorm, -1.0f * Ri.direction))); 
   if(rootContent >= EPSILON){
	   Rrr.direction = ((nr*glm::dot(tmpnorm, -1.0f * Ri.direction)) - rootContent) * tmpnorm - (-1.0f) *(nr * Ri.direction);  
	   Rrr.direction = glm::normalize(Rrr.direction);
		return Rrr;
	}
}

__device__ float rayIntersect(ray Ri, staticGeom* geoms, int numberOfGeoms, int& closestGeo, glm::vec3& interP, glm::vec3& norm)
{
	//return shortest distance, interP, norm, closestGeoID
	glm::vec3 tmpnorm = glm::vec3(0,0,0);
	float dist = -1; //smallest dist
	float intsct = -1;
	  for(int i = 0; i<numberOfGeoms; i++)
			{
				if(geoms[i].type == 0)
				{
					dist = sphereIntersectionTest(geoms[i], Ri, interP, tmpnorm);
				}
				else if(geoms[i].type == 1)
				{
					dist = boxIntersectionTest(geoms[i], Ri, interP, tmpnorm);
				}

				//check if this geo is the closest
				if(epsilonCheck(dist, -1.0f) == false) // dist != -1
				{
					if(epsilonCheck(intsct, -1.0f) == true) //initalize intsct
					{
						intsct = dist;
						norm = tmpnorm;
						closestGeo = i;
					}else 	if(dist < intsct) // get smallest intsct
					{
						intsct = dist;
						norm = tmpnorm;
						closestGeo = i;
					}
				}
		  }
	  return intsct;
}

__host__ __device__ bool ShadowRayUnblocked(ray Rptolight, glm::vec3 lightPos, int numberOfGeoms, staticGeom* geoms, material* mats)
{
	int geoId = -1;
	glm::vec3 tmpnorm = glm::vec3(0,0,0);
	glm::vec3 tmpP = glm::vec3(0,0,0);
	glm::vec3 newOrigin = Rptolight.origin + glm::vec3(Rptolight.direction.x*0.001, Rptolight.direction.y*0.001, Rptolight.direction.z*0.001);

	ray Rp2light_new = Rptolight;
	Rp2light_new.origin = newOrigin;

	//float t = rayIntersect( Rp2light_new, geoms, numberOfGeoms, geoId, tmpP, tmpnorm);
	//if(mats[geoId].emittance > EPSILON) return false;  //hit the light
	//if(epsilonCheck(t, -1.0f) || abs(t) < 0.1f) return true;
	//else return false; 

	float t = -1; float intsct = -1;
	float t_light2Obj = glm::length(lightPos - newOrigin);
	
	for(int i = 0;i<numberOfGeoms;++i)
	{
		//only check geos
		if(epsilonCheck(mats[i].emittance, 0))
		{
			if(geoms[i].type == 0 ) //shpere
			{			
				t = sphereIntersectionTest(geoms[i], Rp2light_new, tmpP, tmpnorm);							
			}
			else if(geoms[i].type == 1) //cube
			{		
				t = boxIntersectionTest(geoms[i], Rp2light_new, tmpP, tmpnorm);		
			}
			if(epsilonCheck(t,-1.0f)==false && abs(t) > 0.01 && t < t_light2Obj) // t!=-1, t not hit itself, not over the maxium length
			{
				intsct = t;
			}
	}
	}
	if(epsilonCheck(intsct, -1)) 	return true;
	else return false;	
}

// add function: recursive raytracing
//__device__ void raytraceRecursive(ray Ri, int rayDepthMax, int rayDepthCur, glm::vec3& color, 
//	staticGeom* geoms, int numberOfGeoms, material* mats, glm::vec3 pos_cam, int index)
//{
//	if(rayDepthCur == 1)
//	{
//	// printf("rayDepthMax: %d; curr: %d  ", rayDepthMax, rayDepthCur);
//	//	printf("%d ", index);
//		if (index == 381210)
//		{
//			int test = 0;
//		}
//		if (index == 383624)
//		{
//			int test = 0;
//		}
//	}
//	if(rayDepthCur > rayDepthMax)
//	{
//		color = glm::vec3(BACK_R, BACK_G, BACK_B);
//		return;
//	}
//	////////////////RAY INTERSECT//////////////////////
//	  glm::vec3 interP = glm::vec3(0,0,0);
//	  glm::vec3 norm = glm::vec3(0,0,0);
//
//	   int closestGeo = -1;
//	
//	    //find the closest geometry 	  //intersection with geometries
//		float dist = rayIntersect(Ri, geoms, numberOfGeoms, closestGeo, interP, norm);		
//
//		 /////////////////////COLOR INITIAL////////////////////
//		glm::vec3 colortmp = glm::vec3(0,0,0); 
//		glm::vec3 color_obj = mats[closestGeo].color;
//		glm::vec3 color_light = glm::vec3(0,0,0);
//		glm::vec3 color_reflect = glm::vec3(1,1,1);
//		glm::vec3 color_refract = glm::vec3(1,1,1);
//		glm::vec3 color_ambient = glm::vec3(AMBIENT_R, AMBIENT_G, AMBIENT_B);
//		color_ambient *= K_AMBIENT;		
//		
//		glm::vec3 color_diff = glm::vec3(0,0,0);
//		glm::vec3 color_spec = glm::vec3(0,0,0);
//
//		glm::vec3 color_sum = glm::vec3(0,0,0);
//		
//		 //if didn't hit anything, background color
//			if(epsilonCheck(dist, -1.0f) == true) //intsct == -1.0
//			{
//				// do nothing, still background color
//				colortmp = glm::vec3(BACK_R, BACK_G, BACK_B);
//			}
//			 //hit something
//			else 
//			{
//				////////////////HIT LIGHT////////////////
//				if (mats[closestGeo].emittance>0)
//				{				
//					colortmp = color_obj;	
//				}			
//			
//			  ////////////////HIT GEO//////////////////
//			  if(epsilonCheck(mats[closestGeo].emittance, 0))
//			  {	
//			  //////////////REFLECTION/////////////////////
//				  ray Rr = calcReflectRay(Ri, norm, interP); //reflect ray
//				  
//				  // if it is reflective
//				 if(mats[closestGeo].hasReflective > EPSILON)
//				  {
//			//		  printf("isreflective");
//					raytraceRecursive(Rr, rayDepthMax, rayDepthCur+1, color_reflect, geoms, numberOfGeoms,mats, pos_cam, index);
//					colortmp = mats[closestGeo].hasReflective * color_reflect;
//				   } 
//				colortmp += color_ambient * color_obj;
//
//			  /////////////Light RAYTRACING & SOFT SHADOW///////////////////
//		for(int gi = 0; gi<numberOfGeoms; gi++)
//			{
//				if(mats[gi].emittance > 0) //is Light
//				{
//					color_light = mats[gi].color;
//				for(int si = 0; si < SAMPLES_SOFT_SHADOW; si++)
//				{
//					//Random light position;
//					glm::vec3 pos_light = getRandomPointOnCube(geoms[gi], hash(si));
//					//calculate light reflect ray
//					ray Rlighti;
//					Rlighti.origin = pos_light;
//					Rlighti.direction = glm::normalize(interP - pos_light);
//					ray Rlightr = calcReflectRay(Rlighti, norm, interP);
//			
//					glm::vec3 localColor = glm::vec3(0,0,0);
//
//					ray Rptolight;
//					Rptolight.origin = interP;
//					Rptolight.direction = glm::normalize(pos_light - interP);
//
//					if(ShadowRayUnblocked(Rptolight, pos_light, numberOfGeoms, geoms, mats))
//					{
//						/////////NOT IN SHADOW//////////			
//					//	printf("not blocked");
//						if(glm::dot(norm, Rptolight.direction) > 0)
//						{
//							color_diff = color_obj * glm::dot(norm, Rptolight.direction) * float(K_DIFFUSE) * color_light;
//						}			
//						localColor += color_diff;
//
//						ray Rptocam;
//						Rptocam.origin = interP;
//						Rptocam.direction = glm::normalize(pos_cam - interP);
//						
//						if(glm::dot(Rlightr.direction, Rptocam.direction) > 0 && mats[closestGeo].specularExponent > 0)
//						{									
//							float temp  = pow((float)glm::dot(Rlightr.direction, Rptocam.direction), (float)mats[closestGeo].specularExponent);
//							color_spec = color_light * float(K_SPEC) * temp;// * mats[lightIndex[j]].emittance;
//						}			
//						localColor += color_spec;	
//					}
//					color_sum += localColor;
//				}
//				color_sum *= mats[gi].emittance;
//				color_sum /= (float)SAMPLES_SOFT_SHADOW;		
//		  }
//			} 
//	///////////shadow end///////
//			colortmp += color_sum;
//			  }
//			}
//			color = colortmp;
//	 }
	  
//	  test++;
	 // printf("color , %f, %f, %f", color[0], color[1], color[2]);
//	  printf("test: %d, index: %d", test, index);


// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void pathtraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepthMax, int currDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mats, ray* rayPool, glm::vec3* colorAccumulator, int numberOfRays){

  /*int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);*/
  //1D:
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(index < numberOfRays)
  {	  
	  //colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	   ray Ri = rayPool[index];
	   Ri.origin += 0.001f * Ri.direction;
	   int index2D = Ri.index2D;

//	  glm::vec3 rayColorSum = glm::vec3(0,0,0);
//	  for (int k = 0; k < SAMPLES_PER_PIXEL; k++)
//	  {
		  // glm::vec3 rayColor = glm::vec3(0,0,0);		   
		   //raytraceRecursive(Ri, rayDepth, 0, rayColor, geoms,numberOfGeoms,mats, cam.position, index);
		 
			if(currDepth > rayDepthMax)
			{
				colors[index2D] = (colors[index2D]*(time-1) + colorAccumulator[index2D])/time;
				//colors[index2D] = colors[index2D]*0.9f + colorAccumulator[index2D]*0.1f;
				return;
			}

			////////////////RAY INTERSECT//////////////////////
		  glm::vec3 interP = glm::vec3(0,0,0);
		  glm::vec3 norm = glm::vec3(0,0,0);
		  int closestGeo = -1;	
			//find the closest geometry 	  //intersection with geometries
		  float dist = rayIntersect(Ri, geoms, numberOfGeoms, closestGeo, interP, norm);		
	//	  rayPool[index].interP = interP;
	//	  rayPool[index].norm = norm;

		 //if didn't hit anything, background color
			if(epsilonCheck(dist, -1.0f) == true) //intsct == -1.0
			{	// do nothing, still background color
			//	colortmp = glm::vec3(BACK_R, BACK_G, BACK_B);
			//	colors[index2D] = (colors[index2D]*(time-1)+ Ri.color)/time;
				colorAccumulator[index2D] *= Ri.k_reflect * glm::vec3(BACK_R, BACK_G, BACK_B);
				//kill the ray
				rayPool[index].isDead = true;
				colors[index2D] = (colors[index2D]*(time-1)+colorAccumulator[index2D])/time;
				//colors[index2D] = (colors[index2D]*0.9f+colorAccumulator[index2D]*0.1f);
				return;
			}
			 //hit something
			else 
			{
				////////////////HIT LIGHT////////////////
				if (mats[closestGeo].emittance>0)
				{				
				//	printf("%f ", Ri.k_spec);
					colorAccumulator[index2D] *= (1+Ri.k_spec) * mats[closestGeo].emittance * mats[closestGeo].color; //add specular
				//	colorAccumulator[index2D] += mats[closestGeo].color;
			
					rayPool[index].isDead = true; //kill the ray
					//	colors[index2D] = (colors[index2D]*0.9f+colorAccumulator[index2D]*0.1f);
					colors[index2D] = (colors[index2D]*(time-1)+colorAccumulator[index2D])/time;
					return;
									
					}
			  ////////////////HIT GEO//////////////////
			  else if(epsilonCheck(mats[closestGeo].emittance, 0))
			  {	
				  //random number generation
				  thrust::default_random_engine rng(hash(time*(currDepth+1))*hash(index2D));
				  thrust::uniform_real_distribution<float> u01(0,1);	
				  float r =(float) u01(rng);


				  ray Rr;
				  Rr.origin = interP;	
				 if(mats[closestGeo].hasReflective > EPSILON)
				  {
					  //////////////REFLECTION/////////////////////
	
				if(r < mats[closestGeo].hasReflective)
				{
					//reflect
					Rr = calcReflectRay(Ri, norm, interP);
					Rr.k_reflect = mats[closestGeo].hasReflective;
				}
				else
				{
					//diffuse
					//get random direction over hemisphere
					Rr.direction = calculateRandomDirectionInHemisphere(norm, (float)u01(rng), (float)u01(rng)); 
					Rr.direction = glm::normalize(Rr.direction);
					//colors[index] += mats[nearestObjIndex].color * mats[nearestObjIndex].hasReflective;
					Rr.k_reflect = 1 - mats[closestGeo].hasReflective;
				}
				   } 
				 else if(mats[closestGeo].hasRefractive > EPSILON)
				 { //////////////Refraction ////////////////
					if(r< mats[closestGeo].hasRefractive)
					{
					 Rr = calcRefractRay(Ri, norm, interP, mats[closestGeo].indexOfRefraction);
					}else{
						Rr = calcReflectRay(Ri, norm, interP);
					}
				 }
				 else
				 { /////diffuse//////
					 Rr.direction = calculateRandomDirectionInHemisphere(norm, u01(rng), u01(rng)); 
					 Rr.direction = glm::normalize(Rr.direction);
					 colorAccumulator[index2D] *= mats[closestGeo].color;	
			//		 colorAccumulator[index2D] *= K_DIFFUSE;
				 }
				 float avgTemp = 0;
				 for(int gi = 0; gi<numberOfGeoms; gi++)
				{
					if(mats[gi].emittance > 0) //is Light
					{
						
						for(int si = 0; si < SAMPLES_SOFT_SHADOW; si++)
						{
						//Random light position;
						glm::vec3 pos_light = getRandomPointOnCube(geoms[gi], hash(si));
				 	//	glm::vec3 pos_light = geoms[gi].translation;
						ray Rlighti;
						Rlighti.origin = pos_light;
						Rlighti.direction = glm::normalize(interP - pos_light);
						ray Rlightr = calcReflectRay(Rlighti, norm, interP);
			
						glm::vec3 localColor = glm::vec3(0,0,0);

						ray Rptolight;
						Rptolight.origin = interP;
						Rptolight.direction = glm::normalize(pos_light - interP);
						
						if(ShadowRayUnblocked(Rptolight, pos_light, numberOfGeoms, geoms, mats))
						{
		//					printf("not blocked");
							ray Rptocam;
							Rptocam.origin = interP;
							Rptocam.direction = glm::normalize(cam.position - interP);
							if(glm::dot(Rlightr.direction, Rptocam.direction) > 0 && mats[closestGeo].specularExponent > 0)
							{	
							//	printf("specular");
								float temp  = pow((float)glm::dot(Rlightr.direction, Rptocam.direction), (float)mats[closestGeo].specularExponent);
								//Rr.k_spec = temp;// * mats[lightIndex[j]].emittance;
								avgTemp += temp;
							}	
						}
						}
						avgTemp /= float(SAMPLES_SOFT_SHADOW);
						Rr.k_spec = avgTemp;
					}
				 }
			//	colortmp += color_ambient * color_obj;
			rayPool[index] = Rr;
			Rr.origin += 0.001f * Rr.direction;
			
	
	//		colortmp += color_sum;
			  }
			}
		//   rayColorSum += rayColor;
//	   }
//	  colors[index2D] = rayColorSum / float(SAMPLES_PER_PIXEL);
//		   colors[index2D] = colortmp;
//		colors[index2D] = (colors[index2D]*(time-1)+colorAccumulator[index2D])/time;
  }
}

__global__ void rayInitial(ray* rayPool, glm::vec3* colorAccumulator, glm::vec2 resolution, float time, cameraData cam)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * resolution.x);	
	ray Ri = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, index); 
	rayPool[index] = Ri;
//	rayPool[index].color = glm::vec3(0,0,0);
	rayPool[index].isDead = false;
	rayPool[index].index2D = index;
	colorAccumulator[index] = glm::vec3(1,1,1);
}
// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = DEPTH_TRACE; //determines how many bounces the raytracer traces
  
  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

  // package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  material* matList = new material[numberOfGeoms];
  
  for(int i=0; i<numberOfGeoms; i++){
	 //geom
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;

	//material
	matList[i] = materials[geoms[i].materialid];
	
  }

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  material* cudamats = NULL;
  cudaMalloc((void**)&cudamats, numberOfGeoms*sizeof(material));
  cudaMemcpy( cudamats, matList, numberOfGeoms*sizeof(material), cudaMemcpyHostToDevice);
  
  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

   /////////// STEP 1: Construct pool of rays ////////////////////
   ////////////STEP 2: Construct accumulator image, initialize black /////////////////
  // package ray
  ray* rayPool = NULL;
  glm::vec3* colorAccumulator = NULL;
  int numberOfRays = (int)cam.resolution.x * (int)cam.resolution.y;
  cudaMalloc((void**)&rayPool, numberOfRays*sizeof(ray)); 
  cudaMalloc((void**)&colorAccumulator, numberOfRays*sizeof(glm::vec3)); 

   rayInitial<<<fullBlocksPerGrid, threadsPerBlock>>>(rayPool, colorAccumulator, renderCam->resolution, (float)iterations, cam);

   
  // send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  ////////////STEP 3: Launch a kernel to trace ONE bounce ////////////////////////////
  int threadsPerBlock_1D_origin = 64;
  int blocksPerGrid_1D_origin = (int)ceil(float(numberOfRays)/(float)threadsPerBlock_1D_origin);
  int threadsPerBlock_1D = 64;
  int blocksPerGrid_1D = (int)ceil(float(numberOfRays)/(float)threadsPerBlock_1D);
  for (int currDepth = 0; currDepth < traceDepth; currDepth++)
  {
	  if(numberOfRays == 0) break;
	  blocksPerGrid_1D = (int)ceil((float)numberOfRays/(float)threadsPerBlock_1D);
	  pathtraceRay<<<blocksPerGrid_1D, threadsPerBlock_1D>>>(renderCam->resolution, (float)iterations, cam, traceDepth, currDepth, 
		  cudaimage, cudageoms, numberOfGeoms, cudamats, rayPool, colorAccumulator, numberOfRays);
	  /////////////// Stream Compaction ///////////////////////
	  thrust::device_ptr<ray> p_start = thrust::device_pointer_cast(rayPool);
	  thrust::device_ptr<ray> p_end = thrust::remove_if(p_start, p_start+numberOfRays, is_ray_dead());
	  numberOfRays = (int)( p_end - p_start);
  }
 // addShadowColor<<<blocksPerGrid_1D_origin, threadsPerBlock_1D_origin>>>((float)iterations, cam, cudaimage, cudageoms, numberOfGeoms, cudamats, rayPool, colorAccumulator, numberOfRays);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  
  cudaFree (rayPool);
  cudaFree (colorAccumulator);

  delete[] geomList;
  delete[] matList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
