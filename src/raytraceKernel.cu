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

struct is_dead{  
	__host__ __device__  bool operator()(const ray& r)  
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

// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

__host__ __device__ glm::vec2 generateRandomNumberAntiAliasing(float seed, float x, float y, float d){
  thrust::default_random_engine rng(hash(seed));
  thrust::uniform_real_distribution<float> u01(0,1);
  float xOffset = (float)u01(rng) * 2 * d;
  float yOffset = (float)u01(rng) * 2 * d;
  return glm::vec2(x - d + xOffset, y - d + yOffset);
}

// TODO: IMPLEMENT THIS FUNCTION
// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
	
	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);
	glm::vec3 M = eye + view;

	glm::vec3 V = (float)tan(fov.y/(float)180.0*PI) * glm::length(view) * glm::normalize(B);
	glm::vec3 H = (float)tan(fov.x/(float)180.0*PI) * glm::length(view) * glm::normalize(A);
	
	//choose point on the image plane based on pixel location
	float Sh = float(x)/float(resolution.x-1);
	float Sv = 1- float(y)/float(resolution.y-1);   //invert y coordinate

	//choose random point on image plane
	/*thrust::default_random_engine rng(hash(index*time));
	thrust::uniform_real_distribution<float> u01(0,1);
	float Sh = (float) u01(rng);
	float Sv = (float) u01(rng);*/

	//sreen coordinates to world coordinates
	glm::vec3 point = M + (float)(2*Sh-1)*H + (float)(2*Sv-1)*V;

	//initial cast of ray
	ray r;
	r.direction = glm::normalize(point - eye);
	r.origin = eye;
	r.color = glm::vec3(1,1,1);
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
 __host__ __device__ int findHitGeo(ray r, glm::vec3& intersect, glm::vec3& normal, staticGeom* geoms, int numberOfGeoms, triangle * cudatris){
	
	 if(r.isDead)
		 return -1;
	 float distMin = -2, dist = -1;
	 glm::vec3 tempIntersect(0.0f);
	 glm::vec3 tempNormal(0.0f);
	 int ID = -1;
	//geometry and ray intersect tesing
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

		//overwrite minimum distance if needed
		if( (distMin < 0 && dist > -0.5f ) || ( distMin > -1 && dist < distMin && dist > -0.5f ) ){
			distMin = dist;   //update minimum dist
			ID = g;            //update ID of geometry
			intersect = tempIntersect;  //update intersct point
			normal = tempNormal;   //update normal
		}
	}
	return ID;
}

 //return true if there is direct lighting
 __host__ __device__ bool ShadowRayTest(ray sr, staticGeom* geoms, int numberOfGeoms, material* materials, triangle * cudatris){
	glm::vec3 intersPoint(0.0f);
	glm::vec3 intersNormal(0.0f);

	//printf("shadow ray: [%f,%f,%f], [%f,%f,%f]\n", sr.origin.x,sr.origin.y,sr.origin.z,sr.direction.x,sr.direction.y,sr.direction.z);
	int geoID = findHitGeo(sr, intersPoint, intersNormal, geoms, numberOfGeoms, cudatris); 
	if( geoID>-1 && geoms[geoID].materialid >= 0 &&materials[geoms[geoID].materialid].emittance > 0){   //hit light soource
		return true;
	}
	else{
		return false;
	}
	
}

 // get shaw ray to a random chosen light, modify the shadowray, return ID of chosen light
 __device__ __host__ int getRandomShadowRayDirection(float seed, glm::vec3& theIntersect, int* lights, int numOfLights, 
	 staticGeom* geoms, ray& shadowRay, float& rayLength, glm::vec3 lightNormal, float lightArea){
	
	 // ******************  choose light first ******************************** //
	int chosenLight = lights[0];   //only one light
	if( numOfLights > 1){   //more than 1 light
		thrust::default_random_engine rng(hash(seed));
		thrust::uniform_real_distribution<float> u01(0,1); 
		chosenLight = lights[(int)((float)u01(rng) * numOfLights)];   //randomly choose a light to sample
	}

	// ******************  find a point on light ******************************** //
	glm::vec3 Lnormal(0.0f);   //light normal
	float Larea;            //light area
	glm::vec3 Plight;       //random point on light
	if( geoms[chosenLight].type == CUBE ){
		//Plight = getRandomPointOnCube( geoms[chosenLight], seed);
		Plight = getRandomPointOnCube( geoms[chosenLight], seed, Lnormal, Larea);
	}
	else if( geoms[chosenLight].type == SPHERE ){
		Plight = getRandomPointOnSphere( geoms[chosenLight], seed, Lnormal, Larea);
	}
	
	// ******************  shadow ray test ******************************** // 
	shadowRay.direction = glm::normalize(Plight - theIntersect);   //from intersect to light
	shadowRay.origin = theIntersect + (float)EPSILON * shadowRay.direction;
	rayLength = glm::length(Plight - theIntersect);
	return chosenLight;
 }


 __device__ __host__ glm::vec3 getTextureColor(glm::vec3* cudatexs, tex* cudatexIDs, glm::vec3 &intersect, staticGeom& geom){
	 tex theTex = cudatexIDs[abs(geom.materialid)-1];
	// printf("theTex: h=%d, w=%d, start=%d\n", theTex.h, theTex.w, theTex.start);
	 glm::vec3 p = multiplyMV(geom.inverseTransform, glm::vec4(intersect,1.0f));
	 float u,v;
	 if(geom.type == CUBE){
		 // printf("p.x=%f, p.y = %f, p.z=%f, intersect.x=%f, intersect.y=%f, intersect.z=%f\n",p.x, p.y, p.z, intersect.x, intersect.y, intersect.z);
		 if(std::abs(0.5f - abs(p.x)) < EPSILON){   //left or right face
			 u = p.z + 0.5f;
			 v = p.y + 0.5f;
		 }else if(std::abs(0.5f - abs(p.y)) < EPSILON){  // top or bottom face
			 u = p.x + 0.5f;
			 v = p.z + 0.5f;
		 }else if(std::abs(0.5f - abs(p.z)) < EPSILON){  //front or back face
			 u = p.x + 0.5f;
			 v = p.y + 0.5f; v = 1.0f - v;
		 }
	 }else if(geom.type == SPHERE){
		 glm::vec3 d = glm::vec3(0.0)- glm::vec3(p.x, p.y, p.z);
		// printf("p.x=%f, p.y = %f, p.z=%f, intersect.x=%f, intersect.y=%f, intersect.z=%f\n",p.x, p.y, p.z, intersect.x, intersect.y, intersect.z);
		 u = 0.5f + atan2(d.z, d.x) * 0.5f / PI;
		 v = 0.5f - asin(d.y) / PI;	
	 }
	int i,j,idx = -1;
	i = u * (float)theTex.w;
	j = v * (float)theTex.h;
	idx = i + j * theTex.w + theTex.start;
	//  printf("x=%f, z=%f, u=%f, v=%f, idx = %d\n",intersect.x * 0.2f,intersect.z * 0.2f, u, v, idx);
	if( idx <= theTex.w * theTex.h + theTex.start && idx>=theTex.start ){
		glm::vec3 color(cudatexs[idx].r/255.0, cudatexs[idx].g/255.0, cudatexs[idx].b/255.0);
		return color;
	}	
	return intersect;
}

 //calculates the direct lighting for a certain hit point and modify color of that hit
__device__ __host__ void directLighting(float seed, glm::vec3& theColor, glm::vec3& theIntersect, glm::vec3& theNormal, int geoID, 
	int* lights, int numOfLights, material* cudamats, staticGeom* geoms, int numOfGeoms, triangle * cudatris){
	ray shadowRay;
	float rayLen,lightArea;
	glm::vec3 lightNormal;
	int lightID = getRandomShadowRayDirection(seed, theIntersect, lights, numOfLights, geoms, shadowRay, rayLen, lightNormal, lightArea);

	// ****************** shading if direct illumination ****************** //
	if(geoms[geoID].materialid >= 0 ){
		material curMat = cudamats[geoms[geoID].materialid];  //material of the hit goemetry
		if(ShadowRayTest(shadowRay, geoms, numOfGeoms, cudamats, cudatris)  ){
			float cosTerm = glm::clamp( glm::dot( theNormal, shadowRay.direction ), 0.0f, 1.0f);  //proportion of facing light
			float cosTerm2 = glm::clamp( glm::dot( lightNormal, -shadowRay.direction ), 0.0f, 1.0f);  //proportion of incoming light
			float areaSampling =  lightArea / (float) pow( rayLen, 2.0f) ;   // dA/r^2
     		theColor += cudamats[lightID].emittance * curMat.color * cosTerm * cosTerm2 * areaSampling ;
		}
	}
	//don't kill any ray in direct lighting calculation
}


// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(ray* rays, float time, int rayDepth, int numOfRays, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* cudamats, int* lights, int numOfLights, 
							cameraData cam, triangle* cudatris, glm::vec3* cudatexs, tex* cudatexIDs){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
 // int index = x * blockDim.y + y;
 //  int index = x + (y * resolution.x);
   int index = x + (int)ceil(sqrt((float)numOfRays))* y;
 //  printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);

   if( index < numOfRays ){
	   ray r = rays[index];

	   glm::vec3 Pintersect(0.0f);
	   glm::vec3 Pnormal(0.0f);
		int hitGeoID = findHitGeo(r, Pintersect, Pnormal, geoms, numberOfGeoms, cudatris);

		if(hitGeoID!=-1){
			material curMat;
			if(geoms[hitGeoID].materialid >= 0)
				curMat = cudamats[geoms[hitGeoID].materialid];
			if( curMat.emittance > 0 ){   //end when hit light source
				if(glm::length(r.color)>0.6f){
			//	printf("ray color:[%f, %f, %f]", r.color.r,r.color.g,r.color.b);
				}
				
				colors[r.pixel] += r.color * curMat.color * curMat.emittance; 
				r.isDead = true;
			}
			else{

			//int mode = calculateBSDF(r, Pintersect, Pnormal, color, uncolor, cudamats[matID], hash(index*time));
				float seed = (float)index * (float)time * ( (float)rayDepth + 1.0f );
				if(curMat.hasReflective > 0 || curMat.hasRefractive > 0){
					//------------------------------- calculate Fresnel reflectance and transmittance --------------------------//
					Fresnel F;
					float reflectance;
					glm::vec3 reflectDir, transmitDir;
					if(glm::dot(r.direction,Pnormal)<0){   //ray is outside
						F = calculateFresnel(Pnormal,r.direction,1.0f, curMat.indexOfRefraction);
						reflectDir = calculateReflectionDirection(Pnormal, r.direction);
						transmitDir = calculateTransmissionDirection(Pnormal, r.direction,1.0f, curMat.indexOfRefraction);
					}
					else{  //ray is inside
						F = calculateFresnel(-Pnormal,r.direction, curMat.indexOfRefraction,1.0f);
						reflectDir = calculateReflectionDirection(-Pnormal, r.direction);
						transmitDir = calculateTransmissionDirection(-Pnormal, r.direction, curMat.indexOfRefraction, 1.0f);
					}
					//--------------------------------------------------------------------------------------------------------//

					//----------------------- choosing between reflection or refraction or both -------------------------------//
					if( curMat.hasRefractive  > 0 && curMat.hasReflective > 0){
						thrust::default_random_engine rng( hash( seed ) );
						thrust::uniform_real_distribution<float> u01(0,1);

						if((float) u01(rng) < F.reflectionCoefficient ){   //reflected
							r.direction = reflectDir;
							r.origin = Pintersect + (float)EPSILON * r.direction;
							//colors[r.pixel] += glm::abs(r.direction);
							if(glm::length(curMat.color)>0)
								r.color *= curMat.color ;
								//r.color *= curMat.color * F.reflectionCoefficient;

						}
						else{   //transmitted
							r.direction = transmitDir;
							r.origin = Pintersect + (float)EPSILON  * r.direction;
							//colors[r.pixel] += glm::abs(r.direction);
							if(glm::length(curMat.color)>0)
								r.color *= curMat.color ;
						
						}
					}
					else if(curMat.hasReflective > 0){   //only reflection
						r.direction = reflectDir;
						r.origin = Pintersect + (float)EPSILON * r.direction;
						//colors[r.pixel] += glm::abs(r.direction);
						if(glm::length(curMat.color)>0)
							r.color *= curMat.color ;
					}
					else if (curMat.hasRefractive  > 0){  //only refraction
						r.direction = transmitDir;
						r.origin = Pintersect + (float)EPSILON  * r.direction;
						//colors[r.pixel] += glm::abs(r.direction);
						if(glm::length(curMat.color)>0)
							r.color *= curMat.color ;
					}
				}
				//--------------------------------------------------------------------------------------------------------//
				else if (curMat.hasScatter>0){
					
				}
				else{    //diffuse rays
					thrust::default_random_engine rng( hash( seed ) );
					thrust::uniform_real_distribution<float> u01(0,1);
					if((float) u01(rng) < 0.01f ){  //proportion to calculate direct lighting
						directLighting(seed,r.color,Pintersect,Pnormal,hitGeoID,lights,numOfLights, cudamats,geoms, numberOfGeoms, cudatris);
					}
					else{   //proportion to calculate indirect lighting
						//cos weighted importance sampling
						r.direction = calculateCosWeightedRandomDirInHemisphere(Pnormal, (float) u01(rng), (float) u01(rng));
						r.origin = Pintersect + (float)EPSILON * r.direction ;
						float diffuseTerm = glm::clamp( glm::dot( Pnormal,r.direction ), 0.0f, 1.0f);
						if(geoms[hitGeoID].materialid < 0){  //texture
							r.color *=  diffuseTerm * getTextureColor(cudatexs, cudatexIDs, Pintersect, geoms[hitGeoID]);	
						}else{
							r.color *=  diffuseTerm * curMat.color;	
						}
					}
				}

				//----------------------------------------- Other Effects ----------------------------------------------//
				if(curMat.specularExponent > 0 ){   //specularity  & glossiness
					thrust::default_random_engine rng( hash( seed ) );
					thrust::uniform_real_distribution<float> u01(0,1);
					ray shadowRay;
					float rayLen,lightArea;
					glm::vec3 lightNormal;
					int lightID = getRandomShadowRayDirection(seed, Pintersect, lights, numOfLights, geoms, shadowRay, rayLen, lightNormal, lightArea);
					glm::vec3 viewDir = glm::normalize( cam.position - Pintersect );
					glm::vec3 halfVector = glm::normalize(shadowRay.direction + viewDir);  //H=(L+V)/length(L+V)
					float D = glm::clamp( glm::dot( Pnormal,halfVector), 0.0f, 1.0f);  
					float specularTerm = pow(D, curMat.specularExponent);   //perfect specular means normal vector = half vector
					r.color *= specularTerm * curMat.color;
				}
			
			}
		
		}
		else{  //hit nothing
			r.isDead = true;
		}
		rays[index] = r;
		
		
   }
}



// establish parrallel ray pool
__global__ void initialRayPool(ray * rayPool, cameraData cam, float iterations,glm::vec3 *colors, staticGeom* geoms, int numberOfGeoms, 
	material* cudamats, int * lightIDs, int numberOfLights, triangle * cudatris){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * cam.resolution.x);
	ray r = rayPool[index];

	if( x<= cam.resolution.x && y <= cam.resolution.y ){
		if(ANTI_ALIASING){
			glm::vec2 jitter = generateRandomNumberAntiAliasing((float)index * iterations, x, y, 0.5f);  //anti-alizsing
			r = raycastFromCameraKernel( cam.resolution, iterations, jitter.x, jitter.y, cam.position, cam.view, cam.up, cam.fov );
		}
		else{
			r = raycastFromCameraKernel( cam.resolution, iterations, x, y, cam.position, cam.view, cam.up, cam.fov );
		}
		r.pixel = index;  //mark ray with pixel indexing, after compaction, (r.pixel) will represent correct pixel location

		if(DEPTH_OF_FIELD){
			glm::vec3 focalPoint = r.origin + r.direction * cam.focalLength / glm::dot(cam.view, r.direction);   //L = f/cos(theta)
			thrust::default_random_engine rng(hash((float)index*iterations));
			thrust::uniform_real_distribution<float> u01(0,1);
			float theta = 2.0f * PI * u01(rng);
			float radius = u01(rng) * cam.aperture;
			glm::vec3 eyeOffset(cos(theta)*radius, sin(theta)*radius, 0);
			glm::vec3 newEyePoint = cam.position + eyeOffset;  //offseted cam eye location
			r.origin = newEyePoint;
			r.direction = glm::normalize(focalPoint - newEyePoint);
		}

		glm::vec3 Pintersect(0.0f);
		glm::vec3 Pnormal(0.0f);
		int geoID = findHitGeo(r, Pintersect, Pnormal, geoms, numberOfGeoms, cudatris);
		if( geoID > -1){
				// cast shadow ray towards lights and calculate direct lighting
				directLighting((float)index*iterations, colors[index], Pintersect, Pnormal,geoID, lightIDs, 
					numberOfLights, cudamats, geoms, numberOfGeoms, cudatris);

		}
		rayPool[index] = r;
	}
}


// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, 
	material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, 
	std::vector<glm::vec3> &textures, std::vector<tex> &textureIDs){
	
	//frame: current frame number
	//iterations: current iteration of rendering <  (cam.iterations)
	if(iterations == 572 || iterations == 46 || iterations == 7){
		printf("problem");
	}

	// send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
 // triangle* triList;
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
		newStaticGeom.bBoxMax = geoms[i].bBoxMax;  //bBox is in local coordinates, dont change over frames.
		newStaticGeom.bBoxMin = geoms[i].bBoxMin;
		newStaticGeom.numOfTris = geoms[i].numOfTris;
		cudaMalloc((void**)&cudatris, geoms[meshID].numOfTris*sizeof(triangle));
		cudaMemcpy( cudatris, geoms[meshID].tris, geoms[meshID].numOfTris *sizeof(triangle), cudaMemcpyHostToDevice);
		//printf("num of tris: %d\n",geoms[meshID].numOfTris);
		/*if(iterations == 3){
			for (int j=0; j<geoms[meshID].numOfTris; j++){
				printf("geoms triangle %d: \n [%.2f, %.2f, %.2f]\n [%.2f, %.2f, %.2f]\n [%.2f, %.2f, %.2f]\n", j,
					geoms[meshID].tris[j].p1.x, geoms[meshID].tris[j].p1.y, geoms[meshID].tris[j].p1.z,
					geoms[meshID].tris[j].p2.x, geoms[meshID].tris[j].p2.y, geoms[meshID].tris[j].p2.z,
					geoms[meshID].tris[j].p3.x, geoms[meshID].tris[j].p3.y, geoms[meshID].tris[j].p3.z);
			}
		}*/
	/*	newStaticGeom.tris = cudatris;
		if(iterations == 3){
			for (int j=0; j<geoms[meshID].numOfTris; j++){
				printf("StaticGeom triangle %d: \n [%.2f, %.2f, %.2f]\n [%.2f, %.2f, %.2f]\n [%.2f, %.2f, %.2f]\n", j,
					newStaticGeom.tris[j].p1.x, newStaticGeom.tris[j].p1.y, newStaticGeom.tris[j].p1.z,
					newStaticGeom.tris[j].p2.x, newStaticGeom.tris[j].p2.y, newStaticGeom.tris[j].p2.z,
					newStaticGeom.tris[j].p3.x, newStaticGeom.tris[j].p3.y, newStaticGeom.tris[j].p3.z);
			}
		}*/
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
		if( geoms[i].materialid >= 0 && materials[geoms[i].materialid].emittance > 0){
			numberOfLights ++ ;
		}
	}

	int *lightIDs = new int[numberOfLights];
	int k = 0;
	for(int i = 0; i < numberOfGeoms; ++i){
		if( geoms[i].materialid >= 0 && materials[geoms[i].materialid].emittance > 0){
			lightIDs[k] = i;
			k++;
		}
	}
	int* cudalightIDs = NULL;
	cudaMalloc((void**)&cudalightIDs, numberOfLights*sizeof(int));
	cudaMemcpy( cudalightIDs, lightIDs, numberOfLights*sizeof(int), cudaMemcpyHostToDevice);

	//set up textures
	int numberOfPixels = textures.size();
	glm::vec3 *texs = new glm::vec3[numberOfPixels];
	for(int i = 0; i < numberOfPixels; ++i){
		texs[i] = textures[i];
	}
	glm::vec3 *cudatexs = NULL;
	cudaMalloc((void**)&cudatexs,numberOfPixels * sizeof(glm::vec3));
	cudaMemcpy( cudatexs, texs,  numberOfPixels * sizeof(glm::vec3),  cudaMemcpyHostToDevice);

	//set up textures id
	int numOfTextures = textureIDs.size();
	tex *texIDs = new tex[numOfTextures];
	for(int i = 0; i < numOfTextures; ++i){
		texIDs[i] = textureIDs[i];
	}
	tex *cudatexIDs = NULL;
	cudaMalloc((void**)&cudatexIDs, numOfTextures * sizeof(tex));
	cudaMemcpy( cudatexIDs, texIDs, numOfTextures* sizeof(tex),  cudaMemcpyHostToDevice);

	//set up ray pool on device
	ray* cudarays = NULL;
	int numOfRays = cam.resolution.x * cam.resolution.y;
	cudaMalloc((void**)&cudarays, numOfRays*sizeof(ray));


	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));

	initialRayPool<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays, cam, (float)iterations, cudaimage,cudageoms, numberOfGeoms, cudamats, cudalightIDs, numberOfLights, cudatris);
 
	for(int k=0; k<MAX_DEPTH && numOfRays>0; k++){
		if(STREAM_COMPACT){
			thrust::device_ptr<ray> Start = thrust::device_pointer_cast(cudarays);  //coverts cuda pointer to thrust pointer
			thrust::device_ptr<ray> End = thrust::remove_if(Start, Start + numOfRays, is_dead());
			numOfRays = thrust::distance(Start, End);	
		}
		//xBlocks * yBlocks = numOfRays / (tileSize*tileSize)
		int xBlocks = (int) ceil( sqrt((float)numOfRays)/(float)(tileSize) );
		int yBlocks = (int) ceil( sqrt((float)numOfRays)/(float)(tileSize) );
		dim3 newBlocksPerGrid(xBlocks,yBlocks);
		
		raytraceRay<<<newBlocksPerGrid, threadsPerBlock>>>(cudarays, (float)iterations, k, (int)numOfRays, cudaimage, cudageoms, numberOfGeoms, cudamats, cudalightIDs, numberOfLights, cam, cudatris, cudatexs, cudatexIDs);

	}

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage,(float)iterations);
 

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamats );
  cudaFree( cudarays );
  cudaFree( cudatexs );
  cudaFree( cudatexIDs );
  cudaFree( cudalightIDs );
  if(meshID >-1 ){
	cudaFree( cudatris );
  }

  delete geomList;
  //delete matList;
  delete lightIDs;
  delete texs;
  delete texIDs;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
