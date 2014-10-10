// CIS565 CUDA Pathtracer: A parallel pathtracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
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
#include "pathtraceKernel.h"
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

// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye; // Ray origin is at the eye here

  //The center of the image plane is offset by the gaze from the eye
  glm::vec3 center = eye + view;

  //The right direction is obtained by crossing the gaze direction with up
  glm::vec3 right = glm::normalize(glm::cross(view, up));
  //The down direction is obtained with the negative of the up direction
  glm::vec3 down = -glm::normalize(up);

  //Convert the field of view to radians (IM CONFUSED, WHY DOES IT SEEM LIKE THE INPUT IS ALREADY RADIANS??)
  //fov.x = fov.x * 180.0 / PI;
  //fov.y = fov.y * 180.0 / PI;

  // Setup a random distribution so that the ray doesn't just go through the center of each pixel
  thrust::default_random_engine rng(hash((time+1)*x*y));
  thrust::uniform_real_distribution<float> u01(-1.0,1.0);

  //Calculate the perpindicular vector component magnitudes from the fov, resolution, and x/y values
  glm::vec2 offset;
  offset.x = (float)x - resolution.x/2.0f + (float)u01(rng);
  offset.y = (float)y - resolution.y/2.0f + (float)u01(rng);
  glm::vec2 mag;
  mag.x = glm::length(view) * tan(fov.x) * offset.x / resolution.x;
  mag.y = glm::length(view) * tan(fov.y) * offset.y / resolution.y;

  //Calculate the point on the plane
  glm::vec3 point = (center + mag.x * right + mag.y * down);

  //Get the direction from the eye to the point to cast the ray
  r.direction = glm::normalize(point - eye);
  return r;
}

// function that collides a ray against a set of geometries
__host__ __device__ float collideRay(const ray& r, const staticGeom* geoms, const int numGeoms, glm::vec3& normal, int& idx) {
	float min_dist = FLT_MAX;
	glm::vec3 norm;
	
	for (int i = 0; i < numGeoms; i++) {
		staticGeom g = geoms[i];
		glm::vec3 i_point;
		float d;

		switch (g.type) {
		case SPHERE : 
			{
			d = sphereIntersectionTest(g, r, i_point, norm);
			if (d >= 0.0f && d < min_dist) {
				idx = i;
				min_dist = d;
				normal = norm;
			}
			break;
			}
		case CUBE :
			{
			d = boxIntersectionTest(g, r, i_point, norm);
			if (d >= 0.0f && d < min_dist) {
				idx = i;
				min_dist = d;
				normal = norm;
			}
			break;	
			}
		//TODO: Handle the mesh case here
		default :
			{
			break;
			}
		}
	}
	return (idx == -1 ? -1.0f : min_dist);
}

// Wrapper kernel to call the camera raycasting
__global__ void getInitialRays(cameraData cam, float time, ray* rays) {
	// Get the x,y coords for this thread
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	// get the ray for the thread
	if(x<=cam.resolution.x && y<=cam.resolution.y) {
		rays[index] = raycastFromCameraKernel(cam.resolution, time, x, y,  cam.position, cam.view, cam.up, cam.fov);
		rays[index].pix_idx = index;
		rays[index].remove = false;
		rays[index].color = glm::vec3(1.0f, 1.0f, 1.0f);
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, const float iterations){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0/iterations;
      color.y = image[index].y*255.0/iterations;
      color.z = image[index].z*255.0/iterations;

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

// Core pathtracer kernel
__global__ void pathtraceRays(float time, int rayDepth, int maxDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
							ray* rays, int numRays) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index < numRays) {
	// Get the ray for this thread
	ray r = rays[index];

	// TEMP: Skip removed rays
	if (r.remove) {
		return;
	}

	// Initialize the normal vector of intersection
	glm::vec3 normal;

	// Initialize an index for the intersecting geometry
	int geom_idx = -1;

	// Check if the ray hits something
	float dist = collideRay(r, geoms, numberOfGeoms, normal, geom_idx);

	//A miss should be removed and the follow steps can be skipped
	if (geom_idx == -1) {
		r.remove = true;

	// Handle the case where the object is a light source (it stops here, and accumulates color)
	} else if (!epsilonCheck(materials[geoms[geom_idx].materialid].emittance, 0.0f)) {
		r.color *= materials[geoms[geom_idx].materialid].emittance;
		colors[r.pix_idx] += r.color;
		r.remove = true;
	} else {

		// Update the ray's color from the object
		r.color *= materials[geoms[geom_idx].materialid].color;

		// Update the origin of the ray based on the intersection point (slightly outside)
		r.origin += r.direction * (dist * 0.99f);

		// Handle the case where the object is reflective
		if (!epsilonCheck(materials[geoms[geom_idx].materialid].hasReflective, 0.0f)) {
			r.direction -= 2.0f*glm::dot(r.direction, normal)*normal;
			r.direction = glm::normalize(r.direction);
		// Handle the case where the object is pure diffusive
		} else {
			thrust::default_random_engine rng(hash((time+1)*(rayDepth+1)*index));
			thrust::uniform_real_distribution<float> u01(0,1);
			r.direction = calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng));
		}

		// Set the flag for removal if its at max depth and add color to the pixel for it
		if (rayDepth >= (maxDepth - 1)) {
			colors[r.pix_idx] += r.color;
			r.remove = true;
		}
	}

	// Put the ray back into global memory
	rays[index] = r;
  }

}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaPathtraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 5; //determines how many bounces the pathtracer traces

  // set up crucial magic
  int tileSize = 16;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  // send image to GPU
  glm::vec3* cudaImage;
  cudaMalloc((void**)&cudaImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy(cudaImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // package geometry
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
  
  // send geometry to GPU
  staticGeom* cudaGeoms;
  cudaMalloc((void**)&cudaGeoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy(cudaGeoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  // send materials to GPU
  material* cudaMaterials;
  cudaMalloc((void**)&cudaMaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudaMaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // allocate ray data on GPU
  ray* cudaRays;
  int numRays = cam.resolution.x * cam.resolution.y;
  cudaMalloc((void**)&cudaRays, numRays*sizeof(ray));

  // Clear off the current image
  //clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaImage);

  // Get initial rays
  getInitialRays<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, (float)iterations, cudaRays);

  // Setup thread/block breakdown for pathtracing
  int rayThreads = 256;
  dim3 rayThreadsPerBlock(rayThreads);
  dim3 rayBlocksPerGrid((int)ceil((float)numRays / (float)rayThreads));

  // loop over kernel launches
  for (int tr = 0; tr < traceDepth; tr++) {
	pathtraceRays<<<rayBlocksPerGrid, rayThreadsPerBlock>>>((float)iterations, tr, traceDepth, cudaImage, cudaGeoms, numberOfGeoms, cudaMaterials, numberOfMaterials, cudaRays, numRays);
	cudaDeviceSynchronize();
  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaImage, (float)iterations);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaImage );
  cudaFree( cudaGeoms );
  cudaFree( cudaMaterials );
  cudaFree( cudaRays );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
