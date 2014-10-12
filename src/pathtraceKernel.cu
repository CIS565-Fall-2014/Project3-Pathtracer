// CIS565 CUDA Pathtracer: A parallel pathtracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

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
__host__ __device__ float collideRay(const ray& r, const worldData* world, worldSizes ws, glm::vec3& normal, int& idx) {
	float min_dist = FLT_MAX;
	glm::vec3 norm;
	glm::vec3 i_point;
	float d;

	// Loop over primitive geometries
	for (int i = 0; i < ws.numberOfGeoms; i++) {
		staticGeom g = world->geoms[i];

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
		default :
			{
			break;
			}
		}
	}

	// Loop over triangles
	glm::vec3 triangle[3];
	for (int i = 0; i < ws.numberOfTriangles; i++) {
		for (int j = 0; j < 3; j++) {
			triangle[j] = world->vertices[world->indices[3*i + j]];
		}
		d = triangleIntersectionTest(triangle, r, i_point, norm);
		if (d >= 0.0f && d < min_dist) {
			//idx = i; TODO: Associate a triangle with a geometry
			min_dist = d;
			normal = norm;
		}
	}

		return (idx == -1 ? -1.0f : min_dist);
}

// predicate struct for thrust stream compaction (utilizing the google code template for doing this)
struct is_kept {
	__host__ __device__
	bool operator() (const ray& r)
	{
		return !r.remove;
	}
};

// templated kernel for prefix sum
template <typename T>
__global__ void nBlockPrefixSum(T* in, T* out) {

	int index = blockIdx.x * blockDim.x + threadIdx.x; //can't keep it simple anymore
	int s_index = threadIdx.x;

	// Create shared memory
	extern __shared__ T s[];
	T* in_s = &s[0];
	T* out_s = &s[blockDim.x];
	__shared__ float lower_tile_value;

	//Load into shared memory
	//Start by shifting right since the calculation is going to be inclusive and we want exclusive
	if (index > 0) {
		in_s[s_index] = in[index - 1];
	}
	else {
		in_s[s_index] = 0.0f;
	}
	__syncthreads();

	//Calculate the max depth
	int max_depth = ceil(log((float)blockDim.x) / log(2.0f));

	//Loop over each depth
	for (int d = 1; d <= max_depth; d++) {

		//Calculate the offset for the current depth
		int off = pow(2.0f, d - 1);

		// compute left->right or right->left
		if ((d % 2) == 1) {

			// calculate the sum
			if (s_index >= off) {
				out_s[s_index] = in_s[s_index - off] + in_s[s_index];
			}
			else {
				//Have to leave other elements alone
				out_s[s_index] = in_s[s_index];
			}

		}
		else {

			// calculate the sum
			if (s_index >= off) {
				in_s[s_index] = out_s[s_index - off] + out_s[s_index];
			}
			else {
				//Have to leave other elements alone
				in_s[s_index] = out_s[s_index];
			}

		}

		//Sync threads before the next depth to use proper values
		__syncthreads();

	}

	//Copy the correct result to global memory
	if ((max_depth % 2) == 1) {
		out[index] = out_s[s_index];
	}
	else {
		out[index] = in_s[s_index];
	}

	__syncthreads();

	//Determine the number of additional loops that will be required
	int kernel_calls = ceil(log((float)gridDim.x) / log(2.0f)) + 1;

	//Loop over the kernel calls, doing a pseudo-serial scan over the remaining layers
	for (int k = 0; k < kernel_calls; k++) {

		//Swap out and in
		T* temp = in;
		in = out;
		out = temp;

		//Load the needed value for this tile into shared memory
		if (s_index == 0) {
			if (blockIdx.x >= (int)pow(2.0f, k)) {
				lower_tile_value = in[(blockIdx.x + 1 - (int)pow(2.0f, k))*blockDim.x - 1];
			}
			else {
				lower_tile_value = 0.0f;
			}
		}
		__syncthreads();

		//Add to the output
		out[index] = in[index] + lower_tile_value;
		__syncthreads();
	}

}

// templated kernel for compaction
template <typename T>
__global__ void compactArray(const T* in, const int* indices, const int* mask, T* out) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (mask[index] == 1) {
		out[indices[index]] = in[index];
	}
}

// Wrapper kernel to call the camera raycasting
__global__ void getInitialRays(cameraData cam, float time, ray* rays/*, int* mask*/) {
	// Get the x,y coords for this thread
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	// get the ray for the thread
	if(x<=cam.resolution.x && y<=cam.resolution.y) {
		rays[index] = raycastFromCameraKernel(cam.resolution, time, x, y,  cam.position, cam.view, cam.up, cam.fov);
		rays[index].pix_idx = index;
		rays[index].color = glm::vec3(1.0f, 1.0f, 1.0f);
		rays[index].remove = false;
		//mask[index] = 1;
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
__global__ void pathtraceRays(float time, int rayDepth, int maxDepth, glm::vec3* colors, worldData* world, worldSizes ws, 
							ray* rays, int numRays/*, int* rayMask*/) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index < numRays) {
	// Get the ray for this thread
	ray r = rays[index];
	//int mask = rayMask[index];

	// Initialize the normal vector of intersection
	glm::vec3 normal;

	// Initialize an index for the intersecting geometry
	int geom_idx = -1;

	// Check if the ray hits something
	float dist = collideRay(r, world, ws, normal, geom_idx);

	//A miss should be removed and the follow steps can be skipped
	if (geom_idx == -1) {
		//mask = 0;
		r.remove = true;

	// Handle the case where the object is a light source (it stops here, and accumulates color)
	} else if (!epsilonCheck(world->materials[world->geoms[geom_idx].materialid].emittance, 0.0f)) {
		r.color *= world->materials[world->geoms[geom_idx].materialid].emittance;
		colors[r.pix_idx] += r.color;
		//mask = 0;
		r.remove = true;
	} else {

		// Update the ray's color from the object
		r.color *= world->materials[world->geoms[geom_idx].materialid].color;

		// Update the origin of the ray based on the intersection point (slightly outside)
		r.origin += r.direction * (dist * 0.9999f);

		// Handle the case where the object is refractive
		if (!epsilonCheck(world->materials[world->geoms[geom_idx].materialid].hasRefractive, 0.0f)) {
			float r1 = 1.0f;
			float r2 = world->materials[world->geoms[geom_idx].materialid].indexOfRefraction;
			if (glm::dot(normal, r.direction) > 0.0f) {
				float temp = r1;
				r1 = r2;
				r2 = temp;
			}
			r.direction = calculateTransmissionDirection(normal, r.direction, r1, r2);
		// Handle the case where the object is reflective
		} else if (!epsilonCheck(world->materials[world->geoms[geom_idx].materialid].hasReflective, 0.0f)) {
			r.direction = calculateReflectionDirection(normal, r.direction);
		// Handle the case where the object is pure diffusive
		} else {
			thrust::default_random_engine rng(hash((time+1)*(rayDepth+1)*index));
			thrust::default_random_engine rng2(hash((time/2)*(rayDepth + 1)*index));
			thrust::uniform_real_distribution<float> u01(0,1);
			r.direction = calculateRandomDirectionInHemisphere(normal, (float)u01(rng), (float)u01(rng2));
		}

		// Randomly kill with russian roulette
		thrust::default_random_engine rng(hash((time + 1)*(rayDepth/2)*index));
		thrust::uniform_real_distribution<float> u01(0, 1);

		// Set the flag for removal if its at max depth and add color to the pixel for it
		if (rayDepth >= (maxDepth - 1) || (float)u01(rng) < 0.01) {
			colors[r.pix_idx] += r.color;
			//mask = 0;
			r.remove = true;
		}
	}

	// Put the ray back into global memory
	rays[index] = r;
	//rayMask[index] = mask;
  }

}

//Wrapper for kernel calls that compact a set of rays (on the device)
void compactRays(ray* raysIn, int num, int* mask, ray* raysOut) {
	int threads_per_block = 256;
	int needed_bytes = 2 * threads_per_block * sizeof(int);
	int n_blocks = num / threads_per_block;

	//Prefix sum the mask to get the indices
	int* indices;
	int* mask_copy;
	int* indices_h = (int*) malloc(num*sizeof(int));
	cudaMalloc((void**)&indices, num*sizeof(int));
	cudaMalloc((void**)&mask_copy, num*sizeof(int));
	cudaMemcpy(mask_copy, mask, num*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	nBlockPrefixSum<int><<<n_blocks, threads_per_block, needed_bytes>>>(mask_copy, indices);
	cudaDeviceSynchronize();

	//Compact into the output by using the indices
	compactArray<ray><<<n_blocks, threads_per_block>>>(raysIn, indices, mask, raysOut);
	cudaDeviceSynchronize();

	//Determine the new length
	cudaMemcpy(indices_h, indices, num*sizeof(int), cudaMemcpyDeviceToHost);
	num = indices_h[num-1] + 1;

	//Cleanup
	cudaFree(indices);
	free(indices_h);
}

//Wrapper for stream compaction with thrust
void compactRaysThrust(ray* raysIn, int& num, ray* raysOut) {
	thrust::device_ptr<ray> in = thrust::device_pointer_cast<ray>(raysIn);
	thrust::device_ptr<ray> out = thrust::device_pointer_cast<ray>(raysOut);
	num = thrust::copy_if(in, in+num, out, is_kept()) - out;
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaPathtraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, geom* geoms, worldSizes ws){
  
  int traceDepth = 20; //determines how many bounces the pathtracer traces

  // set up crucial magic
  int tileSize = 16;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  // send image to GPU
  glm::vec3* cudaImage;
  cudaMalloc((void**)&cudaImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy(cudaImage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // setup the world on the GPU
  worldData hostWorld;
  worldData* cudaWorld;
  cudaMalloc((void**)&cudaWorld, sizeof(worldData));

  // package geometry
  staticGeom* geomList = new staticGeom[ws.numberOfGeoms];
  for(int i=0; i<ws.numberOfGeoms; i++){
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
  cudaMalloc((void**)&cudaGeoms, ws.numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy(cudaGeoms, geomList, ws.numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  hostWorld.geoms = cudaGeoms;
  
  // send materials to GPU
  material* cudaMaterials;
  cudaMalloc((void**)&cudaMaterials, ws.numberOfMaterials*sizeof(material));
  cudaMemcpy(cudaMaterials, materials, ws.numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  hostWorld.materials = cudaMaterials;

  // send triangles to GPU
  /*glm::vec3* cudaVertices;
  int* cudaIndices;
  cudaMalloc((void**)&cudaVertices, ws.numberOfVertices*sizeof(glm::vec3));
  cudaMalloc((void**)&cudaIndices, ws.numberOfTriangles*3*sizeof(int));
  hostWorld.vertices = cudaVertices;
  hostWorld.indices = cudaIndices;*/

  // send the final world to the GPU
  cudaMemcpy(cudaWorld, &hostWorld, sizeof(worldData), cudaMemcpyHostToDevice);

  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  // allocate ray data on GPU (2 sets of rays so we can compact back and forth)
  ray* cudaRays1;
  ray* cudaRays2;
  //int* cudaRayMask;
  int numRays = cam.resolution.x * cam.resolution.y;
  cudaMalloc((void**)&cudaRays1, numRays*sizeof(ray));
  cudaMalloc((void**)&cudaRays2, numRays*sizeof(ray));
  //cudaMalloc((void**)&cudaRayMask, numRays*sizeof(int));

  // Clear off the current image
  //clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaImage);

  // Get initial rays
  getInitialRays<<<fullBlocksPerGrid, threadsPerBlock>>>(cam, (float)iterations, cudaRays1/*, cudaRayMask*/);

  // Setup thread/block breakdown for pathtracing
  int rayThreads = 256;
  dim3 rayThreadsPerBlock(rayThreads);
  dim3 rayBlocksPerGrid((int)ceil((float)numRays / (float)rayThreads));

  // initialize device pointers to use in the main loop
  ray* raysIn = cudaRays1;
  ray* raysOut = cudaRays2;

  // loop over kernel launches
  for (int tr = 0; tr < traceDepth; tr++) {
	//Pathtrace the current rays
	pathtraceRays<<<rayBlocksPerGrid, rayThreadsPerBlock>>>((float)iterations, tr, traceDepth, cudaImage, cudaWorld, ws, raysIn, numRays/*, cudaRayMask*/);
	cudaDeviceSynchronize();

	//Stream compact to get rid of finished rays for the next iteration
	//compactRays(raysIn, numRays, cudaRayMask, raysOut);
	compactRaysThrust(raysIn, numRays, raysOut);

	//No need to continue if there are no more rays
	if (numRays == 0) {
		break;
	}

	///Update the number of needed blocks based on the condensed number of rays
	rayBlocksPerGrid = (int)ceil((float)numRays / (float)rayThreads);

	//Flip the input and output for the next iteration
	ray* raysTemp = raysIn;
	raysIn = raysOut;
	raysOut = raysTemp;
  }

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaImage, (float)iterations);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaImage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaImage );
  cudaFree( cudaWorld );
  cudaFree( cudaGeoms );
  cudaFree( cudaMaterials );
  //cudaFree( cudaVertices );
  //cudaFree( cudaIndices );
  cudaFree( cudaRays1 );
  cudaFree( cudaRays2 );
  //cudaFree(cudaRayMask);
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
