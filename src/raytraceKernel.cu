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


void checkCUDAError( const char *msg )
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		fprintf( stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err ) ); 
		exit( EXIT_FAILURE );
	}
}


// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__
__device__
glm::vec3 generateRandomNumberFromThread( glm::vec2 resolution,
										  float time,
										  int x,
										  int y )
{
	int index = x + ( y * resolution.x );
   
	thrust::default_random_engine rng( hash( index * time ) );
	thrust::uniform_real_distribution<float> u01( 0, 1 );

	return glm::vec3( ( float )u01( rng ),
					  ( float )u01( rng ),
					  ( float )u01( rng ) );
}


// TODO: IMPLEMENT THIS FUNCTION
// Function that does the initial raycast from the camera
__host__
__device__
ray raycastFromCameraKernel( glm::vec2 resolution,
							 float time,
							 int x,
							 int y,
							 glm::vec3 eye,
							 glm::vec3 view,
							 glm::vec3 up,
							 glm::vec2 fov )
{
	// TODO: Supersampled anti-aliasing.
	// TODO: Depth of field.
	// TODO: Motion blur.

	ray r;
	r.origin = glm::vec3( 0.0f,
						  0.0f,
						  0.0f );
	r.direction = glm::vec3( 0.0f,
							 0.0f,
							 -1.0f );
	return r;
}


// Kernel that blacks out a given image buffer
__global__
void clearImage( glm::vec2 resolution,
				 glm::vec3* image )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
	if ( x <= resolution.x && y <= resolution.y ) {
		image[index] = glm::vec3( 0, 0, 0 );
	}
}


// Kernel that writes the image to the OpenGL PBO directly.
__global__
void sendImageToPBO( uchar4* PBOpos,
					 glm::vec2 resolution,
					 glm::vec3* image )
{  
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
  
	if ( x <= resolution.x && y <= resolution.y ) {
		glm::vec3 color;
		color.x = image[index].x * 255.0;
		color.y = image[index].y * 255.0;
		color.z = image[index].z * 255.0;

		if ( color.x > 255 ) {
			color.x = 255;
		}
		if ( color.y > 255 ) {
			color.y = 255;
		}
		if ( color.z > 255 ) {
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
// Core raytracer kernel.
__global__
void raytraceRay( glm::vec2 resolution,
				  float current_iteration, // Used solely for random number generation (I think).
				  cameraData cam,
				  int raytrace_depth,
				  glm::vec3 *image,
				  staticGeom *geoms,
				  int num_geoms )
{
	// TODO: Doesn't this method need materials to perform all necessary computations?
	// TODO: Setup recursion base cases based on raytrace_depth and Russion Roulette.

	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );

	if ( ( x <= resolution.x && y <= resolution.y ) ) {

		// TODO: Generate a ray by calling raycastFromCameraKernel(). 

		// TODO: Replace this random color with the computed pixel color.
		image[index] = generateRandomNumberFromThread( resolution, current_iteration, x, y );
	}
}


// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management.
void cudaRaytraceCore( uchar4 *pbo_pos,
					   camera *render_cam,
					   int frame,
					   int current_iteration,
					   material *materials,
					   int num_materials,
					   geom *geoms,
					   int num_geoms )
{
	// TODO: Is there a reason tile_size = 8?
	// TODO: Do the image x- and y-resolutions always need to be multiples of tile_size to ensure full blocks?
	// TODO: Do the ray pooling thing. That probably goes here since this is the "memory management" method.
	// TODO: A complete iteration needs to be completed inside this method. Meaning, a ray must be launched and
	//	resolved (color value computed) for each pixel in the image.

	// Setup crucial magic.
	int tile_size = 8;
	dim3 threads_per_block( tile_size,
							tile_size );
	dim3 full_blocks_per_grid( ( int )ceil( ( float )render_cam->resolution.x / ( float )tile_size ),
							   ( int )ceil( ( float )render_cam->resolution.y / ( float )tile_size ) );
  
	// Send image to GPU.
	glm::vec3 *cuda_image = NULL;
	float size_image = ( int )render_cam->resolution.x * ( int )render_cam->resolution.y * sizeof( glm::vec3 );
	cudaMalloc( ( void** )&cuda_image,
				size_image );
	cudaMemcpy( cuda_image,
				render_cam->image,
				size_image,
				cudaMemcpyHostToDevice );
  
	// Package up geometry.
	staticGeom *geom_list = new staticGeom[num_geoms];
	for ( int i = 0; i < num_geoms; ++i ) {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geom_list[i] = newStaticGeom;
	}
  
	// Send geometry to GPU.
	staticGeom *cuda_geoms = NULL;
	float size_geom_list = num_geoms * sizeof( staticGeom );
	cudaMalloc( ( void** )&cuda_geoms,
				size_geom_list );
	cudaMemcpy( cuda_geoms,
				geom_list,
				size_geom_list,
				cudaMemcpyHostToDevice );

	// TODO: Package up materials and send to GPU.
  
	// Package up camera.
	cameraData cam;
	cam.resolution = render_cam->resolution;
	cam.position = render_cam->positions[frame];
	cam.view = render_cam->views[frame];
	cam.up = render_cam->ups[frame];
	cam.fov = render_cam->fov;

	// TODO: Allocate GPU camera and send camera to GPU?

	// TODO: Create a pool of rays to raycast?
	// TODO: Call raycastFromCameraKernel(), probably.

	// Launch raytraceRay kernel.
	raytraceRay<<< full_blocks_per_grid, threads_per_block >>>( render_cam->resolution,
																( float )current_iteration,
																cam,
																1, // Start recursion with raytrace depth of 1.
																cuda_image,
																cuda_geoms,
																num_geoms );

	// Launch sendImageToPBO kernel.
	sendImageToPBO<<< full_blocks_per_grid, threads_per_block >>>( pbo_pos,
																   render_cam->resolution,
																   cuda_image);

	// Retrieve image from GPU.
	cudaMemcpy( render_cam->image,
				cuda_image,
				size_image,
				cudaMemcpyDeviceToHost );

	// Cleanup.
	cudaFree( cuda_image );
	cudaFree( cuda_geoms );
	delete geom_list;

	// Make certain the kernel has completed.
	//cudaThreadSynchronize(); // Deprecated.
	cudaDeviceSynchronize();

	checkCUDAError( "Kernel failed!" );
}