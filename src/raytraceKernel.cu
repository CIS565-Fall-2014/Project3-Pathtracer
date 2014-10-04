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
							 float current_iteration,
							 int x,
							 int y,
							 glm::vec3 eyep,
							 glm::vec3 vdir,
							 glm::vec3 uvec,
							 glm::vec2 fov )
{
	// TODO: Supersampled anti-aliasing.
	// TODO: Depth of field.
	// TODO: Motion blur.

	// TODO: Compute a, m, h, and v otuside this method since they remain constant between rays.

	// A - cross product of C and U
	glm::vec3 a = glm::cross( vdir, uvec );
	
	// M - midpoint of frame buffer
	glm::vec3 m = eyep + vdir;

	// H - horizontal NDC value; parallel to A
	glm::vec3 h = ( a * glm::length( vdir ) * ( float )tan( fov.x * ( PI / 180.0f ) ) ) / glm::length( a );
	
	// V - vertical NDC value; parallel to B
	glm::vec3 v = glm::vec3( 0.0f, resolution.y * glm::length( h ) / resolution.x, 0.0f );

	float sx = ( float )x / ( resolution.x - 1.0f );
	float sy = 1.0f - ( ( float )y / ( resolution.y - 1.0f ) ); // TODO: The -1 here might flip the image vertically.
	//float sy = ( float )y / ( resolution.y - 1.0f );

	glm::vec3 image_point = m + ( ( 2.0f * sx - 1.0f ) * h ) + ( ( 2.0f * sy - 1.0f ) * v );
	glm::vec3 dir = image_point - eyep;

	ray r;
	r.origin = eyep;
	r.direction = glm::normalize( dir );
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
				  int num_geoms,
				  material *materials,
				  int num_materials )
{
	// TODO: Doesn't this method need materials to perform all necessary computations?
	// TODO: Setup recursion base cases based on raytrace_depth and Russion Roulette.

	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );

	if ( ( x <= resolution.x && y <= resolution.y ) ) {
		ray r = raycastFromCameraKernel( resolution,
										 current_iteration,
										 x,
										 y,
										 cam.position,
										 cam.view,
										 cam.up,
										 cam.fov );


		////////////////////////////////////////////////////
		// Intersection testing.
		////////////////////////////////////////////////////

		float dist_to_intersection = FLT_MAX;
		glm::vec3 intersection_point;
		glm::vec3 normal;

		float temp_dist_to_intersection = -1.0f;
		glm::vec3 temp_intersection_point;
		glm::vec3 temp_normal;

		bool did_intersect = false;

		// Test.
		glm::vec3 intersected_geom_color;

		// Find nearest intersection, if any.
		for ( int i = 0; i < num_geoms; ++i ) {
			if ( geoms[i].type == SPHERE ) {
				temp_dist_to_intersection = sphereIntersectionTest( geoms[i],
																	r,
																	temp_intersection_point,
																	temp_normal );
			}
			else if ( geoms[i].type == CUBE ) {
				// TODO: boxIntersectionTest().
			}
			else if ( geoms[i].type == MESH ) {
				// TODO: meshIntersectionTest() or triangleIntersectionTest().
			}

			// Update nearest intersection if closer intersection has been found.
			if ( temp_dist_to_intersection > 0.0f && temp_dist_to_intersection < dist_to_intersection ) {
				dist_to_intersection = temp_dist_to_intersection;
				intersection_point = temp_intersection_point;
				normal = temp_normal;

				did_intersect = true;

				// Test.
				intersected_geom_color = materials[geoms[i].materialid].color;
			}
		}


		////////////////////////////////////////////////////
		// Write output.
		////////////////////////////////////////////////////

		// Test sphere intersections.
		if ( did_intersect ) {
			image[index] = intersected_geom_color;
		}
		else {
			image[index] = glm::vec3( 0.0f, 0.0f, 0.0f );
		}

		// Color output image to test for correct ray direction computations.
		//glm::vec3 dir_test = r.direction;
		//if ( dir_test.x < 0.0f ) {
		//	dir_test.x *= -1.0f;
		//}
		//if ( dir_test.y < 0.0f ) {
		//	dir_test.y *= -1.0f;
		//}
		//if ( dir_test.z < 0.0f ) {
		//	dir_test.z *= -1.0f;
		//}
		//image[index] = dir_test;

		// Assign random colors to output image pixels.
		//image[index] = generateRandomNumberFromThread( resolution, current_iteration, x, y );
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
  
	// Send materials to GPU.
	material *cuda_materials = NULL;
	float size_material_list = num_materials * sizeof( material );
	cudaMalloc( ( void** )&cuda_materials,
				size_material_list );
	cudaMemcpy( cuda_materials,
				materials,
				size_material_list,
				cudaMemcpyHostToDevice );
  
	// Package up camera.
	cameraData cam;
	cam.resolution = render_cam->resolution;
	cam.position = render_cam->positions[frame];
	cam.view = render_cam->views[frame];
	cam.up = render_cam->ups[frame];
	cam.fov = render_cam->fov;

	// Launch raytraceRay kernel.
	raytraceRay<<< full_blocks_per_grid, threads_per_block >>>( render_cam->resolution,
																( float )current_iteration,
																cam,
																1, // Start recursion with raytrace depth of 1.
																cuda_image,
																cuda_geoms,
																num_geoms,
																cuda_materials,
																num_materials );

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
	cudaFree( cuda_materials );
	delete geom_list;

	// Make certain the kernel has completed.
	//cudaThreadSynchronize(); // Deprecated.
	cudaDeviceSynchronize();

	checkCUDAError( "Kernel failed!" );
}