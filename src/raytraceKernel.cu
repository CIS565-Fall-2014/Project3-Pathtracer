// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"


// Some forward declarations
__host__ __device__ bool sceneIntersection( const ray &r, staticGeom *geoms, int num_geoms, float &t, int &id, glm::vec3 &intersection_point, glm::vec3 &intersection_normal );


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
					 glm::vec3* image,
					 int iterations_so_far )
{  
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
  
	if ( x <= resolution.x && y <= resolution.y ) {
		glm::vec3 color;
		color.x = ( image[index].x / iterations_so_far ) * 255.0;
		color.y = ( image[index].y / iterations_so_far ) * 255.0;
		color.z = ( image[index].z / iterations_so_far ) * 255.0;

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


/*********** CORE PATHTRACING ALGORITHMS ***********/


__host__
__device__
bool sceneIntersection( const ray &r,
						staticGeom *geoms,
						int num_geoms,
						float &t,
						int &id,
						glm::vec3 &intersection_point,
						glm::vec3 &intersection_normal )
{
	t = FLT_MAX;
	float temp_t = -1.0f;
	glm::vec3 temp_intersection_point;
	glm::vec3 temp_intersection_normal;

	// Find nearest intersection, if any.
	for ( int i = 0; i < num_geoms; ++i ) {
		if ( geoms[i].type == SPHERE ) {
			temp_t = sphereIntersectionTest( geoms[i],
											 r,
											 temp_intersection_point,
											 temp_intersection_normal );
		}
		else if ( geoms[i].type == CUBE ) {
			temp_t = boxIntersectionTest( geoms[i],
										  r,
										  temp_intersection_point,
										  temp_intersection_normal );
		}

		// Update nearest intersection if closer intersection has been found.
		if ( temp_t > 0.0f && temp_t < t ) {
			t = temp_t;
			intersection_point = temp_intersection_point;
			intersection_normal = temp_intersection_normal;
			id = geoms[i].materialid;
		}
	}

	return ( t < FLT_MAX );
}


// Compute rays from camera through pixels and store in ray_pool.
__global__
void raycastFromCameraKernel( ray *ray_pool,
							 glm::vec2 resolution,
							 glm::vec3 eyep,
							 glm::vec3 m,
							 glm::vec3 h,
							 glm::vec3 v )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = ( y * ( int )resolution.x ) + x;

	if ( index > ( resolution.x * resolution.y ) ) {
		return;
	}

	float sx = ( float )x / ( resolution.x - 1.0f );
	float sy = 1.0f - ( ( float )y / ( resolution.y - 1.0f ) );

	glm::vec3 image_point = m + ( ( 2.0f * sx - 1.0f ) * h ) + ( ( 2.0f * sy - 1.0f ) * v );
	glm::vec3 dir = image_point - eyep;

	ray r;
	r.origin = eyep;
	r.direction = glm::normalize( dir );
	r.image_coords = glm::vec2( x, y );

	ray_pool[index] = r;
}


// Test kernel to verify raycastFromCameraKernel results were correct.
__global__
void testOutputKernel( glm::vec3 *image,
					   ray *ray_pool,
					   glm::vec2 resolution )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = ( y * ( int )resolution.x ) + x;

	if ( index > ( resolution.x * resolution.y ) ) {
		return;
	}

	glm::vec3 normal_color = ray_pool[index].direction;
	normal_color.x = ( normal_color.x < 0.0f ) ? ( normal_color.x * -1.0f ) : normal_color.x;
	normal_color.y = ( normal_color.y < 0.0f ) ? ( normal_color.y * -1.0f ) : normal_color.y;
	normal_color.z = ( normal_color.z < 0.0f ) ? ( normal_color.z * -1.0f ) : normal_color.z;

	image[index] = normal_color;
}


__global__
void uselessKernel()
{
	int ray_pool_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	return;
}


// Core raytracer kernel.
__global__
void raytraceRay( ray *ray_pool,
				  int ray_pool_size,
				  glm::vec2 resolution,
				  float current_iteration, // Used solely for random number generation (I think).
				  cameraData cam,
				  int raytrace_depth,
				  glm::vec3 *image,
				  staticGeom *geoms,
				  int num_geoms,
				  material *materials )
{
	// TODO: Russian Roulette to kill rays.

	int ray_pool_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	if ( ray_pool_index > ray_pool_size ) {
		return;
	}

	ray r = ray_pool[ray_pool_index];
	int image_pixel_index = ( r.image_coords.y * ( int )resolution.x ) + r.image_coords.x;

	// Nudge ray along it's direction to avoid intersecting with the surface it originates from.
	r.origin += ( r.direction * 0.001f );

	// Intersection testing.
	float dist_to_intersection;
	int material_index;
	glm::vec3 intersection_point;
	glm::vec3 intersection_normal;
	bool ray_did_intersect_something = sceneIntersection( r,
														  geoms,
														  num_geoms,
														  dist_to_intersection,		// Reference to be filled.
														  material_index,			// Reference to be filled.
														  intersection_point,		// Reference to be filled.
														  intersection_normal );	// Reference to be filled.

	if ( ray_did_intersect_something ) {
		// Properly orient normal for cases where ray intersects inner surface of glass object.
		intersection_normal = ( glm::dot( intersection_normal, r.direction ) < 0.0f ) ? intersection_normal : ( -1.0f * intersection_normal );

		material mat = materials[material_index];

		// Ray hits light source.
		if ( mat.emittance > 0.0f ) {
			r.color = ( r.color * mat.color * mat.emittance );
			image[image_pixel_index] += r.color;
			r.is_active = false;
		}
		else {
			// Diffuse.
			if ( !mat.hasReflective && !mat.hasRefractive ) {
				glm::vec3 rand = generateRandomNumberFromThread( resolution,
																 current_iteration * raytrace_depth,
																 r.image_coords.x,
																 r.image_coords.y );
				r.direction = calculateRandomDirectionInHemisphere( intersection_normal, rand.x, rand.y );
				r.color = r.color * mat.color;
				//image[image_pixel_index] += r.color;

				// TODO: Shadow rays?
			}
			// Perfect specular.
			else {
				r.direction = calculateReflectionDirection( intersection_normal, r.direction );
			}
		}
	}
	else {
		image[image_pixel_index] += glm::vec3( 0.0f, 0.0f, 0.0f ); // Background color.
		r.is_active = false;
	}

	r.origin = intersection_point; // Origin point for next ray.
	ray_pool[ray_pool_index] = r;



	// Test.
	//if ( ray_did_intersect_something ) {
	//	image[image_pixel_index] = materials[material_index].color;
	//}
	//else {
	//	image[image_pixel_index] = glm::vec3( 0.0f, 0.0f, 0.0f ); // Background color.
	//}
}


// thrust predicate to cull inactive rays from ray pool.
struct RayIsInactive
{
	__host__
	__device__
	bool operator()( const ray &r )
	{
		return !r.is_active;
	}
};


// Wrapper that sets up kernel calls and handles memory management.
// Handles one pathtrace iteration. Called many times to produce a rendered image.
void cudaRaytraceCore( uchar4 *pbo_pos,
					   camera *render_cam,
					   int frame,
					   int current_iteration,
					   material *materials,
					   int num_materials,
					   geom *geoms,
					   int num_geoms )
{
	// Tune these for performance.
	int depth = 10;
	int camera_raycast_tile_size = 8;
	int raytrace_tile_size = 128;

	// Setup crucial magic.
	dim3 threads_per_block( camera_raycast_tile_size,
							camera_raycast_tile_size );
	dim3 full_blocks_per_grid( ( int )ceil( ( float )render_cam->resolution.x / ( float )camera_raycast_tile_size ),
							   ( int )ceil( ( float )render_cam->resolution.y / ( float )camera_raycast_tile_size ) );
  
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
	
	// Variables to compute rays originating from render camera.
	glm::vec3 a = glm::cross( cam.view, cam.up );
	glm::vec3 m = cam.position + cam.view; // Midpoint of frame buffer.
	glm::vec3 h = ( a * glm::length( cam.view ) * ( float )tan( cam.fov.x * ( PI / 180.0f ) ) ) / glm::length( a ); // Horizontal NDC value.
	glm::vec3 v = glm::vec3( 0.0f, cam.resolution.y * glm::length( h ) / cam.resolution.x, 0.0f ); // Vertical NDC value.

	// Allocate device memory for ray pool.
	ray *cuda_ray_pool = NULL;
	int num_rays = ( int )( render_cam->resolution.x * render_cam->resolution.y );
	cudaMalloc( ( void** )&cuda_ray_pool,
				num_rays * sizeof( ray ) );
	
	// Initialize ray pool with rays originating at the render camera directed through each pixel in the image buffer.
	raycastFromCameraKernel<<< full_blocks_per_grid, threads_per_block >>>( cuda_ray_pool,
																			render_cam->resolution,
																			cam.position,
																			m,
																			h,
																			v );

	//testOutputKernel<<< full_blocks_per_grid, threads_per_block >>>( cuda_image,
	//																 cuda_ray_pool,
	//																 render_cam->resolution );

	// Launch raytraceRay kernel once per raytrace depth.
	for ( int i = 0; i < depth; ++i ) {
		dim3 threads_per_raytrace_block( raytrace_tile_size );
		dim3 blocks_per_raytrace_grid( ( int )ceil( ( float )num_rays / ( float )raytrace_tile_size ) );

		// Test.
		//uselessKernel<<< blocks_per_raytrace_grid, threads_per_raytrace_block >>>();

		// Launch raytraceRay kernel.
		raytraceRay<<< blocks_per_raytrace_grid, threads_per_raytrace_block >>>( cuda_ray_pool,
																				 num_rays,
																				 render_cam->resolution,
																				 ( float )current_iteration,
																				 cam,
																				 ( i + 1 ), // Start recursion with raytrace depth of 1.
																				 cuda_image,
																				 cuda_geoms,
																				 num_geoms,
																				 cuda_materials );

		// Note: Stream compaction is slow.
		thrust::device_ptr<ray> ray_pool_device_ptr( cuda_ray_pool );
		thrust::device_ptr<ray> culled_ray_pool_device_ptr = thrust::remove_if( ray_pool_device_ptr,
																				ray_pool_device_ptr + num_rays,
																				RayIsInactive() );

		// Compute number of active rays in ray pool.
		num_rays = culled_ray_pool_device_ptr.get() - ray_pool_device_ptr.get();
	}

	// Launch sendImageToPBO kernel.
	sendImageToPBO<<< full_blocks_per_grid, threads_per_block >>>( pbo_pos,
																   render_cam->resolution,
																   cuda_image,
																   current_iteration );

	// Retrieve image from GPU.
	cudaMemcpy( render_cam->image,
				cuda_image,
				size_image,
				cudaMemcpyDeviceToHost );

	// Cleanup.
	cudaFree( cuda_image );
	cudaFree( cuda_geoms );
	cudaFree( cuda_materials );
	cudaFree( cuda_ray_pool );
	delete geom_list;

	// Make certain the kernel has completed.
	//cudaThreadSynchronize(); // Deprecated.
	cudaDeviceSynchronize();

	checkCUDAError( "Kernel failed!" );
}