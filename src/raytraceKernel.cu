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


// Some new types.
enum MaterialType {
	IDEAL_DIFFUSE,
	PERFECT_SPUCULAR,
	GLASS
};


// Some forward declarations.
__host__ __device__ bool sceneIntersection( const ray &r, staticGeom *geoms, int num_geoms, float &t, int &id, glm::vec3 &intersection_point, glm::vec3 &intersection_normal );
__host__ __device__ MaterialType determineMaterialType( material mat );
__host__ __device__ glm::vec2 computePixelSubsampleLocation( glm::vec2 pixel_index, glm::vec2 image_resolution, int current_iteration );


void checkCUDAError( const char *msg )
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		fprintf( stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err ) );
		
		// DEBUG.
		std::cin.ignore();

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
	//float simple_hash_seed = ( float )( ( int )( index * time ) % INT_MAX );
	//float simple_hash_seed = ( float )( ( int )( index * time ) % 1000000 );

	thrust::default_random_engine rng( simpleHash( index * time ) );
	//thrust::default_random_engine rng( simpleHash( simple_hash_seed ) );
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
			//id = geoms[i].materialid;
			id = i;
		}
	}

	return ( t < FLT_MAX );
}


//__host__
//__device__
//bool lightIntersection( const ray &r,
//						staticGeom *geoms,
//						int num_geoms,
//						int &obj_id )
//{
//	float t = FLT_MAX;
//	float temp_t = -1.0f;
//	glm::vec3 temp_intersection_point;
//	glm::vec3 temp_intersection_normal;
//
//	// Find nearest intersection, if any.
//	for ( int i = 0; i < num_geoms; ++i ) {
//		if ( geoms[i].type == SPHERE ) {
//			temp_t = sphereIntersectionTest( geoms[i],
//											 r,
//											 temp_intersection_point,
//											 temp_intersection_normal );
//		}
//		else if ( geoms[i].type == CUBE ) {
//			temp_t = boxIntersectionTest( geoms[i],
//										  r,
//										  temp_intersection_point,
//										  temp_intersection_normal );
//		}
//
//		// Update nearest intersection if closer intersection has been found.
//		if ( temp_t > 0.0f && temp_t < t ) {
//			t = temp_t;
//			obj_id = i;
//		}
//	}
//
//	return ( t < FLT_MAX );
//}


// Compute rays from camera through pixels and store in ray_pool.
__global__
void raycastFromCameraKernel( ray *ray_pool,
							 glm::vec2 resolution,
							 glm::vec3 eyep,
							 glm::vec3 m,
							 glm::vec3 h,
							 glm::vec3 v,
							 int current_iteration )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = ( y * ( int )resolution.x ) + x;

	if ( index > ( resolution.x * resolution.y ) ) {
		return;
	}

	// TODO: Add a switch to turn anti-aliasing on/off.
	glm::vec2 pixel_location = computePixelSubsampleLocation( glm::vec2( x, y ), resolution, current_iteration );

	float sx = pixel_location.x / ( resolution.x - 1.0f );
	float sy = 1.0f - ( pixel_location.y / ( resolution.y - 1.0f ) );

	//float sx = ( float )x / ( resolution.x - 1.0f );
	//float sy = 1.0f - ( ( float )y / ( resolution.y - 1.0f ) );

	glm::vec3 image_point = m + ( ( 2.0f * sx - 1.0f ) * h ) + ( ( 2.0f * sy - 1.0f ) * v );

	glm::vec3 dir = image_point - eyep;

	ray r;
	r.origin = eyep;
	r.direction = glm::normalize( dir );
	r.image_coords = glm::vec2( x, y );

	ray_pool[index] = r;
}


__host__
__device__
glm::vec2 computePixelSubsampleLocation( glm::vec2 pixel_index,
										 glm::vec2 image_resolution,
										 int current_iteration )
{
	// TODO: Set pixel index (x, y) as center of pixel instead of bottom left corner.
	// TODO: Get rid of loop b/c it's completely unnecessary and wastes resources.

	const int SUBSAMPLE_ROWS = 4;
	const int SUBSAMPLE_COLS = 4;

	float row_height = 1.0f / SUBSAMPLE_ROWS;
	float col_width = 1.0f / SUBSAMPLE_COLS;

	float left_bounds, upper_bounds;
	float x_percentage, y_percentage, new_x, new_y;

	int sub_pixel_num = current_iteration % ( SUBSAMPLE_ROWS * SUBSAMPLE_COLS );
	
	// partition pixel into grid using perfect_square and randomly sample one ray through each grid space
	upper_bounds = ( float )pixel_index.y;
	for ( int y = 0; y < SUBSAMPLE_ROWS; ++y ) {
		left_bounds = ( float )pixel_index.x;
		for ( int x = 0; x < SUBSAMPLE_COLS; ++x ) {
			if ( ( y * SUBSAMPLE_COLS ) + x == sub_pixel_num ) {
				glm::vec3 rand = generateRandomNumberFromThread( image_resolution, current_iteration, pixel_index.x, pixel_index.y );
				x_percentage = rand.x;
				y_percentage = rand.y;

				new_x = left_bounds + ( col_width * x_percentage );
				new_y = upper_bounds + ( row_height * y_percentage );

				return glm::vec2( new_x, new_y );
			}
			left_bounds += col_width;
		}
		upper_bounds += row_height;
	}
	return pixel_index;
}


// Core raytracer kernel.
__global__
void raytraceRay( ray *ray_pool,
				  int ray_pool_size,
				  glm::vec2 resolution,
				  float current_iteration, // Used solely for random number generation (I think).
				  cameraData cam,
				  int raytrace_depth,
				  glm::vec3 *image_buffer,
				  staticGeom *geoms,
				  int num_geoms,
				  material *materials,
				  int *texture_info,
				  glm::vec3 *texture_rgb )
{
	int ray_pool_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	if ( ray_pool_index > ray_pool_size ) {
		return;
	}

	ray r = ray_pool[ray_pool_index];
	int image_pixel_index = ( r.image_coords.y * ( int )resolution.x ) + r.image_coords.x;

	// Only necessary when stream compaction is turned off.
	if ( !r.is_active ) {
		return;
	}

	// Nudge ray along it's direction to avoid intersecting with the surface it originates from.
	r.origin += ( r.direction * 0.001f );

	// Intersection testing.
	float dist_to_intersection;
	//int material_index;
	int intersected_obj_id;
	glm::vec3 intersection_point;
	glm::vec3 intersection_normal;
	bool ray_did_intersect_something = sceneIntersection( r,						// Current ray.
														  geoms,					// List of scene geometry.
														  num_geoms,				// Number of pieces of geometry.
														  dist_to_intersection,		// Reference to be filled.
														  intersected_obj_id,		// Reference to be filled.
														  intersection_point,		// Reference to be filled.
														  intersection_normal );	// Reference to be filled.


	// Ray misses. Return background color. Kill ray.
	if ( !ray_did_intersect_something ) {
		image_buffer[image_pixel_index] += glm::vec3( 0.0f, 0.0f, 0.0f );
		r.is_active = false;
		ray_pool[ray_pool_index] = r;
		return;
	}

	// Get intersected object and material of intersected object.
	staticGeom obj = geoms[intersected_obj_id];
	material mat = materials[obj.materialid];

	// Use Roussian Roulette to randomly kill the current ray.
	// rgb is the color of the intersected material, and is used in the computations that follow.
	glm::vec3 rgb = mat.color;
	float p = ( rgb.x > rgb.y && rgb.x > rgb.z ) ? rgb.x : ( ( rgb.y > rgb.z ) ? rgb.y : rgb.z );
    if ( raytrace_depth > 5 ) {
		glm::vec3 rand = generateRandomNumberFromThread( resolution, ( current_iteration * raytrace_depth ), r.image_coords.x, r.image_coords.y );
        if ( rand.x < p ) {
            rgb = rgb * ( 1.0f / p );
        }
        else {
			image_buffer[image_pixel_index] += ( r.color * rgb * mat.emittance );
			r.is_active = false;
			ray_pool[ray_pool_index] = r;
			return;
        }
    }


	// Intersected object has a texture image.
	//glm::vec3 foobar = rgb;
	if ( obj.texture_id != -1 ) {

		glm::vec2 uv( 0.0f, 0.0f );

		if ( obj.type == SPHERE ) {
			uv = computeSphereUVCoordinates( obj, intersection_point );
		}
		else if ( obj.type == CUBE ) {
			uv = computeCubeUVCoordinates( obj, intersection_point );
		}

		// Extract texture info.
		int texture_info_base_index = obj.texture_id * 3;
		int texture_rgb_starting_index = texture_info[texture_info_base_index];
		int texture_width = texture_info[texture_info_base_index + 1];
		int texture_height = texture_info[texture_info_base_index + 2];

		// Compute pixel coordinates within single texture.
		int pixel_coord_x = ( int )( uv.x * texture_width );
		int pixel_coord_y = ( int )( uv.y * texture_height );

		if ( !( pixel_coord_x < 0 || pixel_coord_y < 0 || pixel_coord_x >= texture_width || pixel_coord_y >= texture_height ) ) {
			// Compute texture RGB index.
			int pixel_linear_index = ( pixel_coord_y * texture_width ) + pixel_coord_x;
			int texture_rgb_index = texture_rgb_starting_index + pixel_linear_index;

			if ( texture_rgb_index < texture_rgb_starting_index || texture_rgb_index >= ( texture_rgb_starting_index + texture_width * texture_height ) ) {
				rgb = glm::vec3( 1.0f, 0.078f, 0.577f ); // Hot pink to visualize an error.
			}
			else {
				rgb = texture_rgb[texture_rgb_index];
			}
		}
		else {
			rgb = glm::vec3( 1.0f, 0.078f, 0.577f ); // Hot pink to visualize an error.
		}

	}

	// Ray hits light source. Add acculumated color contribution of ray. Kill ray.
	if ( mat.emittance > 0.0f ) {
		image_buffer[image_pixel_index] += ( r.color * rgb * mat.emittance );
		r.is_active = false;
		ray_pool[ray_pool_index] = r;
		return;
	}

	MaterialType mat_type = determineMaterialType( mat );
	if ( mat_type == IDEAL_DIFFUSE ) {
		glm::vec3 rand = generateRandomNumberFromThread( resolution,
														 current_iteration * raytrace_depth,
														 r.image_coords.x,
														 r.image_coords.y );
		r.direction = calculateRandomDirectionInHemisphere( intersection_normal,
															rand.x,
															rand.y );

		//// Compute direct illumination contribution.
		//glm::vec3 e( 0.0f, 0.0f, 0.0f );
		//for ( int i = 0; i < num_geoms; ++i ) {
		//	const staticGeom &s = geoms[i];
		//	material l_mat = materials[s.materialid];

		//	// Skip geometry that isn't a light source.
		//	if ( l_mat.emittance < 0.0f ) {
		//		continue;
		//	}

		//	glm::vec3 light_dir;
		//	if ( s.type == SPHERE ) {
		//		light_dir = glm::normalize( getRandomPointOnSphere( s, ( current_iteration * raytrace_depth ) ) - intersection_point );
		//	}
		//	else if ( s.type == CUBE ) {
		//		light_dir = glm::normalize( getRandomPointOnCube( s, ( current_iteration * raytrace_depth ) ) - intersection_point );
		//	}
		//	else if ( s.type == MESH ) {
		//		// TODO.
		//	}
		//	else {
		//		// ERROR: Unrecognized geometry type.
		//	}

		//	ray light_ray;
		//	light_ray.direction = light_dir;
		//	light_ray.origin = intersection_point;

		//	// Intersection testing.
		//	int light_id;
		//	bool did_hit_light = lightIntersection( light_ray, geoms, num_geoms, light_id );

		//	glm::vec3 intersection_normal_oriented = ( glm::dot( intersection_normal, r.direction ) < 0.0f ) ? intersection_normal : ( -1.0f * intersection_normal );

		//	if ( did_hit_light && light_id == i ) {
		//		e += f * ( l_mat.emittance * glm::dot( light_dir, intersection_normal_oriented ) );
		//	}
		//}

		r.color = r.color * rgb;		// Only add color contributions of this ray if it makes contact with a light source.
		r.origin = intersection_point;	// Set origin point for next ray.
		ray_pool[ray_pool_index] = r;	// Update ray in ray pool.
		return;
	}
	else if ( mat_type == PERFECT_SPUCULAR ) {
		r.direction = calculateReflectionDirection( intersection_normal, r.direction );	// Mirror surface contributes no color.
		r.origin = intersection_point;													// Set origin point for next ray.
		ray_pool[ray_pool_index] = r;													// Update ray in ray pool.
		return;
	}
	else if ( mat_type == GLASS ) {
		glm::vec3 intersection_normal_oriented = ( glm::dot( intersection_normal, r.direction ) < 0.0f ) ? intersection_normal : ( -1.0f * intersection_normal );
		bool ray_is_entering = ( glm::dot( intersection_normal, intersection_normal_oriented ) > 0.0f );

		const float IOR_GLASS = 1.5f;
		float ior_incident = ray_is_entering ? 1.0f : IOR_GLASS;
		float ior_transmitted = ray_is_entering ? IOR_GLASS : 1.0f;

		glm::vec3 refl_dir = calculateReflectionDirection( intersection_normal, r.direction );
		glm::vec3 trans_dir = calculateTransmissionDirection( intersection_normal, r.direction, ior_incident, ior_transmitted );

		Fresnel fres = calculateFresnel( intersection_normal, r.direction, ior_incident, ior_transmitted, refl_dir, trans_dir );

		glm::vec3 rand = generateRandomNumberFromThread( resolution, ( current_iteration * raytrace_depth ), r.image_coords.x, r.image_coords.y );

		if ( rand.x < fres.reflectionCoefficient ) {
			r.direction = refl_dir;
			r.origin = intersection_point;
			ray_pool[ray_pool_index] = r;
			return;
		}
		else {
			r.direction = trans_dir;
			r.origin = intersection_point;
			ray_pool[ray_pool_index] = r;
			return;
		}
	}
}


__host__
__device__
MaterialType determineMaterialType( material mat )
{
	if ( mat.hasRefractive ) {
		return GLASS;
	}
	else if ( mat.hasReflective ) {
		return PERFECT_SPUCULAR;
	}
	else {
		return IDEAL_DIFFUSE;
	}
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
					   int num_geoms,
					   simpleTexture *textures,
					   int num_textures )
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
		newStaticGeom.texture_id = geoms[i].texture_id;
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


	// TODO: The below computations for textures only need to be done once.

	// Create int *texture_info.
	int *texture_info = new int[num_textures * 3]; // Starting index, width, height.
	int starting_index = 0;
	for ( int i = 0; i < num_textures; ++i ) {
		int ti = i * 3;
		texture_info[ti] = starting_index;
		texture_info[ti + 1] = textures[i].width;
		texture_info[ti + 2] = textures[i].height;
		starting_index += ( textures[i].width * textures[i].height );
	}

	// Create glm::vec3 *texture_rgb.
	int num_texture_pixels = starting_index;
	glm::vec3 *texture_rgb = new glm::vec3[num_texture_pixels];
	int rgb_index = 0;
	for ( int i = 0; i < num_textures; ++i ) {
		for ( int y = 0; y < textures[i].height; ++y ) {
			for ( int x = 0; x < textures[i].width; ++x ) {
				texture_rgb[rgb_index] = ( textures[i].rgb[( y * textures[i].width ) + x] );
				++rgb_index;
			}
		}
	}

	// Send texture_info and texture_rgb to CUDA.
	int *cuda_texture_info = NULL;
	int size_texture_info = num_textures * 3 * sizeof( int );
	cudaMalloc( ( void** )&cuda_texture_info,
				size_texture_info );
	cudaMemcpy( cuda_texture_info,
				texture_info,
				size_texture_info,
				cudaMemcpyHostToDevice );

	glm::vec3 *cuda_texture_rgb = NULL;
	int size_texture_rgb = num_texture_pixels * sizeof( glm::vec3 );
	cudaMalloc( ( void** )&cuda_texture_rgb,
				size_texture_rgb );
	cudaMemcpy( cuda_texture_rgb,
				texture_rgb,
				size_texture_rgb,
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
																			v,
																			( float )current_iteration );

	// Launch raytraceRay kernel once per raytrace depth.
	for ( int i = 0; i < depth; ++i ) {
		dim3 threads_per_raytrace_block( raytrace_tile_size );
		dim3 blocks_per_raytrace_grid( ( int )ceil( ( float )num_rays / ( float )raytrace_tile_size ) );

		// Launch raytraceRay kernel.
		raytraceRay<<< blocks_per_raytrace_grid, threads_per_raytrace_block >>>( cuda_ray_pool,
																				 num_rays,
																				 render_cam->resolution,
																				 ( float )current_iteration,
																				 cam,
																				 ( i + 1 ),
																				 cuda_image,
																				 cuda_geoms,
																				 num_geoms,
																				 cuda_materials,
																				 cuda_texture_info,
																				 cuda_texture_rgb );

		//// Note: Stream compaction is slow.
		//thrust::device_ptr<ray> ray_pool_device_ptr( cuda_ray_pool );
		//thrust::device_ptr<ray> culled_ray_pool_device_ptr = thrust::remove_if( ray_pool_device_ptr,
		//																		ray_pool_device_ptr + num_rays,
		//																		RayIsInactive() );

		//// Compute number of active rays in ray pool.
		//num_rays = culled_ray_pool_device_ptr.get() - ray_pool_device_ptr.get();
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
	cudaFree( cuda_texture_info );
	cudaFree( cuda_texture_rgb );
	delete[] geom_list;
	delete[] texture_info;
	delete[] texture_rgb;

	// Make certain the kernel has completed.
	//cudaThreadSynchronize(); // Deprecated.
	cudaDeviceSynchronize();

	checkCUDAError( "Kernel failed!" );
}