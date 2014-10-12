// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include <glm/glm.hpp>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "utilities.h"

// Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ cudaMat4 transposeM( cudaMat4 m );
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ void swapFloats( float &a, float &b );
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
__host__ __device__ glm::vec3 computeBoxObjectSpaceNormal( const glm::vec3 &object_space_intersection_point );


// Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int simpleHash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}


// Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b)) < EPSILON){
        return true;
    }else{
        return false;
    }
}


// Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t - .0001f) * glm::normalize(r.direction);
}


// LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
// This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
// Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}


__host__
__device__
cudaMat4 transposeM( cudaMat4 m )
{
	cudaMat4 transpose;
	transpose.x = glm::vec4( m.x.x, m.y.x, m.z.x, m.w.x );
	transpose.y = glm::vec4( m.x.y, m.y.y, m.z.y, m.w.y );
	transpose.z = glm::vec4( m.x.z, m.y.z, m.z.z, m.w.z );
	transpose.w = glm::vec4( m.x.w, m.y.w, m.z.w, m.w.w );
	return transpose;
}


// Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}


// Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}


// A swap method I can call from the GPU.
__host__
__device__
void swapFloats( float &a, float &b )
{
  float c = a;
  a = b;
  b = c;
}


// TODO: IMPLEMENT THIS FUNCTION
// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__
__device__
float boxIntersectionTest( staticGeom box,
						   ray r,
						   glm::vec3& intersectionPoint,
						   glm::vec3& normal )
{
	// Solution adapted from "Real-Time Rendering Third Edition" pg 742-744 by Akenine-Moller, Haines, and Hoffman.
	// Slab method for AABB and OBB.
	// Method assumes base side length of cube is 1.

	// Transform ray origin and direction to object space.
	glm::vec3 ro = multiplyMV( box.inverseTransform, glm::vec4( r.origin, 1.0f ) );
	glm::vec3 rd = glm::normalize( multiplyMV( box.inverseTransform, glm::vec4( r.direction, 0.0f ) ) );

	ray rt;
	rt.origin = ro;
	rt.direction = rd;

	// Cube center.
	glm::vec3 ac( 0.0f, 0.0f, 0.0f );

	// Normalized side vectors of cube.
	glm::vec3 ax( 1.0f, 0.0f, 0.0f );
	glm::vec3 ay( 0.0f, 1.0f, 0.0f );
	glm::vec3 az( 0.0f, 0.0f, 1.0f );

	// Cube half lengths.
	float hx = 0.5f;
	float hy = 0.5f;
	float hz = 0.5f;

	glm::vec3 side_vectors[3] = { ax, ay, az };
	float half_lengths[3] = { hx, hy, hz };

	float tmin = -FLT_MAX;
	float tmax = FLT_MAX;

	glm::vec3 p = ac - rt.origin;

	for ( int i = 0; i < 3; ++i ) {
		float e = glm::dot( side_vectors[i], p );
		float f = glm::dot( side_vectors[i], rt.direction );

		// Ray is not parallel to current slab.
		if ( abs( f ) > EPSILON ) {
			float t1 = ( ( e + half_lengths[i] ) / f );
			float t2 = ( ( e - half_lengths[i] ) / f );

			if ( t1 > t2 ) {
				swapFloats( t1, t2 );
			}

			if ( t1 > tmin ) {
				tmin = t1;
			}

			if ( t2 < tmax ) {
				tmax = t2;
			}

			// Ray misses box, so reject.
			if ( tmin > tmax ) {
				return -1.0f;
			}

			// Box is behind ray origin, so reject.
			if ( tmax < 0.0f ) {
				return -1.0f;
			}
		}
		// Ray is parallel to current slab and ray is outside current slab, so reject.
		else if ( -e - half_lengths[i] > 0.0f || -e + half_lengths[i] < 0.0f ) {
			return -1.0f;
		}
	}

	glm::vec3 object_space_intersection_point;
	glm::vec3 realIntersectionPoint;
	if ( tmin > 0.0f ) {
		object_space_intersection_point = getPointOnRay( rt, tmin );
		realIntersectionPoint = multiplyMV( box.transform, glm::vec4( object_space_intersection_point, 1.0f ) );
	}
	else {
		object_space_intersection_point = getPointOnRay( rt, tmax );
		realIntersectionPoint = multiplyMV( box.transform, glm::vec4( object_space_intersection_point, 1.0f ) );
	}

	intersectionPoint = realIntersectionPoint;

	glm::vec3 object_space_normal = computeBoxObjectSpaceNormal( object_space_intersection_point );
	normal = glm::normalize( multiplyMV( transposeM( box.inverseTransform ), glm::vec4( object_space_normal, 0.0f ) ) );

	return glm::length( r.origin - realIntersectionPoint );
}


__host__
__device__
glm::vec3 computeBoxObjectSpaceNormal( const glm::vec3 &object_space_intersection_point )
{
	const float INTERSECTION_THRESHOLD = 0.001f;

	// +x face.
	if ( object_space_intersection_point.x > 0.5f - INTERSECTION_THRESHOLD &&
		 object_space_intersection_point.x < 0.5f + INTERSECTION_THRESHOLD ) {
		return glm::vec3( 1.0f, 0.0f, 0.0f );
	}
	// -x face.
	else if ( object_space_intersection_point.x > -0.5f - INTERSECTION_THRESHOLD &&
			  object_space_intersection_point.x < -0.5f + INTERSECTION_THRESHOLD ) {
		return glm::vec3( -1.0f, 0.0f, 0.0f );
	}
	// +y face.
	else if ( object_space_intersection_point.y > 0.5f - INTERSECTION_THRESHOLD &&
			  object_space_intersection_point.y < 0.5f + INTERSECTION_THRESHOLD ) {
		return glm::vec3( 0.0f, 1.0f, 0.0f );
	}
	// -y face.
	else if ( object_space_intersection_point.y > -0.5f - INTERSECTION_THRESHOLD &&
			  object_space_intersection_point.y < -0.5f + INTERSECTION_THRESHOLD ) {
		return glm::vec3( 0.0f, -1.0f, 0.0f );
	}
	// +z face.
	else if ( object_space_intersection_point.z > 0.5f - INTERSECTION_THRESHOLD &&
			  object_space_intersection_point.z < 0.5f + INTERSECTION_THRESHOLD ) {
		return glm::vec3( 0.0f, 0.0f, 1.0f );
	}
	// -z face.
	else {
		return glm::vec3( 0.0f, 0.0f, -1.0f );
	}
}


// LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
// Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__
__device__
float sphereIntersectionTest( staticGeom sphere,
							  ray r,
							  glm::vec3& intersectionPoint,
							  glm::vec3& normal )
{
	float radius = 0.5f;

	// Transform ray origin and direction to object space.
	glm::vec3 ro = multiplyMV( sphere.inverseTransform, glm::vec4( r.origin, 1.0f ) );
	glm::vec3 rd = glm::normalize( multiplyMV( sphere.inverseTransform, glm::vec4( r.direction, 0.0f ) ) );

	ray rt;
	rt.origin = ro;
	rt.direction = rd;
  
	float vDotDirection = glm::dot( rt.origin, rt.direction );
	float radicand = vDotDirection * vDotDirection - ( glm::dot( rt.origin, rt.origin ) - pow( radius, 2.0f ) );
	if ( radicand < 0 ) {
		return -1.0f;
	}
  
	float squareRoot = sqrt( radicand );
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;
  
	float t = 0.0f;
	if ( t1 < 0.0f && t2 < 0.0f ) {
		return -1.0f;
	} else if ( t1 > 0.0f && t2 > 0.0f ) {
		t = min( t1, t2 );
	} else {
		t = max( t1, t2 );
	}

	glm::vec3 realIntersectionPoint = multiplyMV( sphere.transform, glm::vec4( getPointOnRay( rt, t ), 1.0f ) );
	glm::vec3 realOrigin = multiplyMV( sphere.transform, glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f ) );

	intersectionPoint = realIntersectionPoint;
	normal = glm::normalize( realIntersectionPoint - realOrigin );
        
	return glm::length( r.origin - realIntersectionPoint );
}


// Returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}


// LOOK: Example for generating a random point on an object using thrust.
// Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(simpleHash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    // Get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    // Pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        // x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        // x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        // y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        // y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        // x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        // x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}


// TODO: IMPLEMENT THIS FUNCTION
// Generates a random point on a given sphere
__host__
__device__
glm::vec3 getRandomPointOnSphere( staticGeom sphere,
								  float random_seed )
{
	// Generate two random numbers u and v (0, 1) using thrust.

    thrust::default_random_engine rng( simpleHash( random_seed ) );
    thrust::uniform_real_distribution<float> u01( 0.0f, 1.0f );

	// Compute spherical coordinates theta and phi.
	// theta = 2 * pi * u
	// phi = inverse_cos( 2 * v - 1 )

	float theta = 2.0f * PI * ( float )u01( rng );
	float phi = acos( 2.0f * ( float )u01( rng ) - 1.0f );

	// Convert spherical coordinates (theta and phi) into cartesian coordinates (x, y, z).
	// x = radius * sin( phi ) * cos( theta )
	// y = radius * sin( phi ) * sin( theta )
	// z = radius * cos( phi )

	float x = 0.5f * sin( phi ) * cos( theta );
	float y = 0.5f * sin( phi ) * sin( theta );
	float z = 0.5f * cos( phi );

	return multiplyMV( sphere.transform, glm::vec4( x, y, z, 1.0f) );
}


__host__
__device__
glm::vec2 computeSphereUVCoordinates( staticGeom sphere,
									  glm::vec3 intersection_point )
{
	// Thanks to https://www.cs.unc.edu/~rademach/xroads-RT/RTarticle.html#texturemap

	// Convert intersection point from world-space to object-space.
	glm::vec3 Vp = glm::normalize( multiplyMV( sphere.inverseTransform, glm::vec4( intersection_point, 1.0f ) ) );

	// Define unit vector pointing from the center of the sphere to the "north pole".
	glm::vec3 Vn( 0.0f, 1.0f, 0.0f );

	// Define unit vector pointing from the center of the sphere to any point on the equator.
	glm::vec3 Ve( 1.0f, 0.0f, 0.0f );

	// Compute angle between Vp and Vn, or the latitude.
	float phi = acos( -glm::dot( Vn, Vp ) );

	// Compute v, or the vertical image-space location [0, 1].
	float v = phi / PI;

	// Compute angle between Vp and Ve, or the longitude.
	//float theta = acos( glm::dot( Vp, Ve ) / sin( phi ) ) / TWO_PI;

	float theta;
	if ( sin( phi ) > -EPSILON && sin( phi ) < EPSILON ) {
		theta = 0.0f;
	}
	else {
		theta = acos( glm::dot( Vp, Ve ) / sin( phi ) ) / TWO_PI;
	}

	// Compute u, or the horizontal image location [0, 1]:
	float u;
	if ( glm::dot( glm::cross( Vn, Ve ), Vp ) > 0.0f ) {
		//u = theta;
		u = 1.0f - theta;
	}
	else {
		//u = 1.0f - theta;
		u = theta;
	}

	return glm::vec2( u, v );
}


__host__
__device__
glm::vec2 computeCubeUVCoordinates( staticGeom cube,
									glm::vec3 intersection_point )
{
	// Convert intersection point from world-space to object-space.
	glm::vec3 p = glm::normalize( multiplyMV( cube.inverseTransform, glm::vec4( intersection_point, 1.0f ) ) );

	// Compute face ( 1 => x, 2 => y, 3 => z ).
	int face = ( abs( p.x ) > abs( p.y ) && abs( p.x ) > abs( p.z ) ) ? 1 : ( abs( p.y ) > abs( p.z ) ) ? 2 : 3;
	face = ( p[face - 1] < 0 ) ? ( -1.0f * face  ) : face;

	float u, v;

	if ( face == 1 ) {			// +x
		u = ( ( p.z / abs( p.x ) ) + 1.0f ) / 2.0f;
		v = ( ( p.y / abs( p.x ) ) + 1.0f ) / 2.0f;
	}
	else if ( face == -1 ) {	// -x
		u = ( ( -p.z / abs( p.x ) ) + 1.0f ) / 2.0f;
		v = ( ( p.y / abs( p.x ) ) + 1.0f ) / 2.0f;
	}
	else if ( face == 2 ) {		// +y
		u = ( ( p.x / abs( p.y ) ) + 1.0f ) / 2.0f;
		v = ( ( p.z / abs( p.y ) ) + 1.0f ) / 2.0f;
	}
	else if ( face == -2 ) {	// -y
		u = ( ( p.x / abs( p.y ) ) + 1.0f ) / 2.0f;
		v = ( ( -p.z / abs( p.y ) ) + 1.0f ) / 2.0f;
	}
	else if ( face == 3 ) {		// +z
		u = ( ( p.x / abs( p.z ) ) + 1.0f ) / 2.0f;
		v = ( ( p.y / abs( p.z ) ) + 1.0f ) / 2.0f;
	}
	else if ( face == -3 ) {	// -z
		u = ( ( -p.x / abs( p.z ) ) + 1.0f ) / 2.0f;
		v = ( ( p.y / abs( p.z ) ) + 1.0f ) / 2.0f;
	}

	return glm::vec2( u, v );
}


#endif