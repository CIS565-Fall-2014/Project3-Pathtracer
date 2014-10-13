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
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);

__host__ __device__ float planeIntersectionTest(ray r, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3& intersectionPoint, glm::vec3& real_normal);

__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);


// stuff from utilityCore has been copied here because of some weird CUDA issues
__host__ __device__ cudaMat4 MYglmMat4ToCudaMat4(glm::mat4 a){
    cudaMat4 m; a = glm::transpose(a);
    m.x = a[0];
    m.y = a[1];
    m.z = a[2];
    m.w = a[3];
    return m;
}
__host__ __device__ glm::mat4 MYcudaMat4ToGlmMat4(cudaMat4 a){
    glm::mat4 m;
    m[0] = a.x;
    m[1] = a.y;
    m[2] = a.z;
    m[3] = a.w;
    return glm::transpose(m);
}

// Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
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
    if(fabs(a-b) < EPSILON){ // changed this to make it correct
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

// Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

// Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

// TODO-[DONE]: IMPLEMENT THIS FUNCTION
// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection

// METHOD, based on sphereIntersectionTest below
/*
     - transform the ray into normalized space (this places a unit cube at the origin)
	 - attempt an intersect with each of the 6 faces and find the shortest
	 - transfer everything back into world coordinates (make sure to transfer the normal correctly!)
	 - use original r and realWorldIntersectionPoint to calculate the right distance to return
*/
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	glm::vec3 r_norm_origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));

	glm::vec3 r_norm_direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	double min_t = -1.0;
	bool min_t_set = false;

	ray r_norm; r_norm.origin = r_norm_origin; r_norm.direction = r_norm_direction;

	glm::vec3 isectpt(0,0,0);

	float isectdist;
	glm::vec3 real_normal(0,0,0);

	//x = -.5
	// if the result of planeIntersectionTest is NOT -1 ...
	if (!epsilonCheck(
		isectdist = planeIntersectionTest(r_norm, glm::vec3(-.5,0,0), glm::vec3(-.5,0,1), glm::vec3(-.5,1,0), isectpt, real_normal),
		-1.f)) {
		if (isectpt.y <= .5 + EPSILON && isectpt.y >= -.5 - EPSILON && isectpt.z <= .5 + EPSILON && isectpt.z >= -.5 - EPSILON) {
			if (!min_t_set) {
				min_t_set = true;
				min_t = isectdist;
				//norm_normal = glm::vec3(-1,0,0);
			}
			else {
				//if (isectdist < min_t) norm_normal = glm::vec3(-1,0,0);
				min_t = (isectdist < min_t ? isectdist : min_t);
			}
		}
	}
	//x = .5
	if (!epsilonCheck(
		isectdist = planeIntersectionTest(r_norm, glm::vec3(.5,0,0), glm::vec3(.5,0,1), glm::vec3(.5,1,0), isectpt, real_normal),
		-1.f)) {
		if (isectpt.y <= .5 + EPSILON && isectpt.y >= -.5 - EPSILON && isectpt.z <= .5 + EPSILON && isectpt.z >= -.5 - EPSILON) {
			if (!min_t_set) {
				min_t_set = true;
				min_t = isectdist;
				//norm_normal = glm::vec3(1,0,0);
			}
			else {
				//if (isectdist < min_t) norm_normal = glm::vec3(1,0,0);
				min_t = (isectdist < min_t ? isectdist : min_t);
			}
		}
	}
	//y = -.5
	if (!epsilonCheck(
		isectdist = planeIntersectionTest(r_norm, glm::vec3(0,-.5,0), glm::vec3(0,-.5,1), glm::vec3(1,-.5,0), isectpt, real_normal),
		-1.f)) {
		if (isectpt.x <= .5 + EPSILON && isectpt.x >= -.5 - EPSILON && isectpt.z <= .5 + EPSILON && isectpt.z >= -.5 - EPSILON) {
			if (!min_t_set) {
				min_t_set = true;
				min_t = isectdist;
				//norm_normal = glm::vec3(0,-1,0);
			}
			else {
				//if (isectdist < min_t) norm_normal = glm::vec3(0,-1,0);
				min_t = (isectdist < min_t ? isectdist : min_t);
			}
		}
	}
	//y = .5
	if (!epsilonCheck(
		isectdist = planeIntersectionTest(r_norm, glm::vec3(0,.5,0), glm::vec3(0,.5,1), glm::vec3(1,.5,0), isectpt, real_normal),
		-1.f)) {
		if (isectpt.x <= .5 + EPSILON && isectpt.x >= -.5 - EPSILON && isectpt.z <= .5 + EPSILON && isectpt.z >= -.5 - EPSILON) {
			if (!min_t_set) {
				min_t_set = true;
				min_t = isectdist;
				//norm_normal = glm::vec3(0,1,0);
			}
			else {
				//if (isectdist < min_t) norm_normal = glm::vec3(0,1,0);
				min_t = (isectdist < min_t ? isectdist : min_t);
			}
		}
	}
	//z = -.5
	if (!epsilonCheck(
		isectdist = planeIntersectionTest(r_norm, glm::vec3(0,0,-.5), glm::vec3(0,1,-.5), glm::vec3(1,0,-.5), isectpt, real_normal),
		-1.f)) {
		if (isectpt.x <= .5 + EPSILON && isectpt.x >= -.5 - EPSILON && isectpt.y <= .5 + EPSILON && isectpt.y >= -.5 - EPSILON) {
			if (!min_t_set) {
				min_t_set = true;
				min_t = isectdist;
				//norm_normal = glm::vec3(0,0,-1);
			}
			else {
				//if (isectdist < min_t) norm_normal = glm::vec3(0,0,-1);
				min_t = (isectdist < min_t ? isectdist : min_t);
			}
			
		}
	}
	//z = .5
	if (!epsilonCheck(
		isectdist = planeIntersectionTest(r_norm, glm::vec3(0,0,.5), glm::vec3(0,1,.5), glm::vec3(1,0,.5), isectpt, real_normal),
		-1.f)) {
		if (isectpt.x <= .5 + EPSILON && isectpt.x >= -.5 - EPSILON && isectpt.y <= .5 + EPSILON && isectpt.y >= -.5 - EPSILON) {
			if (!min_t_set) {
				min_t_set = true;
				min_t = isectdist;
				//norm_normal = glm::vec3(0,0,1);
			}
			else {
				//if (isectdist < min_t) norm_normal = glm::vec3(0,0,1);
				min_t = (isectdist < min_t ? isectdist : min_t);
			}
		}
	}

	// no intersection
	if (!min_t_set) return -1;

	// distance to the cube where ray, box, and distance are ALL normed
	isectdist = min_t;

	// calculate world pt and normal
	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(r_norm, isectdist), 1.0));

	//cudaMat4 C = box.transform;
	// I want (C-inverse)-transpose
	//cudaMat4 Cit = MYglmMat4ToCudaMat4(glm::transpose(glm::inverse(MYcudaMat4ToGlmMat4(C))));

	//glm::vec3 realNormal = glm::normalize(multiplyMV(Cit, glm::vec4(norm_normal, 0.0))); // as specified on slide 793 in the FALL 2013 notes for CIS 560
																						 // since the normal is a vector, its w-coordinate is 0

	// use references to update intersection point and normal
	intersectionPoint = realIntersectionPoint;
	// get the right normal (a plane intersection could have 2)
	if (glm::dot(r.direction, real_normal) > 0) {
		//normal and direction are NOT antiparallel; this is wrong
		normal = -1.f * real_normal;
	}
	else {
		normal = real_normal;
	}

	normal = glm::normalize(normal);

	return glm::length(r.origin - realIntersectionPoint); //return the distance, WORLD to WORLD

}

// A helper function that intersects a ray with a plane. This function does NOT perform
// any transformations, nor does it have anything to do with normals. p1-p3 are 3 points
// which define a plane. They must not be collinear.
// returns -1 for no intersection, otherwise, distance to intersection. The point can be found in intersectionPoint
__host__ __device__ float planeIntersectionTest(ray r, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3& intersectionPoint, glm::vec3& realNormal){

	//vectors along the plane
	glm::vec3 vs1 = p2-p1;
	glm::vec3 vs2 = p3-p1;

	glm::vec3 n = glm::normalize(glm::cross(vs1, vs2));

	realNormal = n;

	float A = n.x;
	float B = n.y;
	float C = n.z;
	float D = -1 * (A*p1.x + B*p1.y + C*p1.z);


	float tDenom = glm::dot(n, r.direction);

	if (epsilonCheck(abs(tDenom), 0)) return -1;

	float tNumer = -1 * (glm::dot(n, r.origin) + D);

	float t = tNumer/tDenom;

	if (t < 0.0 && t > -1 * EPSILON) t = 0.0;

	if (t >= 0.0) {

		intersectionPoint = r.origin + t * r.direction;

		return t;
	}
	else {
		return -1;
	}
}

// LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
// Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
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

    thrust::default_random_engine rng(hash(randomSeed));
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

// TODO-[DONE]: IMPLEMENT THIS FUNCTION
// Generates a random point on a given sphere

// METHOD, based on getRandomPointOnCube above
/*
     - get u,v
	 - get theta, phi
	 - initialize vec3 "point"
	 - generate a point on the unit sphere
	 - return multiplyMV(sphere.transform, vec4(point, 1)
*/
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
	
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> r01(0,1);

	// u, v
	float u = (float) r01(rng);
	float v = (float) r01(rng);

	// theta, phi
	float theta = 2. * PI * u;
	float phi = acos(2. * v - 1.);

	// calculate cartesian coords
	float x = sin(phi) * cos(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(phi);

	// transform back into world space and return
	glm::vec3 point(x,y,z);

	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));

    return randPoint;
}

#endif


