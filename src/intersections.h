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
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
//Added
__host__ __device__ bool Intersecttest(ray r, glm::vec3& intersectp, glm::vec3& intersectn, staticGeom* geoms, int numberOfGeoms, int& geomId);
__host__ __device__ bool epsilonCheck(float a, float b);
__host__ __device__ float triIntersectionTest(staticGeom mesh, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ bool BBIntersectionTest(staticGeom cube, ray r);

__host__ __device__ bool Intersecttest(ray r, glm::vec3& intersectp, glm::vec3& intersectn, staticGeom* geoms, int numberOfGeoms, int& geomId)
{
	float tempdist = -1.0f,dist = FLT_MAX;
	glm::vec3 temp_intersectp, temp_intersectn;	
	bool intersected = false;

	bool inbb = false;
	for(int i = 0; i < numberOfGeoms; i++){
		if(geoms[i].type==SPHERE)
			tempdist = sphereIntersectionTest(geoms[i], r, temp_intersectp,temp_intersectn);
		else if (geoms[i].type==CUBE)
			tempdist = boxIntersectionTest(geoms[i], r, temp_intersectp,temp_intersectn);
		else if(geoms[i].type==MESH)
		{
			//Judge whether intersect with bounding box
			if(inbb == false)
				inbb = BBIntersectionTest(geoms[i],r);

			if(inbb == true)
			    tempdist = triIntersectionTest(geoms[i], r, temp_intersectp,temp_intersectn);
			else
				i = i + geoms[i].trinum -1; //if not intersected go to next geometry
		}
		
		if(!epsilonCheck(tempdist, -1.0f)&&dist>tempdist)
		{
			dist = tempdist;
			intersectn = temp_intersectn;
			intersectp = temp_intersectp;
			geomId = i;
			intersected = true;
		}
	}

	return intersected;
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

// Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

// Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

// TODO: IMPLEMENT THIS FUNCTION
// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	float sign=1.0f;
	if(abs(rt.origin.x)-0.5<0&&abs(rt.origin.y)-0.5<0&&abs(rt.origin.z)-0.5<0)
		sign=-1.0f;

	double tnear = -999999;
	double tfar = 999999;
	double t1,t2,temp,t;
	for (int i = 0; i < 3; i++) {
		if (rd[i] ==0 ) {
			if (ro[i] > 0.5 || ro[i] < -0.5) {
				return -1;
			}
		}
		t1 = (-0.5 - ro[i])/rd[i];
		t2 = (0.5 - ro[i])/rd[i];
		if (t1 > t2) {
			temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if (t1 > tnear) {
			tnear = t1;
		}
		if (t2 < tfar) {
			tfar = t2;
		}
		if (tnear > tfar) {
			return -1;
		}
		if (tfar < 0) {
			return -1;
		}
	}

	if (tnear < -0.0001) 
		t=tfar;
	else
		t=tnear;


	glm::vec3 P = getPointOnRay(rt, t);
	if(abs(P[0]-0.5)<0.001)
		normal = glm::vec3(1,0,0);
	else if(abs(P[0]+0.5)<0.001)
		normal = glm::vec3(-1,0,0);
	else if(abs(P[1]-0.5)<0.001)
		normal = glm::vec3(0,1,0);
	else if(abs(P[1]+0.5)<0.001)
		normal = glm::vec3(0,-1,0);
	else if(abs(P[2]-0.5)<0.001)
		normal = glm::vec3(0,0,1);
	else if(abs(P[2]+0.5)<0.001)
		normal = glm::vec3(0,0,-1);

	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(P, 1.0));

	intersectionPoint = realIntersectionPoint;
	normal = glm::normalize(sign * multiplyMV(box.transform, glm::vec4(normal,0)));

	return glm::length(r.origin - realIntersectionPoint);
}

// LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
// Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
	float radius = .5;

	glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

	float sign=1.0f;
	if(sqrt(glm::dot(ro,ro))<radius)
		sign=-1.0f;

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
	normal = glm::normalize(sign*(realIntersectionPoint - realOrigin));

	return glm::length(r.origin - realIntersectionPoint);
}


//Triangle intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float triIntersectionTest(staticGeom mesh, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	glm::vec3 p = glm::vec3(rt.origin.x, rt.origin.y, rt.origin.z);
	glm::vec3 d = glm::vec3(rt.direction.x, rt.direction.y, rt.direction.z);
	glm::vec3 v0 = mesh.tri.p1;
	glm::vec3 v1 = mesh.tri.p2;
	glm::vec3 v2 = mesh.tri.p3;
	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;

	glm::vec3 h = glm::cross(d, e2);
	float a = glm::dot(e1, h);

	if (a > -0.00001 && a < 0.00001) {
		return -1;
	}

	double f = 1.0/a;
	glm::vec3 s = p - v0;
	double u = f * (glm::dot(s, h));
	if (u < -0.00001 || u > 1.00001) {
		return -1;
	}

	glm::vec3 q = glm::cross(s, e1);
	double v = f * (glm::dot(d, q));

	if (v < -0.00001 || u + v > 1.00001) {
		return -1;
	}

	double t = f * glm::dot(e2, q);

	if (t > 0.001) {
		glm::vec3 realIntersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
		intersectionPoint = realIntersectionPoint;

		normal = glm::normalize(multiplyMV(mesh.transinverseTransform, glm::vec4(mesh.tri.normal,0)));
		if(glm::dot(r.direction,normal) > 0) 
			normal = -1.0f * normal;
		return t;
	}

	return -1;
}

//Accelerate OBJ scan
__host__ __device__ bool BBIntersectionTest(staticGeom box, ray r){

	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	//The init size of objs are set to be smaller than 1*1*1,
	//thus they must in area (-0.5,-0.5,-0.5) to (0.5,0.5,0.5) 
	double tnear = -999999;
	double tfar = 999999;
	double t1,t2,temp,t;
	for (int i = 0; i < 3; i++) {
		if (rd[i] ==0 ) {
			if (ro[i] > 0.5 || ro[i] < -0.5) {
				return false;
			}
		}
		t1 = (-0.5 - ro[i])/rd[i];
		t2 = (0.5 - ro[i])/rd[i];
		if (t1 > t2) {
			temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if (t1 > tnear) {
			tnear = t1;
		}
		if (t2 < tfar) {
			tfar = t2;
		}
		if (tnear > tfar) {
			return false;
		}
		if (tfar < 0) {
			return false;
		}
	}

	if (tnear < -0.0001) 
		t=tfar;
	else
		t=tnear;

	return true;
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

// TODO: IMPLEMENT THIS FUNCTION
// Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

	thrust::default_random_engine rng(hash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,180);
	thrust::uniform_real_distribution<float> u02(0,360);

	float radius = .5f;

	glm::vec3 point;

	float theta, phi;
	theta = glm::radians((float)u01(rng));
	phi = glm::radians((float)u02(rng));

	point.x = radius * sin(theta) * cos(phi);
	point.y = radius * sin(theta) * sin(phi);
	point.z = radius * cos(theta);

	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));
    return randPoint;
}

#endif


