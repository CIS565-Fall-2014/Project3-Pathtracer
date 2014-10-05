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

#define FAR_CLIPPING_DISTANCE 999999999.0f

// Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float triangleIntersectionTest(staticGeom geom, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ float intersectionTest(staticGeom * geoms,int num,ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

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

//intersection test against all geometries in the scene
__host__ __device__ float intersectionTest(staticGeom * geoms,int num, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, int & hitMaterialID, int & hitObjID)
{
	float hitDist(FAR_CLIPPING_DISTANCE);
	bool hitSomething = false;
	
	//loop through all geometries
	for(int i=0;i<num;i++)
	{
		glm::vec3 interPt(0.0f);
		glm::vec3 interNorm(0.0f);
		float d(0.0);
		if(geoms[i].type == SPHERE) d = sphereIntersectionTest(geoms[i], r,interPt, interNorm);
		else if(geoms[i].type == CUBE) d = boxIntersectionTest(geoms[i], r,interPt, interNorm);
		else if(geoms[i].type == MESH) d = triangleIntersectionTest(geoms[i], r,interPt, interNorm);

		//when hitting a surface that's closer than previous hit
		if(d > 0.0f && d < hitDist)
		{
			hitSomething = true;
			hitDist = d;

			intersectionPoint = interPt;
			normal = interNorm;
			hitMaterialID = geoms[i].materialid;
			hitObjID = i;
		}
	}

	return (hitSomething) ? hitDist: -1.0f;
}

//single geometry intersection test given geometry ID
__host__ __device__ float intersectionTest(staticGeom * geoms, int ID, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
	float hitDist(FAR_CLIPPING_DISTANCE);
	bool hitSomething = false;
	
	glm::vec3 interPt(0.0f);
	glm::vec3 interNorm(0.0f);
	float d(0.0f);

	if(geoms[ID].type == SPHERE) d = sphereIntersectionTest(geoms[ID], r,interPt, interNorm);
	else if(geoms[ID].type == CUBE) d = boxIntersectionTest(geoms[ID], r,interPt, interNorm);
	else if(geoms[ID].type == MESH) d = triangleIntersectionTest(geoms[ID], r,interPt, interNorm);

	if(d > 0.0f && d < hitDist)
	{
		hitSomething = true;
		hitDist = d;

		intersectionPoint = interPt;
		normal = interNorm;
	}

	return (hitSomething) ? hitDist: -1.0f;
}

//tirangle intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float triangleIntersectionTest(staticGeom geom, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction,0.0f)));
	//triangle tri = geom.m_triangle;
	triangle tri = geom.m_triangle;
	//transform ray
	ray rt;
	rt.origin = ro;
	rt.direction = rd;
	

	//plane intersection
	glm::vec3 U=tri.p1-tri.p0; glm::vec3 V=tri.p2-tri.p0; glm::vec3 N=glm::cross(U,V);
	float D = - glm::dot(tri.p1,N);
	float t =-(glm::dot(ro,N)+D)/(glm::dot(rd,N));
	if (t < 0.0f) return -1;
	glm::vec3 Intersect=ro+t*rd;

	//check whether inside
	glm::vec3 list[4]={tri.p0,tri.p1,tri.p2,tri.p0};
	glm::vec3 testU=tri.p1-tri.p0; glm::vec3 testV=tri.p2-tri.p1; glm::vec3 testN=glm::cross(U,V); glm::vec3 P2P0=ro-Intersect;
	bool counterClockwise=true;
	
	if(glm::dot(testN,P2P0)<0) counterClockwise=false;
	
	glm::vec3 V1,V2,N1;

	for (int i=0;i<3;++i)
	{
		V1=list[i]-ro;
		V2=list[i+1]-ro;
		N1=glm::cross(V2,V1);
		if(!counterClockwise) N1=glm::cross(V1,V2);
		if((glm::dot(Intersect,N1)-glm::dot(ro,N1))<-EPSILON) return -1;
	}

	intersectionPoint = multiplyMV(geom.transform,glm::vec4(Intersect,1.0f));
	normal = multiplyMV(geom.transform, glm::vec4(tri.normal,0.0f));

	return t;


	/* half plane method, had bug, not used
	//test if on inner of edge 1
	glm::vec3 edge1 = tri.p1 - tri.p0;
	if(glm::dot(glm::cross(edge1,Intersect - tri.p0),tri.normal) < 0.0f) return -1;

	//test if on inner of edge 2
	glm::vec3 edge2 = tri.p2 - tri.p1;
	if(glm::dot(glm::cross(edge2,Intersect - tri.p1),tri.normal) < 0.0f) return -1;

	//test if on inner of edge 3
	glm::vec3 edge3 = tri.p0 - tri.p2;
	if(glm::dot(glm::cross(edge3,Intersect - tri.p2),tri.normal) < 0.0f) return -1;

	intersectionPoint = multiplyMV(geom.transform,glm::vec4(Intersect,1.0f));
	normal = multiplyMV(geom.transform, glm::vec4(tri.normal,0.0f));

	return glm::length(intersectionPoint - r.origin);
	*/
}

// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  const float INFINITY_DISTANCE = 999999999.0f;

  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

  //transform ray
  ray rt;
  rt.origin = ro;
  rt.direction = rd;

  float Tnear(-INFINITY_DISTANCE) , Tfar(INFINITY_DISTANCE);
  //intersect X slab
  {
	  float dx = rd.x;
	  float ox = ro.x;
	  //parallel case
	  if(dx == 0.0f) 
	  {
		  if(abs(ox) > 0.5f) return -1;
	  }
	  float t1,t2;

	  t1 = (-0.5f - ox)/(dx);
	  t2 = (0.5f - ox)/(dx);

	  if(min(t1,t2)>Tnear) Tnear=min(t1,t2);

      if(max(t1,t2)<Tfar) Tfar=max(t1,t2);

	  if(Tnear>Tfar) return -1;
		
	  if(Tfar<0) return -1;

  }

  //intersect Y slab
  {
	  float dy = rd.y;
	  float oy = ro.y;
	  //parallel case
	  if(dy == 0.0f) 
	  {
		  if(abs(oy) > 0.5f) return -1;
	  }
	  float t1,t2;

	  t1 = (-0.5f - oy)/(dy);
	  t2 = (0.5f - oy)/(dy);

	  if(min(t1,t2)>Tnear) Tnear=min(t1,t2);

      if(max(t1,t2)<Tfar) Tfar=max(t1,t2);

	  if(Tnear>Tfar) return -1;
		
	  if(Tfar<0) return -1;

  }

  //intersect Z slab
  {
	  float dz = rd.z;
	  float oz = ro.z;
	  //parallel case
	  if(dz == 0.0f) 
	  {
		  if(abs(oz) > 0.5f) return -1;
	  }
	  float t1,t2;

	  t1 = (-0.5f - oz)/(dz);
	  t2 = (0.5f - oz)/(dz);

	  if(min(t1,t2)>Tnear) Tnear=min(t1,t2);

      if(max(t1,t2)<Tfar) Tfar=max(t1,t2);

	  if(Tnear>Tfar) return -1;
		
	  if(Tfar<0) return -1;

  }


  glm::vec3 intPt, intNorm;
  intPt = getPointOnRay(rt,Tnear); 
	
  if(intPt.y >= 0.5f - EPSILON) intNorm = glm::vec3(0.0f,1.0f,0.0f);
	
  else if(intPt.y <= -0.5f + EPSILON) intNorm = glm::vec3(0.0,-1.0f,0.0f);
	
  else if(intPt.x >= 0.5f - EPSILON) intNorm = glm::vec3(1.0f,0.0f,0.0f);
	
  else if(intPt.x <= -0.5f + EPSILON) intNorm = glm::vec3(-1.0f,0.0f,0.0f);
	
  else if(intPt.z >= 0.5f - EPSILON) intNorm = glm::vec3(0.0f,0.0f,1.0f);
	
  else if(intPt.z <= -0.5f + EPSILON) intNorm = glm::vec3(0.0f,0.0f,-1.0f);

  intersectionPoint = multiplyMV(box.transform, glm::vec4(intPt,1.0f));	
  normal = glm::normalize(multiplyMV(box.transform, glm::vec4(intNorm,0.0f)));
  return glm::length(intersectionPoint - r.origin);
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

// Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

  return glm::vec3(0,0,0);
}

#endif


