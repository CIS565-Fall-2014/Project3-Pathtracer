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
__host__ __device__ cudaMat4 transpose(cudaMat4 mat){
	glm::mat4 mm=glm::transpose(glm::mat4(mat.x,mat.y,mat.z,mat.w));
	mat.x=mm[0];
	mat.y=mm[1];
	mat.z=mm[2];
	mat.w=mm[3];

	return mat;
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

__host__ __device__ bool rayBoxIntersect(ray r, float t[2])
{
	float t_min,t_max;

	t[0] = FLT_MIN;//t_near

	t[1] = FLT_MAX;//t_far

	glm::vec3 min_box = glm::vec3(-.5,-.5,-.5);
	glm::vec3 max_box = glm::vec3(.5,.5,.5);

	bool intersectFlag = true;

	for(int i = 0;i<3;i++){
		if(r.direction[i] == 0){
			if(r.origin[i] < min_box[i] || r.origin[i] >max_box[i]){
				intersectFlag = false;
			}
		}
		else{
			t_min = (min_box[i] - r.origin[i]) / r.direction[i];
			t_max = (max_box[i] - r.origin[i]) / r.direction[i];
			
			if(t_min > t_max){
				float temp = t_min;
				t_min = t_max;
				t_max = temp;
			}
			if(t_min > t[0])
				t[0] = t_min;
			if(t_max < t[1]){
				t[1] = t_max;
			}
			if(t[0]>t[1]){				
				intersectFlag = false;
			}					
			if(t[1]<0){				
				intersectFlag = false;
			}
		}
	}
	return intersectFlag;
}

__host__ __device__ glm::vec3 getUnitBoxNommal(glm::vec3& p)
{
	glm::vec3 normal;
	float eps = 0.001f;
	if(abs(p.x - 0.5) < eps)
		normal = glm::vec3(1,0,0);
	else if(abs(p.x + 0.5) < eps){
		normal = glm::vec3(-1,0,0);
	}
	else if(abs(p.y - 0.5) < eps){
		normal = glm::vec3(0,1,0);
	}
	else if(abs(p.y + 0.5) < eps){
		normal = glm::vec3(0,-1,0);		
	}
	else if(abs(p.z - 0.5) < eps){
		normal = glm::vec3(0,0,1);
	}
	else if(abs(p.z + 0.5) < eps){
		normal = glm::vec3(0,0,-1);
	}
	return normal;
}

// TODO: IMPLEMENT THIS FUNCTION
// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));

	ray rt; rt.origin = ro; rt.direction = rd;

	float t[2] = {FLT_MIN, FLT_MAX};
	//float t[2] = {0};
	bool isIntersect = rayBoxIntersect(rt, t);

	float ti;
	if(isIntersect)
	{			
		if(t[0]<=0){
			ti = t[1];			
		}
		else
			ti = t[0];		


		glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, ti), 1.0));
		glm::vec3 realOrigin = multiplyMV(box.transform, glm::vec4(0,0,0,1));

		glm::vec3 unitNormal = getUnitBoxNommal(getPointOnRay(rt, ti));				

		intersectionPoint = realIntersectionPoint;				

		glm::vec3 realNormal = multiplyMV(box.transform, glm::vec4(unitNormal, 1));
		
		normal = glm::normalize(realNormal - realOrigin);

		//normal = glm::normalize(realOrigin - realNormal);
			
		return glm::length(r.origin - realIntersectionPoint);
	}
	else
		return -1;

	//glm::vec3 newP0 = multiplyMV(box.inverseTransform, glm::vec4(r.origin,1.0f));
	//glm::vec3 newV0 = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction,0.0f)));
	//if (newV0[0] == 0 && newV0[1] == 0 && newV0[2] == 0) {
	//		return -1.0f;
	//}
	//ray rt; rt.origin = newP0; rt.direction = newV0;
	//for (int i = 0; i < 3; i++) {
	//		if (newP0[i] > .5f && newV0[i] > 0)
	//			return -1;
	//		if (newP0[i] < -.5f && newV0[i] < 0)
	//			return -1;
	//	}

	//	glm::vec3 face(0.0f);
	//	for (int i = 0; i < 3; i++) {
	//		if (newV0[i] > 0) {
	//			if (newP0[i] >= -.5f)
	//				face[i] = .5f;
	//			else
	//				face[i] = -.5f;
	//		} else {
	//			if (newP0[i] <= .5f)
	//				face[i] = -.5f;
	//			else
	//				face[i] = .5f;
	//		}
	//	}
	//	glm::vec3 t(-1.0f,-1.0f,-1.0f);
	//	glm::vec3 IntNom(0.0f);
	//	float minT = -1.0f;
	//	float tmp1 = 0.0f, tmp2 = 0.0f;
	//	for (int i = 0; i < 3; i++) {
	//		if (newV0[i] == 0.0f)
	//			continue;
	//		t[i] = (face[i] - newP0[i]) / newV0[i];
	//		if (t[i] > 0) {
	//			tmp1 = newP0[(i + 1) % 3] + t[i] * newV0[(i + 1) % 3];
	//			tmp2 = newP0[(i + 2) % 3] + t[i] * newV0[(i + 2) % 3];
	//		}
	//		if (!(tmp1 >= -.5f && tmp1 <= .5f && tmp2 >= -.5f && tmp2 <= .5f))
	//			t[i] = -1.0f;
	//	}
	//	minT = -1.0f;
	//	for (int i = 0; i < 3; i++) {
	//		if (minT <= 0.0f) {
	//			if (t[i] > 0.0f) {
	//				minT = t[i];
	//				IntNom[i] = 2.0f*face[i];
	//				IntNom[(i + 1) % 3] = 0.0f;
	//				IntNom[(i + 2) % 3] = 0.0f;
	//			}
	//		} 
	//		else if (t[i] <= 0.0f)
	//			continue;
	//		else {
	//			if (t[i] < minT) {
	//				minT = t[i];
	//				IntNom[i] = 2.0f*face[i];
	//				IntNom[(i + 1) % 3] = 0.0f;
	//				IntNom[(i + 2) % 3] = 0.0f;
	//			}
	//		}

	//	}
	//if(minT<0) return minT;

	//glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, minT), 1.0f));
	//glm::vec3 realOrigin = multiplyMV(box.transform, glm::vec4(0,0,0,1.0f));

	//intersectionPoint = realIntersectionPoint;
	////normal = glm::normalize(realIntersectionPoint - realOrigin);
	//normal=multiplyMV(transpose(box.inverseTransform),glm::vec4(IntNom,1.0f));
	//return minT;
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

// TODO: IMPLEMENT THIS FUNCTION
// Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){
	//Marsaglia (1972) uniform distribution of points on sphere
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u1(-1.0f,1.0f);
    //thrust::uniform_real_distribution<float> u02(0,2.0f);
	float x1=1.0f;
	float x2=1.0f;
	while(x1*x1+x2*x2>=1.0f){
		x1=u1(rng);
		x2=u1(rng);
	}

	return multiplyMV(sphere.transform,glm::vec4(x1*sqrt(1-x1*x1-x2*x2),x2*sqrt(1-x1*x1-x2*x2),.5f-(x1*x1+x2*x2),1.0));
}

__host__ __device__ glm::vec3 getRandomPointOnSphere(float randomSeed){
	//Marsaglia (1972) uniform distribution of points on sphere
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u1(-1.0f,1.0f);
    //thrust::uniform_real_distribution<float> u02(0,2.0f);
	float x1=1.0f;
	float x2=1.0f;
	while(x1*x1+x2*x2>=1.0f){
		x1=u1(rng);
		x2=u1(rng);
	}

	return glm::vec3(x1*sqrt(1-x1*x1-x2*x2),x2*sqrt(1-x1*x1-x2*x2),.5f-(x1*x1+x2*x2));
}
#endif


