// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include <glm/glm.hpp>
#include <thrust/random.h>
#include <math.h>
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
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed);

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

//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
	ray rt;
	
	glm::vec3 bounds[2];
	if(box.type == MESH){
		bounds[0] = box.boundingBoxMin;
		bounds[1] = box.boundingBoxMax;
		rt.origin = r.origin;
		rt.direction = r.direction;   //already transformed
	}
	else{
		rt.origin = multiplyMV ( box.inverseTransform, glm::vec4( r.origin, 1.0f ));
		rt.direction  = glm::normalize( multiplyMV ( box.inverseTransform, glm::vec4( r.direction, 0.0f )) );
		bounds[0] = glm::vec3(-0.5f, -0.5f, -0.5f);
		bounds[1] = glm::vec3(0.5f, 0.5f, 0.5f);
	}
		
	glm::vec3 invDir = getInverseDirectionOfRay(rt);
	int sign[3];
	sign[0] = (invDir.x < 0);
	sign[1] = (invDir.y < 0);
	sign[2] = (invDir.z < 0);

	double tmin, tmax, tymin, tymax, tzmin, tzmax, t;
	tmin = (bounds[sign[0]].x - rt.origin.x) * invDir.x;
	tmax = (bounds[1-sign[0]].x - rt.origin.x) * invDir.x;
	tymin = (bounds[sign[1]].y - rt.origin.y) * invDir.y;
	tymax = (bounds[1-sign[1]].y - rt.origin.y) * invDir.y;

	if ((tmin > tymax) || (tymin > tmax))
		return -1.0f;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	tzmin = (bounds[(int)sign[2]].z - rt.origin.z) * invDir.z;
	tzmax = (bounds[1-(int)sign[2]].z - rt.origin.z) * invDir.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return -1.0f;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	if (tmax < 0.0) 
		return  -1.0f; 
	if (tmin < 0.0) 
		t = tmax; // inside
	else 
		t = tmin; //outside
	
	glm::vec3 intersect_os = float(t) * rt.direction  + rt.origin;
	glm::vec3 tempNormal;
	if( fabs(intersect_os.x - bounds[0].x) < EPSILON) 
		tempNormal = glm::vec3(-1.0f, 0.0f, 0.0f);
	else if( fabs( intersect_os.x - bounds[1].x) < EPSILON) 
		tempNormal = glm::vec3(1.0f, 0.0f, 0.0f);
	else if( fabs( intersect_os.y - bounds[0].y) < EPSILON) 
		tempNormal = glm::vec3(0.0f, -1.0f, 0.0f);
	else if( fabs( intersect_os.y - bounds[1].y) < EPSILON) 
		tempNormal = glm::vec3(0.0f, 1.0f, 0.0f);
	else if( fabs( intersect_os.z - bounds[0].z) < EPSILON) 
		tempNormal = glm::vec3(0.0f, 0.0f, -1.0f);
	else if( fabs( intersect_os.z - bounds[1].z) < EPSILON) 
		tempNormal = glm::vec3(0.0f, 0.0f, 1.0f);

	intersectionPoint = multiplyMV(box.transform, glm::vec4(intersect_os, 1.0f));

	normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tempNormal, 0.0f)));

	return glm::length(r.origin - intersectionPoint);

	//a slow traditional version of box intersection test
	/*cudaMat4 Ti = box.inverseTransform;
	glm::vec3 R0 = multiplyMV(Ti, glm::vec4(r.origin, 1.0));
	float l = glm::length(r.direction);
	glm::vec3 Rd = multiplyMV(Ti, glm::vec4(glm::normalize(r.direction), 0.0));
	double tnear = -10000000;
	double tfar = 10000000;
	int slab = 0;
	double t, t1, t2;
	while(slab < 3){
		if(Rd[slab] == 0){
			if(R0[slab] > .5 || R0[slab] < -.5){
				return -1;
			}
		}
		t1 = (-.5 - R0[slab]) / Rd[slab];
		t2 = (.5 - R0[slab]) / Rd[slab];
		if(t1 > t2){
			double temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if(t1 > tnear) tnear = t1;
		if(t2 < tfar) tfar = t2;
		if(tnear > tfar){
			return -1;
		}
		if(tfar < 0){
			return -1;
		}
		slab++;
	}

	if(tnear > -.0001) t = tnear;
	else t = tfar;

	glm::vec3 p = R0 + (float)t * Rd;
	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(p, 1.0));
	intersectionPoint = realIntersectionPoint;

	glm::vec4 temp_normal;
	if(abs(p[0] - .5) < .001){
		temp_normal = glm::vec4(1,0,0,0);
	}else if(abs(p[0] + .5) < .001){
		temp_normal = glm::vec4(-1,0,0,0);
	}else if(abs(p[1] - .5) < .001){
		temp_normal = glm::vec4(0,1,0,0);
	}else if(abs(p[1] + .5) < .001){
		temp_normal = glm::vec4(0,-1,0,0);
	}else if(abs(p[2] - .5) < .001){
		temp_normal = glm::vec4(0,0,1,0);
	}else if(abs(p[2] + .5) < .001){
		temp_normal = glm::vec4(0,0,-1,0);
	}
	normal = glm::normalize(multiplyMV(box.transform, temp_normal));
        
	return glm::length(r.origin - realIntersectionPoint);*/

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

__host__ __device__ float triangleIntersectionTest(triangle& tri, ray rt, glm::vec3& intersectionPoint, glm::vec3& normal){

	glm::vec3 p1 = tri.p1;
	glm::vec3 p2 = tri.p2;
	glm::vec3 p3 = tri.p3;
	glm::vec3 edge1 = p2 - p1;
    glm::vec3 edge2 = p3 - p1;
	glm::vec3 n = glm::cross(edge1, edge2); 

    if (glm::length(n) < EPSILON)
        return -1;   

	glm::vec3 w0 = rt.origin - p1; 
	float a = - glm::dot(n, w0);
	float b = glm::dot(n, rt.direction);
	if (fabs(b) < EPSILON) 
		return -1;
    if (fabs(a) < EPSILON) 	      
		return -1; 

	float L = a/b;
    if (L < 0)
        return -1; 

	glm::vec3 p = rt.origin + L*rt.direction; 

	float e11 = glm::dot(edge1, edge1);
    float e12 = glm::dot(edge1, edge2);
    float e22 = glm::dot(edge2, edge2);
	glm::vec3 w = p - p1;
    float we1 = glm::dot(w, edge1);	
    float we2 = glm::dot(w, edge2);

	float D = e12 * e12 - e11 * e22;
    float s = (e12 * we2 - e22 * we1) / D;
    float t = (e12 * we1 - e11 * we2) / D;

	if(s < 0 || s > 1 || t < 0 || (s + t) > 1) 
        return -1;
	
	normal = glm::normalize(n);
	intersectionPoint = p;
	return L;
   
}

__host__ __device__ float polygonIntersectionTest(staticGeom polygon, ray r, glm::vec3& intersectionPoint, glm::vec3& normal, triangle * cudatris){
	
	glm::vec3 P = multiplyMV (polygon.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 V = glm::normalize( multiplyMV (polygon.inverseTransform, glm::vec4(r.direction, 0.0f)));
	ray rt; rt.origin = P; rt.direction = V; 
	float t = -1, tMin = -2;
	glm::vec3 localIntersect, localNormal;
	if(boxIntersectionTest(polygon, rt, intersectionPoint, normal) > 0){
		glm::vec3 tempIntersect, tempNormal; 
		
		for (int i=0; i<polygon.numOfTris; i++){ 
			t = triangleIntersectionTest(cudatris[i], rt, tempIntersect, tempNormal);
			if( (tMin < 0 && t > -0.5f ) || ( tMin > -1 && t < tMin && t > -0.5f ) ){
				tMin = t;   
				localIntersect = tempIntersect;  
				localNormal = tempNormal;   
			}
		}
		intersectionPoint = multiplyMV(polygon.transform, glm::vec4(localIntersect, 1.0f));
		glm::vec3 normalP_OS = intersectionPoint + localNormal;
		glm::vec3 normalP_WS = multiplyMV(polygon.transform, glm::vec4(normalP_OS, 1.0f));
		normal = glm::normalize(normalP_WS - intersectionPoint);
		return tMin;
	}
	return -1;
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
    thrust::uniform_real_distribution<float> u01(-PI,PI);
	glm::vec3 radius = getRadiuses(sphere);
	float theta = (float)u01(rng);
	float phi = (float)u01(rng);
	glm::vec3 p = glm::vec3(glm::sin(theta) * glm::cos(phi), glm::sin(theta) * glm::sin(phi), glm::cos(theta));
	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(p, 1.0));
	return randPoint;
		
	//return glm::vec3(0,0,0);
}


#endif


