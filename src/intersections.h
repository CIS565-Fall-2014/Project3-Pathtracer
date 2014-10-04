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
__host__ __device__ float calculateArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);
__host__ __device__ float intersectTri(glm::vec3 rayOrigin, glm::vec3 rayDir, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 faceNormal);
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

__host__ __device__ float calculateArea(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3){
	float value1 = (p1[1] * p2[2] - p2[1] * p1[2]) + (p1[2] * p3[1] - p3[2] * p1[1]) + (p2[1] * p3[2]  - p3[1] * p2[2]);
	float value2 = (p1[2] * p2[0] - p2[2] * p1[0]) + (p1[0] * p3[2] - p3[0] * p1[2]) + (p2[2] * p3[0]  - p3[2] * p2[0]);
	float value3 = (p1[0] * p2[1] - p2[0] * p1[1]) + (p1[1] * p3[0] - p3[1] * p1[0]) + (p2[0] * p3[1]  - p3[0] * p2[1]);

	float area = 0.5 * sqrt(value1 * value1 + value2 * value2 + value3 * value3);
	return area;
}


__host__ __device__ float intersectTri(glm::vec3 rayOrigin, glm::vec3 rayDir, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 faceNormal){

	glm::vec3 vec_Origin_P1 = rayOrigin - p1;//, rayOrigin[1] - p1[1], rayOrigin[2] - p1[2]);
	
	if(abs(glm::dot(vec_Origin_P1, faceNormal)) < 0.000001){// on the surface
		return -1;
	}
	else{
		if(abs(glm::dot(faceNormal, rayDir)) < 0.000001){//V0 is parallel with surface
			return -1;
		}
		else
		{	
			float t;
			t = glm::dot(faceNormal, p1 - rayOrigin) / glm::dot(faceNormal, rayDir);

			if(t < 0)
				return -1;
			else
			{
				glm::vec3 pt = rayOrigin + rayDir * t;
				float s = calculateArea(p1, p2, p3);
				float s1 = calculateArea(pt, p2, p3) / s;
				float s2 = calculateArea(p1, pt, p3) / s;
				float s3 = calculateArea(p1, p2, pt) / s;

				if(s1 <= 1 && s2 <= 1 && s3 <= 1 && abs(s1 + s2 + s3 - 1) < 0.000001)//0.000001
					//return t / rayLength;
					return t;
				else 
					return -1;
			}
		}
	}
}

__host__ __device__ float intersectRec(glm::vec3 rayOrigin, glm::vec3 rayDir, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 p4, glm::vec3 faceNormal){

	glm::vec3 vec_Origin_P1 = rayOrigin - p1;
	
	if(abs(glm::dot(vec_Origin_P1, faceNormal)) < 0.000001){// on the surface
		return -1;
	}
	else{
		if(abs(glm::dot(faceNormal, rayDir)) < 0.000001){//V0 is parallel with surface
			return -1;
		}
		else
		{	
			float t;
			t = glm::dot(faceNormal, p1 - rayOrigin) / glm::dot(faceNormal, rayDir);

			if(t < 0)
				return -1;
			else
			{
				glm::vec3 pt = rayOrigin + rayDir * t;
				
				float s = calculateArea(p1, p2, p3);
				float s1 = calculateArea(pt, p2, p3) / s;
				float s2 = calculateArea(p1, pt, p3) / s;
				float s3 = calculateArea(p1, p2, pt) / s;

				if(s1 <= 1 && s2 <= 1 && s3 <= 1 && abs(s1 + s2 + s3 - 1) < 0.000001)
					//return t / rayLength;
					return t;
				else {
					s = calculateArea(p1, p3, p4);
					s1 = calculateArea(pt, p3, p4) / s;
					s2 = calculateArea(p1, pt, p4) / s;
					s3 = calculateArea(p1, p3, pt) / s;
					if(s1 <= 1 && s2 <= 1 && s3 <= 1 && abs(s1 + s2 + s3 - 1) < 0.000001)
						return t;
					else
						return -1;
				}
			}
		}
	}
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

	ray rt; 
	rt.origin = ro; 
	rt.direction = rd;

	
	glm::vec3 p1(0.5, 0.5, 0.5);
	glm::vec3 p2(-0.5, 0.5, 0.5);
	glm::vec3 p3(0.5, -0.5, 0.5);
	glm::vec3 p4(-0.5, -0.5, 0.5);
	glm::vec3 p5(0.5, 0.5, -0.5);
	glm::vec3 p6(-0.5, 0.5, -0.5);
	glm::vec3 p7(0.5, -0.5, -0.5);
	glm::vec3 p8(-0.5, -0.5, -0.5);

	glm::vec3 normal_p1_p3_p7_p5(1,0,0);
	glm::vec3 normal_p1_p5_p6_p2(0,1,0);
	glm::vec3 normal_p1_p2_p4_p3(0,0,1);
	glm::vec3 normal_p4_p2_p6_p8(-1,0,0);
	glm::vec3 normal_p3_p4_p8_p7(0,-1,0);
	glm::vec3 normal_p5_p7_p8_p6(0,0,-1);

	
	float tx1 = -1, tx2 = -1, ty1 = -1, ty2 = -1, tz1 = -1, tz2 = -1;
	int xNormal = 0, yNormal = 0, zNormal = 0;

	if(rd.x > 0){
		if(ro.x < -0.5){
			tx1 = intersectRec(ro, rd, p4, p2, p6, p8, normal_p4_p2_p6_p8);
		}
		else if(ro.x > -0.5 && ro.x < 0.5){
			tx1 = intersectRec(ro, rd, p1, p3, p7, p5, normal_p1_p3_p7_p5);	
		}
		xNormal = 1;
	}else if(rd.x < 0){
		if(ro.x > 0.5){	
			tx1 = intersectRec(ro, rd, p1, p3, p7, p5, normal_p1_p3_p7_p5);
		}
		else if(ro.x > -0.5 && ro.x < 0.5){
			tx1 = intersectRec(ro, rd, p4, p2, p6, p8, normal_p4_p2_p6_p8);
		}


		xNormal = -1;
	}

	if(rd.y > 0){
		if(ro.y < -0.5){	
			ty1 = intersectRec(ro, rd, p3, p4, p8, p7, normal_p3_p4_p8_p7);
		}
		else if(ro.y > -0.5 && ro.y < 0.5){
			ty1 = intersectRec(ro, rd, p1, p5, p6, p2, normal_p1_p5_p6_p2);
		}


		yNormal = 1;
	}else if(rd.y < 0){
		if(ro.y > 0.5){		
			ty1 = intersectRec(ro, rd, p1, p5, p6, p2, normal_p1_p5_p6_p2);
		}
		else if(ro.y > -0.5 && ro.y < 0.5){
			ty1 = intersectRec(ro, rd, p3, p4, p8, p7, normal_p3_p4_p8_p7);
		}


		yNormal = -1;
	}

	if(rd.z > 0){
		if(ro.z < -0.5){		
			tz1 = intersectRec(ro, rd, p5, p7, p8, p6, normal_p5_p7_p8_p6);
		}
		else if(ro.z > -0.5 && ro.z < 0.5){
			tz1 = intersectRec(ro, rd, p1, p2, p4, p3, normal_p1_p2_p4_p3);
		}


		zNormal = 1;
	}
	else if(rd.z < 0){
		if(ro.z > 0.5){		
			tz1 = intersectRec(ro, rd, p1, p2, p4, p3, normal_p1_p2_p4_p3);
		}
		else if(ro.z > -0.5 && ro.z < 0.5){
			tz1 = intersectRec(ro, rd, p5, p7, p8, p6, normal_p5_p7_p8_p6);
		}


		zNormal = -1;
	}

	float t = -1;
	glm::vec3 localNormal;
	if((tx1 != -1 && t == -1) || (tx1 != -1 && t != -1 && tx1 < t)){
		t = tx1;
		if(xNormal == 1){
			localNormal = normal_p4_p2_p6_p8;
		}
		else if(xNormal == -1){
			localNormal = normal_p1_p3_p7_p5;
		}
	}
	if((ty1 != -1 && t == -1) || (ty1 != -1 && t != -1 && ty1 < t)){
		t = ty1;
		if(yNormal == 1){
			localNormal = normal_p3_p4_p8_p7;
		}
		else if(yNormal == -1){
			localNormal = normal_p1_p5_p6_p2;
		}
	}
	if((tz1 != -1 && t == -1) || (tz1 != -1 && t != -1 && tz1 < t)){
		t = tz1;
		if(zNormal == 1){
			localNormal = normal_p5_p7_p8_p6;
		}
		else if(zNormal == -1){
			localNormal = normal_p1_p2_p4_p3;
		}
	}
	if(t == -1)
		return -1;

	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, t), 1.0));

	intersectionPoint = realIntersectionPoint;
	
	normal = glm::normalize( multiplyMV(box.transform, glm::vec4(localNormal, 0.0f)));
	
        
	return glm::length(r.origin - realIntersectionPoint);

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
	if (radicand <= 0){
		return -1;
	}
  
	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	bool outside;
	float t = 0;
	if (t1 < 0 && t2 < 0) {
		return -1;
	} else if (t1 > 0 && t2 > 0) {
		t = min(t1, t2);
		outside = true;
	} else {
		t = max(t1, t2);
		outside = false;
	}

	glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
	glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

	intersectionPoint = realIntersectionPoint;
	if(outside == true)
		normal = glm::normalize(realIntersectionPoint - realOrigin);
	else
		normal = glm::normalize(realOrigin - realIntersectionPoint);
        
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

	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-1,1);

	float russianRoulette1 = (float)u01(rng);
	float russianRoulette2 = (float)u02(rng);

	float xCoor = 0.5 * sqrt(1 - russianRoulette2 * russianRoulette2) * cos(russianRoulette1 * 2 * PI);
	float yCoor = 0.5 * sqrt(1 - russianRoulette2 * russianRoulette2) * sin(russianRoulette1 * 2 * PI);
	float zCoor = 0.5 * russianRoulette2;
	glm::vec3 localRandPoint = glm::vec3(xCoor, yCoor, zCoor);

	glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(localRandPoint, 1.0f));

	return randPoint;
}

__host__ __device__ glm::vec3 getRandomPointOnAperture(glm::vec3 centerPt, glm::vec3 view, glm::vec3 up, float radius, float randomSeed){
	
	thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0, 1);
    //thrust::uniform_real_distribution<float> u02(-1, 1);

	glm::vec3 wing = glm::cross(view, up);

	float russianRoulette1 = (float)u01(rng);
	float russianRoulette2 = (float)u01(rng);

	float xCoor = radius * sqrt(1 - russianRoulette2 * russianRoulette2) * cos(russianRoulette1 * 2 * PI);
	float yCoor = radius * sqrt(1 - russianRoulette2 * russianRoulette2) * sin(russianRoulette1 * 2 * PI);
	return centerPt + xCoor * wing; + yCoor * up;
}

#endif


