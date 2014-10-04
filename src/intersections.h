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

__host__ __device__ cudaMat4 transposeM(cudaMat4 m){
	cudaMat4 mt;
	mt.x.x = m.x.x;  mt.y.y = m.y.y;  mt.z.z = m.z.z;  mt.w.w = m.w.w;   //diagonal term remains
	mt.x.y = m.y.x;  mt.x.z = m.z.x;  mt.x.w = m.w.x;
	mt.y.z = m.z.y;  mt.y.w = m.w.y;
	mt.z.w = m.w.z; 

	return mt;
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


//find the smallest positive value within an array
__host__ __device__ float find_smallest_t( float* ptr, int length){
	//double* ptr = t_ptr;
	float min=-1;
	bool flag=true;
	for ( int i=0; i<length; i++){
		if( flag == true && *ptr>=0){
			min = *ptr;
			flag = false;  //turn off first flag
		}
		if(*ptr<min && *ptr>=0 && flag==false)
			min = *ptr;
		ptr++;
	}
	return min;
}

// TODO: IMPLEMENT THIS FUNCTION
// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){

	float min = -1, max = 1;
	glm::vec4 O,PP,VV;
	float t[6],D,ans;
	glm::vec3 N,A,B,Pi,p[8],p1,p2,p3;

	//deifning vertex for cube
	p[0] = glm::vec3(min,min,min);  p[1] = glm::vec3(min,min,max); p[2] = glm::vec3(max,min,max); p[3] = glm::vec3(max,min,min);  
	p[4] = glm::vec3(min,max,min);  p[5] = glm::vec3(min,max,max); p[6] = glm::vec3(max,max,max); p[7] = glm::vec3(max,max,min);

	//transform the ray
	PP = glm::vec4( r.origin, 1.0f );   //vec4 of ray start
	VV = glm::vec4( r.direction, 0.0f );   //vec4 of ray direction
	glm::vec3 P = multiplyMV ( box.inverseTransform, PP);
	glm::vec3 V = multiplyMV ( box.inverseTransform, VV);

	//if ray origin is within the cube
	if( (P[0]==min || P[0]==max) && P[1]>=min && P[1]<=max && P[2]>=min && P[2]<=max)
		return -1;
	else if( (P[1]==min || P[1]==max) && P[0]>=min && P[0]<=max && P[2]>=min && P[2]<=max)
		return -1;
	else if ((P[2]==min || P[2]==max) && P[0]>=min && P[0]<=max && P[1]>=min && P[1]<=max)
		return -1;

	//initializing t array, each correspond to the t value of left,front,right,back,bottom,top square
	for(int i=0; i<6; i++)
		t[i] = -1;

	for(int i=0; i<6; i++){
		if(i<4){ //left,front, right, back plane
			p1 = p[i]; p2 = p[i+1]; p3 = p[i+4];
		}
		else if(i==4){//bottom plane
			p1 = p[0]; p2 = p[1]; p3 = p[2];
		}
		else if(i==5){//top plane
			p1 = p[4]; p2 = p[5]; p3 = p[6];
		}
		
		//defining plane
		A = p2 - p1;
		B = p3 - p1;
		N = glm::cross( A, B );  
		Pi = glm::vec3( 0.0, 0.0, 0.0 );

		//ray is not parallel with plane
		if(glm::dot(N,V)!=0){  
			D = glm::dot(N,p1);
			t[i] = -( glm::dot(N,P) + D ) / glm::dot(N,V);
			Pi = P + glm::vec3( t[i]*V[0], t[i]*V[1], t[i]*V[2] );
		}

		//intersection point is not within the quare boundary
		if( t[i]>0 && (Pi[0]>max || Pi[1]>max || Pi[2]>max || Pi[0]<min || Pi[1]<min || Pi[2]<min) )
				t[i] = -1;
	}
	ans = find_smallest_t (t,6);

	if(ans!=-1){ //has intersection
		intersectionPoint = r.origin + ans * r.direction;
			if(t[0]==ans){  
			normal = multiplyMV ( transposeM(box.inverseTransform), glm::vec4(-1,0,0,0) );  //left plane
		}
		else if(t[1]==ans){
			normal = multiplyMV ( transposeM(box.inverseTransform), glm::vec4(0,0,1,0) );  //front plane
		}
		else if(t[2]==ans){
			normal = multiplyMV ( transposeM(box.inverseTransform), glm::vec4(1,0,0,0) );  //right plane
		}
		else if(t[3]==ans){
			normal = multiplyMV ( transposeM(box.inverseTransform), glm::vec4(0,0,-1,0) );  //back plane
		}
		else if(t[4]==ans){
			normal = multiplyMV ( transposeM(box.inverseTransform), glm::vec4(0,-1,0,0) );  //bottom plane
		}
		else if(t[5]==ans){
			normal = multiplyMV ( transposeM(box.inverseTransform), glm::vec4(0,1,0,0) );  //top plane
		}
		return glm::length(intersectionPoint - r.origin) ;
	}
	else{
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

// TODO: IMPLEMENT THIS FUNCTION
// Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

  return glm::vec3(0,0,0);
}

#endif


