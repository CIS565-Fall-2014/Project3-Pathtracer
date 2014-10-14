// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>

#include "macros.h"

enum GEOMTYPE{ SPHERE, CUBE, MESH };

//class RenderCre{
//	static RenderCre* Inst;
//public:
//	static RenderCre* GetInstance(){
//		if(!Inst) //return Inst;
//			Inst=new RenderCre;
//		return Inst;
//	}
//	RenderCre(){
//	}
//};
//
//RenderCre* RenderCre::Inst=NULL;

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
	__host__ __device__ ray(){}
	__host__ __device__ ray(glm::vec3& o, glm::vec3& d){
		origin=o+d*0.001f;
		direction=glm::normalize(d);
	}
	__host__ __device__ ray(glm::vec3& o, glm::vec3& d, glm::vec3& n){
		//origin=o;
		direction=glm::dot(d,n)<0?d:-d;
		direction=glm::normalize(direction);
		origin=o+direction*0.001f;
	}
};
struct pathray{
	bool isDead;
	int depth;
	ray curray;
	glm::vec2 onscreen;
	//int geoId[traceDepth];
	//float blnPng[traceDepth];
	__host__ __device__ pathray(){
		depth=0;
		isDead=false;
		//memset(geoId,-1,traceDepth*sizeof(int));		
	}
	__host__ __device__ pathray(int x, int y, glm::vec3& o,glm::vec3& d):onscreen(x,y),curray(o,d){
		depth=0;
		isDead=false;
		//memset(geoId,-1,traceDepth*sizeof(int));
	}
	__host__ __device__ pathray(int x, int y, glm::vec3& o,glm::vec3& d, glm::vec3& n):onscreen(x,y),curray(o,d,n){
		depth=0;
		isDead=false;
		//memset(geoId,-1,traceDepth*sizeof(int));
	}
	
	__host__ __device__ ray& operator=(ray& r){
		curray=r;
		return r;
	};
	__host__ __device__ void pushback(int id,ray& r,glm::vec3& n){
		//geoId[depth]=id;
		//blnPng[depth]=0;//glm::clamp(glm::abs(glm::dot(n,glm::normalize(glm::normalize(r.direction)-glm::normalize(curray.direction))))/*/glm::dot(curray.origin-r.origin,curray.origin-r.origin)*/,0.0f,1.0f);
		//blnPng[depth]=blnPng[depth]>0?blnPng[depth]:0;
		/*depth++;
		curray=r;
		isDead=(depth>=traceDepth);*/
	}
	__host__ __device__ int getback(){
		//return depth>0?geoId[--depth]:-1;
	}
	__host__ __device__ void getback(int& id, float& phg){
		if(depth>0){
			//id=geoId[--depth];
			//phg=blnPng[depth];
		}
		else{
			id=-1;
			phg=0;
		}

	}
};

struct IsDead{
	__host__ __device__ bool operator()(const pathray r){
		return r.isDead;
	}
};


struct geom {
	enum GEOMTYPE type;
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
};

struct staticGeom {
	enum GEOMTYPE type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
};

struct camera {
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int frames;
	glm::vec2 fov;
	unsigned int iterations;
	glm::vec3* image;
	ray* rayList;
	std::string imageName;
};

struct material{
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};

#endif //CUDASTRUCTS_H
