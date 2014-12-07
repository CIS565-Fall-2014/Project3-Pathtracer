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
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

struct ray {
	glm::vec3 origin;
	glm::vec3 direction;
	bool exist;
	glm::vec3 raycolor;
	int initindex;
	float IOR; //REFRIOR
};

//Mesh struct 
struct triangle{
	glm::vec3 p1,p2,p3,normal;
	triangle()
	{
		p1= glm::vec3(0,0,0); p2= glm::vec3(0,0,0); p3= glm::vec3(0,0,0); normal= glm::vec3(0,0,0);
	};
	triangle(glm::vec3 t1,glm::vec3 t2,glm::vec3 t3,glm::vec3 n)
	{
		p1 = t1;
		p2 = t2;
		p3 = t3;
		normal = n;
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
	cudaMat4* transinverseTransforms;

    //data for Motion Blur
	glm::vec3* MBV;

	//data for obj
	triangle tri;

	//To accelerate
	int trinum;

	//Texture map data
	int texindex;
	int theight;
	int twidth;

	//Bump map data
	int bumpindex;
	int bheight;
	int bwidth;
};


struct staticGeom {
	enum GEOMTYPE type;
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
	cudaMat4 transinverseTransform;

	//data for Motion Blur
	glm::vec3 MBV;

	//data for obj
	int trinum;
	//To accelerate
	triangle tri;

	//Texture map data
	int texindex;
	int theight;
	int twidth;

	//Bump map data
	int bumpindex;
	int bheight;
	int bwidth;
};

struct cameraData {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;

	//Added for DOF
	float focallength;
	float blurradius;
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

	//Added for DOF
	float focall;
	float blurr;
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
