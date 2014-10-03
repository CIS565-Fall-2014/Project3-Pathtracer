// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient;
};

// Forward declaration
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION 
//Useless as I caculate ray in/out of object in intersection test and just use glm::refract
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//Useless as I caculate ray in/out of object in intersection test and just use glm::reflect
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 rdirect, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  Fresnel fresnel;
  float n1 = incidentIOR;
  float n2 = transmittedIOR;
  float n = n1 / n2;

  rdirect = glm::normalize(rdirect);
  normal = glm::normalize(normal);
  float c1 = glm::dot(-rdirect, normal);
  float c2 = 1 - (n*n)*(1 - c1*c1);

  if(c2>=0)
  {
	  c2 = sqrt(c2);
	  float R1 = glm::abs( (n1*c1 - n2*c2) / (n1*c1 + n2*c2) ) * glm::abs( (n1*c1 - n2*c2) / (n1*c1 + n2*c2) );
	  float R2 = glm::abs( (n1*c2 - n2*c1) / (n1*c2 + n2*c1) ) * glm::abs( (n1*c2 - n2*c1) / (n1*c2 + n2*c1) );

	  float R = (R1 + R2) / 2.0f;
	  float T = 1.0 - R;

	  fresnel.reflectionCoefficient   = R;
	  fresnel.transmissionCoefficient = T;
  }
  else
  {
	  fresnel.reflectionCoefficient   = 1.0f;
	  fresnel.transmissionCoefficient = 0;
  }

  return fresnel;
}

// LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    // Crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    // Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}

// TODO: IMPLEMENT THIS FUNCTION
// Now that you know how cosine weighted direction generation works, try implementing 
// non-cosine (uniform) weighted random direction generation.
// This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
	float theta = 2 * TWO_PI * xi1;
	float phi = acos(2 * xi2 - 1);
	float u = cos(phi);

	return glm::vec3(sqrt(1 - u*u) * cos(theta), sqrt(1 - u*u) * sin(theta), u);
}

// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
// Returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
#define MINNUM 0.001f
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 InterSectP,glm::vec3 InterSectN, material m,float randomSeed,int depth){

	thrust::uniform_real_distribution<float> u01(0,1);
	thrust::default_random_engine rng(hash(randomSeed));
	float russianRoulette = (float)u01(rng);
	if(depth>5 && russianRoulette < 0.2){   //russian roulette rule: ray is absorbed
		r.exist = false;
	}
	//Only diffuse
	if(m.hasReflective<MINNUM&&m.hasRefractive<MINNUM)	
	{
		//diffuse reflect
		r.direction = calculateRandomDirectionInHemisphere(InterSectN, (float)u01(rng), (float)u01(rng));
		r.origin = InterSectP + MINNUM * r.direction;
		r.IOR = 1.0f;
		return 0;
	}

	float oldIOR = r.IOR;
	float newIOR = m.indexOfRefraction;
	glm::vec3 reflectR = glm::reflect(r.direction, InterSectN);
	glm::vec3 refractR;
	Fresnel fresnel;

	if(m.hasRefractive>MINNUM)
	{			
		float reflect_range = -1;
		float IOR12 = oldIOR/newIOR;
		refractR = glm::refract(r.direction,InterSectN,IOR12);
		fresnel = calculateFresnel(InterSectN,r.direction,oldIOR,newIOR,reflectR,refractR);

		//return reflect ray,refract ray randomly depends on their Coefficients
		if(m.hasReflective>MINNUM) 
			reflect_range =  fresnel.reflectionCoefficient;  

		if((float)u01(rng)<reflect_range)
		{
			r.direction = reflectR;
			r.origin = InterSectP + MINNUM * r.direction;
			r.IOR = newIOR;
			return 1;
		}
		else
		{
			r.IOR = newIOR;
			r.direction = refractR;
			r.origin = InterSectP + MINNUM * r.direction;
			return 2;
		}
	}
	else
	{
		r.IOR = 1.0f;
		r.direction = reflectR;
		r.origin = InterSectP + MINNUM * r.direction;
		return 1;
	}
};

__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m){

  return 1;
};

#endif
