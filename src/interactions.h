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
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
	return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
  return glm::normalize(incident-2.0f*glm::dot(incident,normal)*normal);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
	Fresnel fresnel;
  	float n1=incidentIOR;
	float n2=transmittedIOR;
	
	//if n1 or n2 is 0, then only relfection
	if(n1<=0.001f ||n2<=0.001f)
	{
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
		return fresnel;
	}

	float ratio=n1/n2;
	float cos_theta1=glm::dot(incident,-normal);
	float cos_theta2_square=1-ratio*ratio*(1-cos_theta1*cos_theta1);
	//check if this is a total internal reflection
	if(cos_theta2_square<0)
	{
		reflectionDirection=calculateReflectionDirection(normal,incident);
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
		return fresnel;
	}

	float cos_theta2=glm::sqrt(cos_theta2_square);
	
	if(cos_theta1>0.0f)
	{
		//out-side-in
		transmissionDirection=ratio*incident+(ratio*cos_theta1-cos_theta2)*normal;
	}
	else
	{
		//in-side-out
		transmissionDirection=ratio*incident+(ratio*cos_theta1+cos_theta2)*normal;
	}

	//s-polarize light
	float Rs=0;
	float Rp=0;
	if(!epsilonCheck(n1*cos_theta1+n2*cos_theta2,0))
	{
		Rs=abs(n1*cos_theta1-n2*cos_theta2)*abs(n1*cos_theta1+n2*cos_theta2);
	}
	if(!epsilonCheck(n1*cos_theta2+n2*cos_theta1,0))
	{
		Rs=abs(n1*cos_theta2-n2*cos_theta1)*abs(n1*cos_theta2+n2*cos_theta1);
	}

	fresnel.reflectionCoefficient=(Rs+Rp)*0.5f;
	fresnel.transmissionCoefficient=1-fresnel.reflectionCoefficient;
	reflectionDirection=calculateReflectionDirection(normal,incident);
	return fresnel;
}

// LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    // Crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    // Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). 
	//Learned this trick from Peter Kutz.
    
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
	float alpha,beta;
	alpha=xi1*2.0f*PI;
	beta=xi2*2.0f*PI;
	glm::vec3 D;
	D=glm::vec3(cos(beta)*cos(alpha),cos(beta)*sin(alpha),sin(beta));
	return D;
}

#define	THRESHOLD  0.001f
// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
// Returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(float randseed, ray& Ray, staticGeom* geoms,int ObjectID,glm::vec3 intersectPoint,glm::vec3 normal,material M)
{

	thrust::default_random_engine rng(hash(randseed));
	thrust::uniform_real_distribution<float> u01(0.0f,1.0f);
	Fresnel fresnel;

	
	float random_shift=0.01f*(float)u01(rng);

	if(M.hasReflective<THRESHOLD && M.hasRefractive<THRESHOLD)
	{
		//only diffuse
		Ray.origin=intersectPoint+random_shift*normal;
		Ray.direction=calculateRandomDirectionInHemisphere(normal,(float)u01(rng),(float)u01(rng));
		Ray.color=Ray.color*M.color;
		return 0;
	}
	else 
	{
		float n1,n2;
		if(glm::dot(Ray.direction,normal)<0.0f)
		{
			//out-side-in
			n1=1.0;
			n2=M.indexOfRefraction;
			glm::vec3 reflectionDirection;
			glm::vec3 transmissionDirection;
			fresnel=calculateFresnel(normal,Ray.direction,n1,n2,reflectionDirection,transmissionDirection);
			if(epsilonCheck(fresnel.reflectionCoefficient,1)||!M.hasRefractive)
			{
				//only reflection
				Ray.origin=intersectPoint+random_shift*normal;
				Ray.direction=reflectionDirection;
				Ray.color=Ray.color*M.specularColor;
				return 1;
			}
			else if(!M.hasReflective && M.hasRefractive)
			{
				//only refraction
				Ray.origin=intersectPoint+random_shift*normal;
				Ray.direction=transmissionDirection;
				Ray.color=Ray.color*M.color;
				
				return 2;
			}
			else if(M.hasReflective&& M.hasRefractive)
			{
				float random_flag=(float)u01(rng);
				if(random_flag<fresnel.reflectionCoefficient)
				{
					//reflection
					Ray.origin=intersectPoint+random_shift*normal;
					Ray.direction=reflectionDirection;
					Ray.color=Ray.color*M.specularColor;
					return 1;
				}
				else
				{
					//refraction
					Ray.origin=intersectPoint+random_shift*normal;
					Ray.direction=transmissionDirection;
					Ray.color=Ray.color*M.color;
					return 2;
				}
			}
		}
		else
		{
			//in-side-out
			n1=M.indexOfRefraction;
			n2=1.0;
			glm::vec3 reflectionDirection;
			glm::vec3 transmissionDirection;
			fresnel=calculateFresnel(-normal,Ray.direction,n1,n2,reflectionDirection,transmissionDirection);
			if(epsilonCheck(fresnel.reflectionCoefficient,1)||!M.hasRefractive)
			{
				//only reflection
				Ray.origin=intersectPoint+random_shift*normal;
				Ray.direction=reflectionDirection;
				Ray.color=Ray.color*M.specularColor;
				return 1;
			}
			else if(!M.hasReflective && M.hasRefractive)
			{
				//only refraction
				Ray.origin=intersectPoint+random_shift*normal;
				Ray.direction=transmissionDirection;
				Ray.color=Ray.color*M.color;
				return 2;
			}
			else if(M.hasReflective&& M.hasRefractive)
			{
				float random_flag=(float)u01(rng);
				if(random_flag<fresnel.reflectionCoefficient)
				{
					//reflection
					Ray.origin=intersectPoint+random_shift*normal;
					Ray.direction=reflectionDirection;
					Ray.color=Ray.color*M.specularColor;
					return 1;
				}
				else
				{
					//refraction
					Ray.origin=intersectPoint+random_shift*normal;
					Ray.direction=transmissionDirection;
					Ray.color=Ray.color*M.color;
					return 2;
				}
			}
		}

	}

  return 0;
};


#endif
