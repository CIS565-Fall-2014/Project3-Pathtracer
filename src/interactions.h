// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

#define RAY_SPAWN_OFFSET 0.01f

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
  return glm::vec3(0,0,0);
}

//calculate fresnel
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float n1, float n2, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
	Fresnel fresnel;

	//safety normalize
	incident = glm::normalize(incident);
	transmissionDirection = glm::normalize(transmissionDirection);
	normal = glm::normalize(normal);

	float cosIncident =  glm::dot(incident,-normal);
	float cosTransmitted =glm::dot(transmissionDirection,-normal);
	float sinIncident = sin(acos(cosIncident));

	//case of total internal reflection
	if(n2 < n1)
	{
		if( asin(sinIncident) > asin(n2/n1) )
		{
			fresnel.reflectionCoefficient = 1.0f;
			fresnel.transmissionCoefficient = 0.0f;
			return fresnel;
		}
	}

	float RS = abs( (n1*cosIncident - n2* cosTransmitted)/(n1*cosIncident + n2* cosTransmitted))
			* abs( (n1*cosIncident - n2* cosTransmitted)/(n1*cosIncident + n2* cosTransmitted));

	float RP = abs( (n1*cosTransmitted - n2* cosIncident)/(n1*cosTransmitted + n2* cosIncident))
			* abs( (n1*cosTransmitted - n2* cosIncident)/(n1*cosTransmitted + n2* cosIncident));

	fresnel.reflectionCoefficient = (RS + RP) * 0.5f;

	fresnel.transmissionCoefficient = 1.0f - fresnel.reflectionCoefficient;
	return fresnel;
}

//This function demonstrates cosine weighted random direction generation in a hemi-sphere!
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
  return glm::vec3(0,0,0);
}

// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
// Returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, float randSeed,glm::vec3 intersect, glm::vec3 normal, material mat){

	thrust::default_random_engine eng(hash(randSeed));
	thrust::uniform_real_distribution<float> distribution(0.0f,1.0f);

	float dice = (float)distribution((eng));
	//diffuse
	if(dice <mat.diffuseCoe)
	{
		r.direction = calculateRandomDirectionInHemisphere(normal,(float)distribution((eng)),(float)distribution(eng));
		r.origin = intersect + RAY_SPAWN_OFFSET * normal;
		r.color *= mat.color;
	}

	//other interactions
	else
	{
		//compute common paras
		bool isInsideOut = glm::dot(r.direction,normal) > 0.0f;
		float n1 = isInsideOut ? mat.indexOfRefraction : 1.0f;
		float n2 = isInsideOut ? 1.0f : mat.indexOfRefraction;
		glm::vec3 reflectDir = glm::reflect(r.direction,normal);
		glm::vec3 transmitDir = glm::refract(r.direction,(isInsideOut) ? -normal: normal,n1/n2);

		//reflect only
		if(mat.hasReflective && !mat.hasRefractive)
		{
			r.direction = reflectDir;
			r.origin = intersect + RAY_SPAWN_OFFSET * ((isInsideOut) ? (-normal): normal) ;
			r.color *= mat.specularColor;
			return 1;
		}

		//refract only
		else if(!mat.hasReflective && mat.hasRefractive)
		{
			r.direction = transmitDir;
			r.origin = intersect + RAY_SPAWN_OFFSET *  ((!isInsideOut) ? (-normal): normal);
			r.color *= mat.color;
			return 2;
		}

		else if(!mat.hasReflective && !mat.hasRefractive)
		{
			return 0;
		}

		//both refract and reflect, using fresnel
		else
		{
			Fresnel fres = calculateFresnel(normal,r.direction,n1,n2,reflectDir,transmitDir);
			float newDice = (float)distribution((eng));
			//fres.reflectionCoefficient = 0.3f;
			//fres.transmissionCoefficient = 0.7f;
			//to reflect
			if( newDice < fres.reflectionCoefficient)
			//if( dice < (1.0f -mat.diffuseCoe) * fres.reflectionCoefficient + mat.diffuseCoe)
			{
				r.direction = reflectDir;
				r.origin = intersect + RAY_SPAWN_OFFSET * ((isInsideOut) ? (-normal): normal) ;
				r.color *= mat.specularColor;
				return 1;
			}

			//transmit
			else
			{
				r.direction = transmitDir;
				r.origin = intersect + RAY_SPAWN_OFFSET *  ((!isInsideOut) ? (-normal): normal);
				r.color *= mat.color;
				return 2;
			}
		}		
	}
	return 0;
};

#endif
