// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"
#include "utilities.h"
#include <thrust/random.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//struct Fresnel {
//  float reflectionCoefficient;
//  float transmissionCoefficient;
//};

//struct AbsorptionAndScatteringProperties{
//    glm::vec3 absorptionCoefficient;
//    float reducedScatteringCoefficient;
//};

// Forward declaration
//__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
//__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
//__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
//__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
//__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

//// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
//  return glm::vec3(0,0,0);
//}
//
//// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
//                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
//  return false;
//}
//
//// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
//  return glm::vec3(0,0,0);
//}
//
//// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
//  //nothing fancy here
//  return glm::vec3(0,0,0);
//}
//
//// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
//__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
//  Fresnel fresnel;
//
//  fresnel.reflectionCoefficient = 1;
//  fresnel.transmissionCoefficient = 0;
//  return fresnel;
//}

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

// TODO-[DONE]: IMPLEMENT THIS FUNCTION
// Now that you know how cosine weighted direction generation works, try implementing 
// non-cosine (uniform) weighted random direction generation.
// This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.

// NOTE: this seems really similar to just getting some point in intersections.h.
// It looks like NOTHING that's actually "random" happens in either this function or the one above (random hemisphere).
// I guess xi1 and xi2 are random numbers given to me. Based on personal testing of calculateRandomDirectionInHemisphere,
// I expect these random numbers to be between 0 and 1.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {

	// theta, phi
	float theta = 2. * PI * xi1;
	float phi = acos(2. * xi2 - 1.);

	// calculate cartesian coords
	float x = sin(phi) * cos(theta);
	float y = sin(phi) * sin(theta);
	float z = cos(phi);

	glm::vec3 point(x,y,z);

	return point;
}

// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
// Returns 0 if diffuse scatter, 1 if reflected, 2 if it hits LIGHT (active is then set to false).

// NOTE: this function REQUIRES both "diffuse" and "perfect specular reflective" functionality!
// I will NOT be supporting transmittance, so only the "hemisphere" random function will be used.
__host__ __device__ int calculateBSDF(ray& r, int depth, material m, glm::vec3 intersectpt, glm::vec3 normal, int iterationNumber){
	
	//if the material the ray hit is a light, just stop the bounces there.
	// I examined the code of some other, more successful students, and decided to make simplifications.
	// You can see what lines I removed by noticing that they have been commented out. (That was really obvious)
	if (m.emittance > 0.f) {
		//// ray has hit the light directly
		//if (depth == 0) {
		//	r.color = m.emittance * m.color * r.intensityMultiplier;
		//}
		// ray hit the light after some bounces
		//else {
			r.color *= (m.emittance * m.color);// * r.intensityMultiplier;
		//}

		return 2;
	}
										  
	// hasReflective of 0 means no reflection
	if (m.hasReflective < EPSILON) {
		//update the intensity multiplier using the Lambertian model
		// first, get a new ray
		//srand (time(NULL));
		thrust::default_random_engine rng1(hash(iterationNumber * r.sourceindex));
		thrust::default_random_engine rng2(hash(iterationNumber * r.sourceindex * 2.));
		thrust::uniform_real_distribution<float> r01(0,1);
		thrust::uniform_real_distribution<float> r02(0,1);

		glm::vec3 newRay = calculateRandomDirectionInHemisphere(normal, (float)r01(rng1), (float)r02(rng2));

		// calculate the intensity
		float cos_th = glm::dot(newRay, normal) / glm::length(newRay) / glm::length(normal);
		//r.intensityMultiplier *= cos_th;

		// add the color of the surface
		//r.color += m.color * r.intensityMultiplier;

		// Actually, it seems what I should do is MULTIPLY the color
		r.color *= (m.color * cos_th);

		// update r's attributes
		r.direction = glm::normalize(newRay);
		r.origin = intersectpt + (float)EPSILON * r.direction; // ensure it doesn't hit itself

		return 0;
	}
	// if hasReflective is anything else, PERFECT REFLECTION
	else {
		glm::vec3 incomingDirection = r.direction;
		float c1 = -1. * glm::dot(normal, incomingDirection);
		//update r's attributes
		r.direction = incomingDirection + (normal * (2.f * c1));
		r.origin = intersectpt + (float)EPSILON * r.direction; // ensure it doesn't hit itself

		// no difficult color math needed for perfect reflection, so don't worry about any other inputs.

		return 1;
	}
};

#endif
