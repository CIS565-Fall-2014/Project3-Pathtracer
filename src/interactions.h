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
	//constructor
	AbsorptionAndScatteringProperties(glm::vec3 absorbCoeff, float scatterCoeff){
		absorptionCoefficient = absorbCoeff;
		reducedScatteringCoefficient = scatterCoeff;
	}
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
 //snell's law:   sin(thetaI) / sin(thetaT) = nt/ni
	float eta = incidentIOR/transmittedIOR;
	float cosI = -glm::dot(incident, normal);
	float sin2T = eta * eta * (1.0f - cosI * cosI);
	if(sin2T>1)  //TIR
		return calculateReflectionDirection(incident, normal);
	return eta * incident - (eta * cosI + sqrt(1-sin2T)) * normal;

	//return glm::normalize(glm::refract(glm::normalize(incident),glm::normalize(normal), (float)incidentIOR/(float)transmittedIOR));

	

}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {


	//Direction - 2*dot(Direction,Normal)*Normal
	return incident - 2.0f * glm::dot(incident, normal) * normal;
	//return glm::normalize(glm::reflect(incident,normal));
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {

	Fresnel f;
	f.reflectionCoefficient = 1;
	f.transmissionCoefficient = 0;

	float eta = incidentIOR/transmittedIOR;
	float cosI = -glm::dot(normal,incident);
	float sinT2 = eta * eta *(1.0f-cosI*cosI);
	if(sinT2>1){
		f.reflectionCoefficient = 1.0f;  //TIR
	}
	else{
		float cosT = sqrt(1.0f - sinT2);
		float rOrth = ( incidentIOR * cosI - transmittedIOR * cosT )/( incidentIOR * cosI + transmittedIOR * cosT );  //s-polarization, orthogonal
		float rPar = ( incidentIOR * cosT - transmittedIOR * cosI )/( incidentIOR * cosT + transmittedIOR * cosI);   //p- polarization, parallel
		f.reflectionCoefficient =  (rOrth*rOrth + rPar*rPar)/2.0f;
	}
	f.transmissionCoefficient = 1.0f - f.reflectionCoefficient;
	return f;
	
  /*  glm::vec3 inDir = glm::normalize(incident);

    float cosIncidentAngle = glm::dot(inDir, -normal);
	float sinIncidentAngle = glm::length(glm::cross(inDir, normal));
	float cosTransmitAngle;

	if(incidentIOR < transmittedIOR) // from air to glass
	{
		cosTransmitAngle = cos(asin(incidentIOR / transmittedIOR * sinIncidentAngle));
		fresnel.reflectionCoefficient  = pow(abs((incidentIOR * cosIncidentAngle - transmittedIOR * cosTransmitAngle) 
			                                   / (incidentIOR * cosIncidentAngle + transmittedIOR * cosTransmitAngle)), 2);
		fresnel.reflectionCoefficient += pow(abs((incidentIOR * cosTransmitAngle - transmittedIOR * cosIncidentAngle) 
			                                   / (incidentIOR * cosTransmitAngle + transmittedIOR * cosIncidentAngle)), 2);
		fresnel.reflectionCoefficient /= 2.0f;
	}
	else                                    // from glass to air
	{
		float sinCriticalAngle = transmittedIOR / incidentIOR;
		if(sinIncidentAngle > sinCriticalAngle) 
		{
			fresnel.reflectionCoefficient = 1.0f;
		}
		else
		{
			cosTransmitAngle = cos(asin(incidentIOR / transmittedIOR * sinIncidentAngle));
			fresnel.reflectionCoefficient  = pow(abs((incidentIOR * cosIncidentAngle - transmittedIOR * cosTransmitAngle) 
				/ (incidentIOR * cosIncidentAngle + transmittedIOR * cosTransmitAngle)), 2);
			fresnel.reflectionCoefficient += pow(abs((incidentIOR * cosTransmitAngle - transmittedIOR * cosIncidentAngle) 
				/ (incidentIOR * cosTransmitAngle + transmittedIOR * cosIncidentAngle)), 2);
			fresnel.reflectionCoefficient /= 2.0f;
		}
		
	}

    fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
    return fresnel;*/
}

__host__ __device__ glm::vec3  calculateCosWeightedRandomDirInHemisphere( glm::vec3 n, float Xi1, float Xi2)
{
    float  theta = acos(sqrt(1.0-Xi1));
    float  phi = 2.0 * 3.1415926535897932384626433832795 * Xi2;

    float xs = sinf(theta) * cosf(phi);
    float ys = cosf(theta);
    float zs = sinf(theta) * sinf(phi);

    glm::vec3 y(n.x, n.y, n.z);
    glm::vec3 h = y;
    if (fabs(h.x)<=fabs(h.y) && fabs(h.x)<=fabs(h.z))
        h.x= 1.0;
    else if (fabs(h.y)<=fabs(h.x) && fabs(h.y)<=fabs(h.z))
        h.y= 1.0;
    else
        h.z= 1.0;


	glm::vec3 x = glm::normalize( glm::cross( h, y ) );
	glm::vec3 z = glm::normalize( glm::cross(x, y) );

   glm::vec3 direction = glm::normalize(xs * x + ys * y + zs * z);
    return direction;
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

	// Crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    
    
    return glm::vec3(0,0,0);
    
  //return glm::vec3(0,0,0);
}

// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
// Returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3& color, 
	glm::vec3& unabsorbedColor, material m, int seed){

	AbsorptionAndScatteringProperties ASP( m.absorptionCoefficient, m.reducedScatterCoefficient);

	if(m.hasReflective>0){   //relection
		return 1;
	}
	else if(m.hasRefractive){   //refraction
		return 2;
	}
	else{   //diffuse
		thrust::default_random_engine rng(seed);
		thrust::uniform_real_distribution<float> u01(0,1);
		r.direction = calculateRandomDirectionInHemisphere( normal, (float) u01(rng), (float) u01(rng));
		r.origin = intersect + r.direction * 0.000001f;
		color = m.color;
		return 0;
	}
};

#endif
