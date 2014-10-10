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
__host__
__device__
glm::vec3 calculateTransmission( glm::vec3 absorptionCoefficient,
								 float distance )
{
	return glm::vec3( 0.0f, 0.0f, 0.0f );
}


// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__
__device__
bool calculateScatterAndAbsorption( ray& r,
									float& depth,
									AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
									glm::vec3& unabsorbedColor,
									material m,
									float randomFloatForScatteringDistance,
									float randomFloat2,
									float randomFloat3 )
{
	return false;
}


// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__
__device__
glm::vec3 calculateTransmissionDirection( glm::vec3 normal,			// Intersection normal.
										  glm::vec3 incident,		// Incoming ray direction.
										  float ior_incident,		// Incoming index of refraction.
										  float ior_transmitted )	// Outgoing index of refraction.
{
	glm::vec3 normal_oriented = ( glm::dot( normal, incident ) < 0.0f ) ? normal : ( -1.0f * normal );
	bool ray_is_entering = ( glm::dot( normal, normal_oriented ) > 0.0f );

	float n = ior_incident / ior_transmitted;
	float c1 = glm::dot( incident, normal_oriented );
	float c2 = 1.0f - ( n * n * ( 1.0f - c1 * c1 ) );

	// If angle is too shallow, then all light is reflected.
	if ( c2 < 0.0f ) {
		//return glm::vec3( 0.0f, 0.0f, 0.0f );
		return calculateReflectionDirection( normal, incident );
	}

	return glm::normalize( ( incident * n ) - ( normal * ( ( ray_is_entering ? 1.0f : -1.0f ) * ( c1 * n + sqrt( c2 ) ) ) ) );
}


// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__
__device__
glm::vec3 calculateReflectionDirection( glm::vec3 normal,		// Intersection normal.
										glm::vec3 incident )	// Incoming ray direction.
{
	//glm::vec3 normal_oriented = ( glm::dot( normal, incident ) < 0.0f ) ? normal : ( -1.0f * normal );
	//return glm::normalize( incident - ( 2.0f * glm::dot( normal_oriented, incident ) * normal_oriented ) );

	return incident - ( 2.0f * glm::dot( normal, incident ) * normal );
}


// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__
__device__
Fresnel calculateFresnel( glm::vec3 normal,				// Intersection normal.
						  glm::vec3 incident,			// Incoming ray direction.
						  float ior_incident,			// Incoming index of refraction.
						  float ior_transmitted,		// Outgoing index of refraction.
						  glm::vec3 reflection_dir,		// Reflecton direction.
						  glm::vec3 transmission_dir )	// Transmission direction.
{
	float n = ior_incident / ior_transmitted;
	float a = n - 1.0f;
	float b = n + 1.0f;
	float Ro = ( a * a ) / ( b * b );

	glm::vec3 normal_oriented = ( glm::dot( normal, incident ) < 0.0f ) ? normal : ( -1.0f * normal );
	bool ray_is_entering = ( glm::dot( normal, normal_oriented ) > 0.0f );
	float angle = ray_is_entering ? glm::dot( -incident, normal ) : glm::dot( transmission_dir, normal );
	float c = 1.0f - angle;

	float R = Ro + ( 1.0f - Ro ) * c * c * c * c * c;

	//float P = 0.25f + 0.5f * R;
	//float RP = R / P;
	//float TP = ( 1.0f - R ) / ( 1.0f - P );

	Fresnel fresnel;
	fresnel.reflectionCoefficient = R;
	fresnel.transmissionCoefficient = 1.0f - R;
	
	return fresnel;
}


// LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__
__device__
glm::vec3 calculateRandomDirectionInHemisphere( glm::vec3 normal,
												float xi1,
												float xi2 )
{
	// Crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
	float up = sqrt( xi1 ); // cos(theta)
	float over = sqrt( 1.0f - up * up ); // sin(theta)
	float around = xi2 * TWO_PI;
    
	// Find a direction that is not the normal based off of whether or not the normal's components are all equal to
	// sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
	glm::vec3 directionNotNormal;
	if ( abs( normal.x ) < SQRT_OF_ONE_THIRD ) {
		directionNotNormal = glm::vec3( 1.0f, 0.0f, 0.0f );
	} else if ( abs( normal.y ) < SQRT_OF_ONE_THIRD ) {
		directionNotNormal = glm::vec3( 0.0f, 1.0f, 0.0f );
	} else {
		directionNotNormal = glm::vec3( 0.0f, 0.0f, 1.0f );
	}
    
	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 = glm::normalize( glm::cross( normal, directionNotNormal ) );
	glm::vec3 perpendicularDirection2 = glm::normalize( glm::cross( normal, perpendicularDirection1 ) );
    
	return ( up * normal ) + ( cos( around ) * over * perpendicularDirection1 ) + ( sin( around ) * over * perpendicularDirection2 );
}


// TODO: IMPLEMENT THIS FUNCTION
// Now that you know how cosine weighted direction generation works, try implementing 
// non-cosine (uniform) weighted random direction generation.
// This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__
__device__
glm::vec3 getRandomDirectionInSphere( float xi1,
									  float xi2 )
{
	float r = sqrt( 1.0f - xi1 * xi1 ); // sin(theta)
	float phi = xi2 * TWO_PI;
	return glm::vec3( r * cos( phi ), r * sin( phi ), xi1 );
}


// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
// Returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
__host__
__device__
int calculateBSDF( ray& r,
				   glm::vec3 intersect,
				   glm::vec3 normal,
				   glm::vec3 emittedColor,
				   AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
				   glm::vec3& color,
				   glm::vec3& unabsorbedColor,
				   material m )
{
	//// Diffuse.
	//if ( !m.hasReflective && !m.hasRefractive ) {
	//	return 0;
	//}
	//// Perfect specular.
	//else {
	//	return 1;
	//}

	return 0;
};

#endif
