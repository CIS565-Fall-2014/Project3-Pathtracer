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
__host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b)
{
    if (fabs(fabs(a) - fabs(b)) < EPSILON) {
        return true;
    } else {
        return false;
    }
}

// Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t)
{
    return r.origin + float(t - .0001f) * glm::normalize(r.direction);
}

// LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
// This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
// Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v)
{
    glm::vec3 r(1, 1, 1);
    r.x = (m.x.x * v.x) + (m.x.y * v.y) + (m.x.z * v.z) + (m.x.w * v.w);
    r.y = (m.y.x * v.x) + (m.y.y * v.y) + (m.y.z * v.z) + (m.y.w * v.w);
    r.z = (m.z.x * v.x) + (m.z.y * v.y) + (m.z.z * v.z) + (m.z.w * v.w);
    return r;
}

// Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r)
{
    return glm::vec3(1.0 / r.direction.x, 1.0 / r.direction.y, 1.0 / r.direction.z);
}

// Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r)
{
    glm::vec3 inv_direction = getInverseDirectionOfRay(r);
    return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}


__host__ __device__ float squareIntersectionTest(ray r, int xyz, float dist, glm::vec3 &normal)
{
    if (glm::abs(r.direction[xyz]) < 0.00001f) {
        return -1;
    }

    float t = (dist - r.origin[xyz]) / r.direction[xyz];
    glm::vec3 ix = getPointOnRay(r, t);

    normal = glm::vec3();
    normal[xyz] = dist;

    int yzx = (xyz + 1) % 3;
    int zxy = (xyz + 2) % 3;
    return (-0.5 <= ix[yzx] && ix[yzx] <= 0.5 && -0.5 <= ix[zxy] && ix[zxy] <= 0.5) ? t : -1;
}

// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
    ray r1;
    r1.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    r1.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    const glm::vec3 vmin(-0.5, -0.5, -0.5);
    const glm::vec3 vmax(+0.5, +0.5, +0.5);
    float norfact = (glm::all(glm::lessThan(vmin, r1.origin))
                     && glm::all(glm::lessThan(r1.origin, vmax))) ? -1 : 1;
    float tmin = 1e38f;
    for (int xyz = 0; xyz < 3; ++xyz) {
        for (float dist = -0.5; dist < 0.75f; dist += 1) {
            glm::vec3 nor;
            float t = squareIntersectionTest(r1, xyz, dist, nor);
            if (t > 0 && t < tmin) {
                tmin = t;
                normal = nor;
            }
        }
    }
    normal *= norfact;
    normal = glm::normalize(multiplyMV(box.transform, glm::vec4(normal, 0.0f)));
    intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(r1, tmin), 1.0f));
    return tmin < 1e37f ? tmin : -1;
}

// LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
// Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{

    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
    if (radicand < 0) {
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
    glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0, 0, 0, 1));

    intersectionPoint = realIntersectionPoint;
    normal = glm::normalize(realIntersectionPoint - realOrigin);

    return glm::length(r.origin - realIntersectionPoint);
}

// Returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom)
{
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0, 0, 0, 1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5, 0, 0, 1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0, .5, 0, 1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0, 0, .5, 1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

// LOOK: Example for generating a random point on an object using thrust.
// Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed)
{

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> u02(-0.5, 0.5);

    // Get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z * 4.0f; //x-z face
    float totalarea = 2.0f * (side1 + side2 + side3);

    // Pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);

    glm::vec3 point = glm::vec3(.5, .5, .5);

    if (russianRoulette < (side1 / totalarea)) {
        // x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    } else if (russianRoulette < ((side1 * 2) / totalarea)) {
        // x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    } else if (russianRoulette < (((side1 * 2) + (side2)) / totalarea)) {
        // y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    } else if (russianRoulette < (((side1 * 2) + (side2 * 2)) / totalarea)) {
        // y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    } else if (russianRoulette < (((side1 * 2) + (side2 * 2) + (side3)) / totalarea)) {
        // x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    } else {
        // x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }

    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point, 1.0f));

    return randPoint;

}

// TODO: TEST THIS FUNCTION: probably wrong with nonuniform scaling applied
// Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed)
{
    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u(0, 2 * PI);
    thrust::uniform_real_distribution<float> v(-1, 1);
    float th = (float) u(rng);
    float ph = (float) glm::acos(v(rng));

    return multiplyMV(sphere.transform, glm::vec4(
            glm::cos(ph) * glm::cos(th),
            glm::cos(ph) * glm::sin(th),
            glm::sin(ph), 1.0f));
}

#endif


