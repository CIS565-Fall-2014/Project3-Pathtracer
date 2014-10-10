// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y)
{
    int index = x + (y * resolution.x);

    thrust::default_random_engine rng(hash(index * time));
    thrust::uniform_real_distribution<float> u01(0, 1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
    std::cout << fov.x << " " << fov.y << std::endl;
    fov *= PI / 180.f;
    glm::vec2 ndc = glm::vec2(1 - x / resolution.x * 2, 1 - y / resolution.y * 2);
    glm::vec3 dir = glm::normalize(view);
    glm::vec3 norX = glm::normalize(glm::cross(dir , up )) * glm::tan(fov.x);
    glm::vec3 norY = glm::normalize(glm::cross(norX, dir)) * glm::tan(fov.y);

#if 0
    // This is probably totally wrong but is mainly here for checking to make
    // sure that the time-averaging code results in convergence
    const float BLUR = 0.02f;
    thrust::default_random_engine rng(hash((x + y * resolution.x) * time));
    thrust::uniform_real_distribution<float> u(-BLUR, BLUR);
    thrust::uniform_real_distribution<float> v(-BLUR, BLUR);
    glm::vec3 lens = norX * u(rng) + norY * v(rng);
#else
    glm::vec3 lens;
#endif

    ray r;
    r.origin = eye + lens;
    r.direction = glm::normalize(dir + lens + norX * ndc.x + norY * ndc.y);
    return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        image[index] = glm::vec3(0, 0, 0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image)
{

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {

        glm::vec3 color;
        color.x = image[index].x * 255.0;
        color.y = image[index].y * 255.0;
        color.z = image[index].z * 255.0;

        if (color.x > 255) {
            color.x = 255;
        }

        if (color.y > 255) {
            color.y = 255;
        }

        if (color.z > 255) {
            color.z = 255;
        }

        // Each thread writes one pixel location in the texture (textel)
        PBOpos[index].w = 0;
        PBOpos[index].x = color.x;
        PBOpos[index].y = color.y;
        PBOpos[index].z = color.z;
    }
}

struct pathray {
    int index;
    glm::vec3 color;
    ray r;
};

// TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms,
                            staticGeom* lights, int numberOfLights,
                            material *mats, int numberOfMaterials)
{

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    thrust::default_random_engine rng(hash(time));
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (x <= resolution.x && y <= resolution.y) {
        struct pathray pr = {
            index, glm::vec3(1, 1, 1),
            raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov),
        };
        for (int depth = 0; depth < rayDepth; ++depth) {
            if (pr.index == -1) {
                // This path was terminated
                continue;
            }

            ray r = pr.r;
            staticGeom tmin_geom;
            glm::vec3 tmin_pos;
            glm::vec3 tmin_nor;
            float tmin = 1e38f;
            for (int i = 0; i < numberOfGeoms; ++i) {
                glm::vec3 p;
                glm::vec3 n;
                float t = 2e38f;
                staticGeom g = geoms[i];
                if (g.type == SPHERE) {
                    t = sphereIntersectionTest(g, r, p, n);
                } else if (g.type == CUBE) {
                    t = boxIntersectionTest(g, r, p, n);
                } else if (g.type == MESH) {
                }
                if (t > 0 && t < tmin) {
                    tmin = t;
                    tmin_geom = g;
                    tmin_pos = p;
                    tmin_nor = n;
                }
            }

            if (tmin > 9e37) {
                // Empty space; abort ray
                // TODO: add some here instead of treating as black?
                pr.index = -1;
                pr.color = glm::vec3();
                continue;
            }

            material mat = mats[tmin_geom.materialid];

            if (mat.emittance) {
                // Hit a light; abort ray
                pr.color = mat.emittance * mat.color;
                pr.index = -1;
                continue;
            }

            // Calculate the ray of the next bounce
            float raytype = u01(rng) * 1;
            if (raytype < 1) {
                // Next bounce is diffuse
                pr.color *= mat.color;
                pr.r.direction = calculateRandomDirectionInHemisphere(
                        tmin_nor, u01(rng), u01(rng));
                pr.r.origin = tmin_pos + pr.r.direction * 0.001f;
            } else {
                pr.index = -1;
                continue;
            }
        }
        if (pr.index != -1) {
            pr.color = glm::vec3();
        }
        colors[index] = (colors[index] * time + pr.color) / (time + 1);
    }
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms)
{

    int traceDepth = 2; //determines how many bounces the raytracer traces

    // set up crucial magic
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x) / float(tileSize)), (int)ceil(float(renderCam->resolution.y) / float(tileSize)));

    // send image to GPU
    glm::vec3* cudaimage = NULL;
    cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3));
    cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    // package geometry and materials and sent to GPU
    staticGeom* geomList = new staticGeom[numberOfGeoms];
    staticGeom* lightList = new staticGeom[numberOfGeoms];
    int numberOfLights = 0;
    for (int i = 0; i < numberOfGeoms; i++) {
        staticGeom newStaticGeom;
        newStaticGeom.type = geoms[i].type;
        newStaticGeom.materialid = geoms[i].materialid;
        newStaticGeom.translation = geoms[i].translations[frame];
        newStaticGeom.rotation = geoms[i].rotations[frame];
        newStaticGeom.scale = geoms[i].scales[frame];
        newStaticGeom.transform = geoms[i].transforms[frame];
        newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
        geomList[i] = newStaticGeom;
        if (materials[newStaticGeom.materialid].emittance > 0) {
            lightList[numberOfLights] = newStaticGeom;
            numberOfLights += 1;
        }
    }

    staticGeom* cudageoms = NULL;
    cudaMalloc((void**)&cudageoms, numberOfGeoms * sizeof(staticGeom));
    cudaMemcpy(cudageoms, geomList, numberOfGeoms * sizeof(staticGeom), cudaMemcpyHostToDevice);
    staticGeom* cudalights = NULL;
    cudaMalloc((void**)&cudalights, numberOfLights * sizeof(staticGeom));
    cudaMemcpy(cudalights, lightList, numberOfLights * sizeof(staticGeom), cudaMemcpyHostToDevice);

    material* cudamats = NULL;
    cudaMalloc((void**)&cudamats, numberOfMaterials * sizeof(material));
    cudaMemcpy(cudamats, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);

    // package camera
    cameraData cam;
    cam.resolution = renderCam->resolution;
    cam.position = renderCam->positions[frame];
    cam.view = renderCam->views[frame];
    cam.up = renderCam->ups[frame];
    cam.fov = renderCam->fov;

    // kernel launches
    raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(
            renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage,
            cudageoms, numberOfGeoms,
            cudalights, numberOfLights,
            cudamats, numberOfMaterials);

    sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

    // retrieve image from GPU
    cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // free up stuff, or else we'll leak memory like a madman
    cudaFree(cudaimage);
    cudaFree(cudageoms);
    cudaFree(cudamats);
    delete[] geomList;

    // make certain the kernel has completed
    cudaThreadSynchronize();

    checkCUDAError("Kernel failed!");
}
