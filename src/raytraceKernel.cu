// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
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
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int _x, int _y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
    float x = _x, y = _y;
    std::cout << fov.x << " " << fov.y << std::endl;
    fov *= PI / 180.f;
    glm::vec3 dir = glm::normalize(view);
    glm::vec3 norX = glm::normalize(glm::cross(dir , up )) * glm::tan(fov.x);
    glm::vec3 norY = glm::normalize(glm::cross(norX, dir)) * glm::tan(fov.y);

    thrust::default_random_engine rng(hash((x + y * resolution.x) * time));
    thrust::uniform_real_distribution<float> u01(0, 1);
    thrust::uniform_real_distribution<float> uhalf(-0.5, 0.5);

    // Antialiasing: jitter the pixel by up to half a pixel
    x += uhalf(rng);
    y += uhalf(rng);

    // Depth of field: jitter the origin and angle appropriately
    //   (TODO: move these constants out of here)
    const float blur_aper_rad = 0.00f;
    const float blur_foc_dist = 12.f;
    float sqrtr = glm::sqrt(u01(rng) * blur_aper_rad);
    float theta = u01(rng) * TWO_PI;
    glm::vec3 lens = norX * sqrtr * glm::cos(theta) + norY * sqrtr * glm::sin(theta);

    glm::vec2 ndc = glm::vec2(1 - x / resolution.x * 2, 1 - y / resolution.y * 2);
    ray r;
    r.origin = eye + lens * blur_foc_dist;
    r.direction = glm::normalize(dir - lens + norX * ndc.x + norY * ndc.y);
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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec4* image)
{

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {
        glm::vec4 pix = image[index];

        glm::vec3 color;
        if (pix.w > 0.5f) {
            color.x = pix.x / pix.w * 255.0;
            color.y = pix.y / pix.w * 255.0;
            color.z = pix.z / pix.w * 255.0;
        }

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


enum pathray_status { ALIVE, DEAD, CULLED };

struct pathray {
    enum pathray_status status;
    int index;
    glm::vec3 color;
    ray r;
};

struct pathray_is_alive {
    __host__ __device__ bool operator()(const struct pathray pr) {
        return pr.status == ALIVE;
    }
};

__global__ void init_pathrays(struct pathray *pathrays, float time, cameraData cam)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * cam.resolution.x);

    if (x < cam.resolution.x && y < cam.resolution.y) {
        struct pathray pr ={
            ALIVE, index, glm::vec3(1, 1, 1),
            raycastFromCameraKernel(cam.resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov),
        };
        pathrays[index] = pr;
    }
}

__device__ void merge_pathray(const struct pathray &pr, glm::vec4 *colors)
{
    glm::vec4 c = colors[pr.index];
    colors[pr.index] = glm::vec4(glm::vec3(c) + pr.color, c.w + 1);
}

__global__ void merge_dead_pathrays(struct pathray *pathrays, int pathraycount, float time, glm::vec4 *colors)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= pathraycount) {
        return;
    }

    struct pathray pr = pathrays[index];
    if (pr.status == DEAD) {
        merge_pathray(pr, colors);
    }
}

__device__ float scene_intersect(ray r,
        staticGeom *geoms, int numberOfGeoms, bool &tmin_ext,
        staticGeom &tmin_geom, glm::vec3 &tmin_pos, glm::vec3 &tmin_nor)
{
    float tmin = 1e38f;
    for (int i = 0; i < numberOfGeoms; ++i) {
        glm::vec3 p;
        glm::vec3 n;
        float t = 2e38f;
        bool e;
        staticGeom g = geoms[i];
        if (g.type == SPHERE) {
            t = sphereIntersectionTest(g, r, p, n, e);
        } else if (g.type == CUBE) {
            t = boxIntersectionTest(g, r, p, n, e);
        } else if (g.type == MESH) {
        }
        if (t > 0 && t < tmin) {
            tmin = t;
            tmin_geom = g;
            tmin_pos = p;
            tmin_nor = n;
            tmin_ext = e;
        }
    }
    return tmin;
}

__global__ void merge_live_pathrays(struct pathray *pathrays,
        int pathraycount, float time, glm::vec4 *colors)
        //staticGeom* geoms, int numberOfGeoms,
        //staticGeom* lights, int numberOfLights,
        //material *mats, int numberOfMaterials);
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= pathraycount) {
        return;
    }

    struct pathray pr = pathrays[index];
    if (pr.status == ALIVE) {
        // If the path never hit a light, assume it was black
        //   This causes the image to be darker at lower depths, but such is
        //   life, I guess. I removed the direct lighting because it sucked.
        pr.color = glm::vec3();
        merge_pathray(pr, colors);
    }
}

__global__ void pathray_step(struct pathray *pathrays,
        int pathraycount, float time, int depth,
        staticGeom* geoms, int numberOfGeoms,
        staticGeom* lights, int numberOfLights,
        material *mats, int numberOfMaterials)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= pathraycount) {
        return;
    }

    struct pathray pr = pathrays[index];

    ray r = pr.r;
    staticGeom tmin_geom;
    bool tmin_ext;
    glm::vec3 tmin_pos;
    glm::vec3 tmin_nor;
    float tmin = scene_intersect(r, geoms, numberOfGeoms, tmin_ext, tmin_geom, tmin_pos, tmin_nor);

    if (tmin > 9e37) {
        // Empty space; abort ray
        pr.status = DEAD;
        pr.color = glm::vec3();
        pathrays[index] = pr;
        return;
    }

    material mat = mats[tmin_geom.materialid];

    if (mat.emittance) {
        // Hit a light; abort ray
        pr.status = DEAD;
        pr.color *= mat.emittance * mat.color;
        pathrays[index] = pr;
        return;
    }

    // Calculate the ray of the next bounce
    thrust::default_random_engine rng(hash(index * time) ^ hash(depth));
    thrust::uniform_real_distribution<float> u01(0, 1);
    float hasDiffuse = glm::length(mat.color) > 0.0001f ? 1 : 0;
    float branchcount = hasDiffuse + mat.hasReflective + mat.hasRefractive;
    float raytype = u01(rng) * branchcount;
    float branch = 0.f;
    glm::vec3 c(branchcount);
    if (raytype < (branch += hasDiffuse)) {
        // Next bounce is diffuse
        c *= mat.color;
        pr.r.direction = calculateRandomDirectionInHemisphere(tmin_nor, u01(rng), u01(rng));
    } else if (raytype < (branch += mat.hasReflective)) {
        // Next bounce is specular
        c *= mat.specularColor;
        pr.r.direction = calculateReflectionDirection(tmin_nor, r.direction);
    } else if (raytype < (branch += mat.hasRefractive)) {
        // Next bounce is refractive
        //   (color doesn't change)
        float eta = mat.indexOfRefraction;
        if (tmin_ext) {
            eta = 1 / eta;
        }
        pr.r.direction = calculateTransmissionDirection(tmin_nor, r.direction, eta);
    }
    pr.r.origin = tmin_pos + pr.r.direction * 0.001f;
    pr.color *= c;
    if (glm::length(pr.color) < 0.00001f) {
        pr.status = DEAD;
    }
    pathrays[index] = pr;
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms)
{
    const int traceDepth = 16; //determines how many bounces the raytracer traces
    const int pixelcount = ((int) renderCam->resolution.x) * ((int) renderCam->resolution.y);

    // set up crucial magic
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x) / float(tileSize)), (int)ceil(float(renderCam->resolution.y) / float(tileSize)));

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

    // send image to GPU
    static glm::vec4* cudaimage = NULL;
    static staticGeom* cudageoms = NULL;
    static staticGeom* cudalights = NULL;
    static material* cudamats = NULL;
    static struct pathray *pathrays[2];
    if (cudaimage == NULL) {
        cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec4));
        cudaMalloc((void**)&cudageoms, numberOfGeoms * sizeof(staticGeom));
        cudaMalloc((void**)&cudalights, numberOfLights * sizeof(staticGeom));
        cudaMalloc((void**)&cudamats, numberOfMaterials * sizeof(material));
        cudaMalloc((void**)&pathrays[0], pixelcount * sizeof(struct pathray));
        cudaMalloc((void**)&pathrays[1], pixelcount * sizeof(struct pathray));
    }

    cudaMemcpy(cudaimage, renderCam->image, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec4), cudaMemcpyHostToDevice);
    cudaMemcpy(cudageoms, geomList, numberOfGeoms * sizeof(staticGeom), cudaMemcpyHostToDevice);
    cudaMemcpy(cudalights, lightList, numberOfLights * sizeof(staticGeom), cudaMemcpyHostToDevice);
    cudaMemcpy(cudamats, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);

    // package camera
    cameraData cam;
    cam.resolution = renderCam->resolution;
    cam.position = renderCam->positions[frame];
    cam.view = renderCam->views[frame];
    cam.up = renderCam->ups[frame];
    cam.fov = renderCam->fov;

    // kernel launches
    const int TPB = 64;
    int pathraycount = pixelcount;
    int bc = (pathraycount + TPB - 1) / TPB;
    float time = iterations;
    int which = 0;
    init_pathrays<<<fullBlocksPerGrid, threadsPerBlock>>>(pathrays[which], time, cam);
    for (int depth = 0; bc > 0 && depth < traceDepth; ++depth) {
        struct pathray *prs = pathrays[which];
        struct pathray *prd = pathrays[which ^ 1];
        // Compute one ray along each path
        pathray_step<<<bc, TPB>>>(prs, pathraycount, time, depth,
            cudageoms, numberOfGeoms,
            cudalights, numberOfLights,
            cudamats, numberOfMaterials);
        // Merge all of the dead paths into the image
        merge_dead_pathrays<<<bc, TPB>>>(prs, pathraycount, time, cudaimage);
        // Stream compact all of the dead paths away
        //   (this is required; otherwise dead paths keep getting merged!)
        pathraycount = thrust::copy_if(thrust::device,
                prs, prs + pathraycount, prd, pathray_is_alive()) - prd;
        bc = (pathraycount + TPB - 1) / TPB;
        which = which ^ 1;
    }
    // And finally handle all of the paths that haven't died yet
    if (bc > 0) {
        merge_live_pathrays<<<bc, TPB>>>(pathrays[which],
                pathraycount, time, cudaimage);
                //cudageoms, numberOfGeoms,
                //cudalights, numberOfLights,
                //cudamats, numberOfMaterials);
    }

    sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

    // retrieve image from GPU
    cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec4), cudaMemcpyDeviceToHost);

    // free up stuff, or else we'll leak memory like a madman
    //cudaFree(cudaimage);
    //cudaFree(cudageoms);
    //cudaFree(cudamats);
    //cudaFree(pathrays);
    delete[] geomList;
    delete[] lightList;

    // make certain the kernel has completed
    //cudaThreadSynchronize();

    checkCUDAError("Kernel failed!");
}
