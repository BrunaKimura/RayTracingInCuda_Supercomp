#include <iostream>
#include <stdio.h>
#include <stdlib.h>
// #include "vec3.h"
#include "ray.h"

#define NX 800
#define NY 400
#define SIZE NX*NY*3*sizeof(int)


__global__ void generate(int *A){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    float u = float(i) / float(NX);
    float v = float(j) / float(NY);

    //color e hit_sphere
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    vec3 center(0.0,0.0,-1.0);
    float radius = 0.5;

    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2*dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b*b - 4*a*c;
    float hit_sphere;
    if(discriminant < 0){
        hit_sphere = -1.0;
    }else{
        hit_sphere = (-b -sqrt(discriminant))/(2.0*a);
    }

    vec3 col;
    
    float t = hit_sphere;
    if(t > 0.0){
        vec3 N = unit_vector(r.point_at_parameter(t) - vec3(0,0,-1));
        col = 0.5 * vec3(N.x()+1, N.y()+1, N.z()+1);
    }else{
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5 * (unit_direction.y() + 1.0);
        col =  (1.0 - t)*vec3(1.0, 1.0, 1.0)+t*vec3(0.5, 0.7, 1.0);
    }

    int ir = int(255.99*col[0]);
    int ig = int(255.99*col[1]);
    int ib = int(255.99*col[2]);
    A[(i*NY + j)*3] = ir;
    A[(i*NY + j)*3 + 1] = ig;
    A[(i*NY + j)*3 + 2] = ib;
}

int main() {

    dim3 dimGrid(ceil(NX/(float)16), ceil(NY/(float)16));
    dim3 dimBlock(16, 16);

    // dim3 numBlocks(ceil(, ceil(NY / threadsPerBlock.y));
    int *cpuA;
    int *gpuA;
    cpuA = (int *)malloc(SIZE);
    cudaMalloc((void **)&gpuA,SIZE);
    generate<<<dimGrid, dimBlock>>>(gpuA);
    cudaMemcpy(cpuA, gpuA, SIZE, cudaMemcpyDeviceToHost);
    cudaFree(gpuA);

    std::cout << "P3\n" << NX << " " << NY << "\n255\n";
    for (int j = NY-1; j >= 0; j--){
        for(int i = 0; i < NX; i++){
            std::cout << cpuA[(i*NY + j)*3] << " " << cpuA[(i*NY + j)*3 + 1] << " " << cpuA[(i*NY + j)*3 + 2] << "\n";
        }
    }
    delete[] cpuA;
}