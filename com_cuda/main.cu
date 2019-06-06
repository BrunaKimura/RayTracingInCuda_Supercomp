#include <iostream>
#include <float.h>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"

#define NX 200
#define NY 100
#define SIZE NX*NY*3*sizeof(int)


__device__ vec3 color (const ray& r, hitable **world){
    hit_record rec;

    if((*world)->hit(r, 0.0, FLT_MAX, rec)){
        return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
    }
    else{
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y()+1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void generate(int *matrix, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hitable **world){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if((i >= NX) || (j >= NY)) return;
    float u = float(i)/float(NX);
    float v = float(j)/float(NY);

    ray r(origin, lower_left_corner + u*horizontal + v*vertical);

    vec3 p = r.point_at_parameter(2.0);
    vec3 col = color(r, world);

    int ir = int(255.99*col[0]);
    int ig = int(255.99*col[1]);
    int ib = int(255.99*col[2]);

    matrix[(i*NY + j)*3] = ir;
    matrix[(i*NY + j)*3 + 1] = ig;
    matrix[(i*NY + j)*3 + 2] = ib;

}

__global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0, 0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *(d_list+2) = new sphere(vec3(1,0,-1), 0.5);
        *(d_list+3) = new sphere(vec3(-1,0,-1), 0.5);
        *d_world    = new hitable_list(d_list, 4);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *(d_list+2);
    delete *(d_list+3);
    delete *d_world;
}

int main(){

    clock_t start, stop;
    start = clock();

    dim3 dimGrid(ceil(NX/(float)16), ceil(NY/(float)16));
    dim3 dimBlock(16, 16);

    int *cpu_matrix;
    int *gpu_matrix;

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    hitable **d_list;
    cudaMalloc((void **)&d_list, 4*sizeof(hitable *));
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *));
    create_world<<<1,1>>>(d_list,d_world);

    cpu_matrix = (int *)malloc(SIZE);
    cudaMalloc((void **)&gpu_matrix,SIZE);

    generate<<<dimGrid, dimBlock>>>(gpu_matrix, lower_left_corner, horizontal, vertical, origin, d_world);

    cudaMemcpy(cpu_matrix, gpu_matrix, SIZE, cudaMemcpyDeviceToHost);
    cudaFree(gpu_matrix);
    free_world<<<1,1>>>(d_list,d_world);
    cudaFree(d_list);
    cudaFree(d_world);

    std::cout << "P3\n" << NX << " " << NY << "\n255\n";
    for (int j = NY-1; j >= 0; j--){
        for(int i = 0; i < NX; i++){
            std::cout << cpu_matrix[(i*NY + j)*3] << " " << cpu_matrix[(i*NY + j)*3 + 1] << " " << cpu_matrix[(i*NY + j)*3 + 2] << "\n";
        }
    }
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";
    
    delete[] cpu_matrix;
}