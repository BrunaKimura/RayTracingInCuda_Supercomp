#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"

#define NX 800
#define NY 400
#define SIZE NX*NY*3*sizeof(int)


__device__ vec3 color (const ray& r, hitable **world){
    hit_record rec;

    if((*world)->hit(r, 0.0, MAXFLOAT, rec)){
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

__global__ void create_world(hitable **list, hitable **world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(list)   = new sphere(vec3(0,0,-1), 0.5);
        *(list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *world    = new hitable_list(list,2);
    }
}

__global__ void free_world(hitable **list, hitable **world) {
    delete *(list);
    delete *(list+1);
    delete *world;
 }


int main(){

    dim3 dimGrid(ceil(NX/(float)16), ceil(NY/(float)16));
    dim3 dimBlock(16, 16);

    int *cpu_matrix;
    int *gpu_matrix;

    

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);
    hitable **list;
    cudaMalloc((void **)&list, 2*sizeof(hitable *));
    hitable **world;
    cudaMalloc((void **)&world, sizeof(hitable *));
    create_world<<<dimGrid, dimBlock>>>(list,world);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // sphere *esfera = new sphere(vec3(0,0,-1), 0.5);
    // sphere *sp_device;
    // list[1] = new sphere(vec3(0,-100.5,-1), 100);
    // hitable *world = new hitable_list(list,2);

    // cudaMalloc((void **)&sp_device, sizeof(sphere));
    // cudaMemcpy(sp_device, esfera, sizeof(sphere), cudaMemcpyHostToDevice);

    cpu_matrix = (int *)malloc(SIZE);
    cudaMalloc((void **)&gpu_matrix,SIZE);

    // checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(list,world);
    // checkCudaErrors(cudaGetLastError());
    cudaFree(list);
    cudaFree(world);

    generate<<<dimGrid, dimBlock>>>(gpu_matrix, lower_left_corner, horizontal, vertical, origin, world);

    cudaMemcpy(cpu_matrix, gpu_matrix, SIZE, cudaMemcpyDeviceToHost);

    cudaFree(gpu_matrix);
    

    std::cout << "P3\n" << NX << " " << NY << "\n255\n";
    for (int j = NY-1; j >= 0; j--){
        for(int i = 0; i < NX; i++){
            std::cout << cpu_matrix[(i*NY + j)*3] << " " << cpu_matrix[(i*NY + j)*3 + 1] << " " << cpu_matrix[(i*NY + j)*3 + 2] << "\n";
        }
    }
    delete[] cpu_matrix;
}