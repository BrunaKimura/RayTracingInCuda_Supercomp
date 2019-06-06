#include <iostream>
#include "ray.h"

#define NX 800
#define NY 400
#define SIZE NX*NY*3*sizeof(int)

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r){
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;

    if (discriminant < 0.0){
        return -1.0;
    }
    else{
        return (-b-sqrt(discriminant)) / (2.0*a);
    }
}

__device__ vec3 color (const ray& r){
    float t = hit_sphere(vec3(0,0,-1), 0.5, r);
    if (t > 0.0){
        vec3 N = unit_vector(r.point_at_parameter(t) - vec3(0,0,-1));
        return 0.5*vec3(N.x()+1, N.y()+1, N.z()+1);
    }
    else{
        vec3 unit_direction = unit_vector(r.direction());
        t = 0.5*(unit_direction.y()+1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
    
    }
    

__global__ void generate(int *matrix, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float u = float(i)/float(NX);
    float v = float(j)/float(NY);

    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    vec3 col = color(r);

    int ir = int(255.99*col[0]);
    int ig = int(255.99*col[1]);
    int ib = int(255.99*col[2]);

    matrix[(i*NY + j)*3] = ir;
    matrix[(i*NY + j)*3 + 1] = ig;
    matrix[(i*NY + j)*3 + 2] = ib;

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

    cpu_matrix = (int *)malloc(SIZE);
    cudaMalloc((void **)&gpu_matrix,SIZE);

    generate<<<dimGrid, dimBlock>>>(gpu_matrix, lower_left_corner, horizontal, vertical, origin);

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