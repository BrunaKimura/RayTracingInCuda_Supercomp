#include <iostream>
#include <values.h>
#include <float.h>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"

vec3 color_ray(const ray& r, hitable *world) {
    const vec3 white = vec3(1.0, 1.0, 1.0);
    const vec3 blue = vec3(0.5, 0.7, 1.0);
    const vec3 red = vec3(1, 0, 0);

    hit_record record;
    if (world->hit(r, 0.0, MAXFLOAT, record)) {
        return 0.5 * vec3(record.normal.x() + 1, record.normal.y() + 1, record.normal.z() + 1);
    } else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - t) * white + t * blue;
  }
}

int main() {
    clock_t start, stop;
    start = clock();

    int nx = 800;
    int ny = 400;

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    hitable *list[4];
    list[0] = new sphere(vec3(0, 0, -1), 0.5);
    list[1] = new sphere(vec3(0, -100.5, -1), 100);
    list[2] = new sphere(vec3(1, 0, -1), 0.5);
    list[3] = new sphere(vec3(-1, 0, -1), 0.5);
    hitable *world = new hitable_list(list, 4);

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            float u = float(i) / float(nx);
            float v = float(j) / float(ny);

            ray r(origin, lower_left_corner + u*horizontal + v*vertical);
            vec3 color = color_ray(r, world);

            int ir = int(255.99*color.r());
            int ig = int(255.99*color.g());
            int ib = int(255.99*color.b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    return 0;
}