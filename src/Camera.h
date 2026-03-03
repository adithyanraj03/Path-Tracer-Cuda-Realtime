#pragma once
#include "Ray.h"
#include "utils.h"

class Camera {  
public:    
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;

    __host__ __device__ Camera() {} // default needed for GPU

    __host__ __device__ Camera(
        const Point3& lookfrom,
        const Point3& lookat,
        const Vec3& vup,
        float vfov_degrees,
        float aspect_ratio
    ) {
        float theta = degrees_to_radians(vfov_degrees);
        float h = tan(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - w;
    }

    __host__ __device__ Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
    }
};
