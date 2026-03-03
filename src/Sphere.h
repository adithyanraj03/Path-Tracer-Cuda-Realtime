#pragma once
#include "Hittable.h"

class Sphere {
public:
    Point3 center;
    float radius;
    int material_idx;

    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Point3 c, float r, int m) : center(c), radius(r), material_idx(m) {}

    __host__ __device__ AABB bounding_box() const {
        return AABB(
            center - Vec3(radius, radius, radius),
            center + Vec3(radius, radius, radius)
        );
    }
};
