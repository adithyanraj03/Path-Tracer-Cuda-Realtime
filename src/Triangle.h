#pragma once
#include "Hittable.h"

class Triangle {
public:
    Point3 v0, v1, v2;
    Vec3 normal;
    int material_idx;

    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(const Point3& a, const Point3& b, const Point3& c, int m) 
        : v0(a), v1(b), v2(c), material_idx(m) {
        normal = normalize(cross(v1 - v0, v2 - v0));
    }

    __host__ __device__ AABB bounding_box() const {
        Point3 min_p(
            fmin(v0.x(), fmin(v1.x(), v2.x())),
            fmin(v0.y(), fmin(v1.y(), v2.y())),
            fmin(v0.z(), fmin(v1.z(), v2.z()))
        );
        Point3 max_p(
            fmax(v0.x(), fmax(v1.x(), v2.x())),
            fmax(v0.y(), fmax(v1.y(), v2.y())),
            fmax(v0.z(), fmax(v1.z(), v2.z()))
        );
        Point3 pad(0.0001f, 0.0001f, 0.0001f);
        return AABB(min_p - pad, max_p + pad);
    }
};
