#pragma once
#include "Ray.h"
#include "AABB.h"

struct HitRecord {
    Point3 p;
    Vec3 normal;
    float t;
    bool front_face;
    int material_idx;

    __host__ __device__ inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
