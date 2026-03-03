#pragma once
#include "Vec3.h"

// Ray: P(t) = origin + t*direction
class Ray {
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3& origin, const Vec3& direction)
        : orig(origin), dir(direction)
    {}

    __host__ __device__ Point3 origin() const  { return orig; }
    __host__ __device__ Vec3 direction() const { return dir; }

    __host__ __device__ Point3 at(float t) const {
        return orig + t*dir;
    }

private:
    Point3 orig;
    Vec3 dir;
};
