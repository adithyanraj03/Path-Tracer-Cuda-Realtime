#pragma once
#include "Ray.h"

class AABB {
public:
    Point3 minimum;
    Point3 maximum;

    __host__ __device__ AABB() {}
    __host__ __device__ AABB(const Point3& min, const Point3& max) : minimum(min), maximum(max) {}

    __host__ __device__ Point3 min() const { return minimum; }
    __host__ __device__ Point3 max() const { return maximum; }

    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (minimum[a] - r.origin()[a]) * invD;
            float t1 = (maximum[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

__host__ __device__ inline AABB surrounding_box(AABB box0, AABB box1) {
    Point3 small_pt(
        fmin(box0.min().x(), box1.min().x()),
        fmin(box0.min().y(), box1.min().y()),
        fmin(box0.min().z(), box1.min().z()));

    Point3 large_pt(
        fmax(box0.max().x(), box1.max().x()),
        fmax(box0.max().y(), box1.max().y()),
        fmax(box0.max().z(), box1.max().z()));

    return AABB(small_pt, large_pt);
}
