#pragma once
#include "Sphere.h"
#include "Triangle.h"

enum PrimitiveType {
    PRIM_SPHERE,
    PRIM_TRIANGLE
};

struct Primitive {
    PrimitiveType type;
    int material_idx;

    Point3 v0; 
    union {
        float radius; // For spheres
        struct {
            // These don't have constructors so they are POD inside union wrapper on CUDA
            float v1_x, v1_y, v1_z;
            float v2_x, v2_y, v2_z;
            float n_x, n_y, n_z;
        } tri;
    };

    __host__ __device__ Primitive() {}

    __host__ __device__ Primitive(const Sphere& s) {
        type = PRIM_SPHERE;
        material_idx = s.material_idx;
        v0 = s.center;
        radius = s.radius;
    }

    __host__ __device__ Primitive(const Triangle& t) {
        type = PRIM_TRIANGLE;
        material_idx = t.material_idx;
        v0 = t.v0;
        tri.v1_x = t.v1.x(); tri.v1_y = t.v1.y(); tri.v1_z = t.v1.z();
        tri.v2_x = t.v2.x(); tri.v2_y = t.v2.y(); tri.v2_z = t.v2.z();
        tri.n_x = t.normal.x(); tri.n_y = t.normal.y(); tri.n_z = t.normal.z();
    }

    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        if (type == PRIM_SPHERE) {
            Vec3 oc = r.origin() - v0;
            float a = r.direction().length_squared();
            float half_b = dot(oc, r.direction());
            float c = oc.length_squared() - radius * radius;
            float discriminant = half_b * half_b - a * c;

            if (discriminant < 0.0f) return false;

            float sqrtd = sqrt(discriminant);
            float root = (-half_b - sqrtd) / a;
            if (root < t_min || t_max < root) {
                root = (-half_b + sqrtd) / a;
                if (root < t_min || t_max < root)
                    return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            Vec3 outward_normal = (rec.p - v0) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.material_idx = material_idx;
            return true;
        } else {
            const float EPSILON = 1e-8f;
            Vec3 v1(tri.v1_x, tri.v1_y, tri.v1_z);
            Vec3 v2(tri.v2_x, tri.v2_y, tri.v2_z);
            Vec3 normal(tri.n_x, tri.n_y, tri.n_z);
            
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec3 h = cross(r.direction(), edge2);
            float a = dot(edge1, h);

            if (a > -EPSILON && a < EPSILON) return false;

            float f = 1.0f / a;
            Vec3 s = r.origin() - v0;
            float u = f * dot(s, h);

            if (u < 0.0f || u > 1.0f) return false;

            Vec3 q = cross(s, edge1);
            float v = f * dot(r.direction(), q);

            if (v < 0.0f || u + v > 1.0f) return false;

            float t = f * dot(edge2, q);

            if (t < t_min || t > t_max) return false;

            rec.t = t;
            rec.p = r.at(t);
            rec.set_face_normal(r, normal);
            rec.material_idx = material_idx;
            return true;
        }
    }

    __host__ __device__ AABB bounding_box() const {
        if (type == PRIM_SPHERE) {
            return AABB(
                v0 - Vec3(radius, radius, radius),
                v0 + Vec3(radius, radius, radius)
            );
        } else {
            Point3 min_p(
                fmin(v0.x(), fmin(tri.v1_x, tri.v2_x)),
                fmin(v0.y(), fmin(tri.v1_y, tri.v2_y)),
                fmin(v0.z(), fmin(tri.v1_z, tri.v2_z))
            );
            Point3 max_p(
                fmax(v0.x(), fmax(tri.v1_x, tri.v2_x)),
                fmax(v0.y(), fmax(tri.v1_y, tri.v2_y)),
                fmax(v0.z(), fmax(tri.v1_z, tri.v2_z))
            );
            Point3 pad(0.0001f, 0.0001f, 0.0001f);
            return AABB(min_p - pad, max_p + pad);
        }
    }
};
