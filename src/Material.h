#pragma once
#include "Ray.h"
#include "Hittable.h"
#include "Texture.h"
#include <curand_kernel.h>

enum MaterialType {
    MAT_DIFFUSE,
    MAT_METAL,
    MAT_DIELECTRIC,
    MAT_EMISSIVE
};

struct Material {
    MaterialType type;
    Texture albedo;
    float roughness; // used by metal
    float ior;       // used by dielectric 
    Color emission;  // used by emissive

    __host__ __device__ Material() {}
    
    // Helper Constructors
    __host__ __device__ static Material Diffuse(Texture a) {
        Material m; m.type = MAT_DIFFUSE; m.albedo = a; m.emission = Color(0,0,0); return m;
    }
    __host__ __device__ static Material Metal(Texture a, float r) {
        Material m; m.type = MAT_METAL; m.albedo = a; m.roughness = r; m.emission = Color(0,0,0); return m;
    }
    __host__ __device__ static Material Dielectric(float i) {
        Material m; m.type = MAT_DIELECTRIC; m.albedo = Texture(Color(1,1,1)); m.ior = i; m.emission = Color(0,0,0); return m;
    }
    __host__ __device__ static Material Emissive(Color e) {
        Material m; m.type = MAT_EMISSIVE; m.albedo = Texture(Color(0,0,0)); m.emission = e; return m;
    }
};

__device__ inline float random_float(curandState *local_rand_state) {
    return curand_uniform(local_rand_state);
}

__device__ inline float random_float(curandState *local_rand_state, float min, float max) {
    return min + (max-min)*random_float(local_rand_state);
}

__device__ inline Vec3 random_in_unit_sphere(curandState *local_rand_state) {
    Vec3 p;
    do {
        p = 2.0f*Vec3(random_float(local_rand_state), random_float(local_rand_state), random_float(local_rand_state)) - Vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ inline Vec3 random_unit_vector(curandState *local_rand_state) {
    return normalize(random_in_unit_sphere(local_rand_state));
}

__device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__device__ inline bool scatter(const Ray& r_in, const HitRecord& rec, const Material& mat, curandState *local_rand_state, Color& attenuation, Ray& scattered, Color& emitted) {
    emitted = mat.type == MAT_EMISSIVE ? mat.emission : Color(0,0,0);
    
    if (mat.type == MAT_EMISSIVE) {
        return false;
    } 
    else if (mat.type == MAT_DIFFUSE) {
        Vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);
        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }
        scattered = Ray(rec.p, scatter_direction);
        attenuation = mat.albedo.value(rec.p);
        return true;
    } 
    else if (mat.type == MAT_METAL) {
        Vec3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + mat.roughness * random_in_unit_sphere(local_rand_state));
        attenuation = mat.albedo.value(rec.p);
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    } 
    else if (mat.type == MAT_DIELECTRIC) {
        attenuation = Color(1.0f, 1.0f, 1.0f);
        float refraction_ratio = rec.front_face ? (1.0f / mat.ior) : mat.ior;

        Vec3 unit_direction = normalize(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        Vec3 direction;

        if (cannot_refract || schlick(cos_theta, refraction_ratio) > random_float(local_rand_state)) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = Ray(rec.p, direction);
        return true;
    }
    
    return false;
}
