#pragma once
#include "Vec3.h"

// Types of texturing
enum TextureType {
    TEX_SOLID,
    TEX_CHECKER
};

struct Texture {
    TextureType type;
    Color albedo1;
    Color albedo2;
    float scale;

    __host__ __device__ Texture() {}
    __host__ __device__ Texture(Color c) : type(TEX_SOLID), albedo1(c) {}
    __host__ __device__ Texture(Color c1, Color c2, float s) : type(TEX_CHECKER), albedo1(c1), albedo2(c2), scale(s) {}

    __host__ __device__ Color value(const Point3& p) const {
        if (type == TEX_SOLID) {
            return albedo1;
        } else {
            float sines = sin(scale * p.x()) * sin(scale * p.y()) * sin(scale * p.z());
            if (sines < 0) return albedo2;
            else return albedo1;
        }
    }
};
