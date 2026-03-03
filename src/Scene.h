#pragma once
#include <vector>
#include "Camera.h"
#include "Primitive.h"
#include "Material.h"
#include "BVH.h"
#include <string>

struct SceneConfig {
    int image_width = 800;
    int image_height = 800;
    int samples_per_pixel = 64;
    int max_depth = 50;
    Color background_color = Color(0, 0, 0);
};

struct SceneEnv {
    SceneConfig config;
    Camera camera;
    std::vector<Primitive> primitives;
    std::vector<Material> materials;
    std::vector<BVHNode> bvh_nodes;

    bool load_from_json(const std::string& filename);
    void build_bvh();
};
