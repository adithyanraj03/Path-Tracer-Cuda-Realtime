#include "Scene.h"
#include <fstream>
#include <iostream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

bool SceneEnv::load_from_json(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open scene config: " << filename << std::endl;
        return false;
    }
    
    json j;
    try {
        file >> j;
    } catch(const std::exception& e) {
        std::cerr << "JSON Parse error: " << e.what() << std::endl;
        return false;
    }

    if (j.contains("image_width")) config.image_width = j["image_width"];
    if (j.contains("image_height")) config.image_height = j["image_height"];
    if (j.contains("samples_per_pixel")) config.samples_per_pixel = j["samples_per_pixel"];
    if (j.contains("max_depth")) config.max_depth = j["max_depth"];
    if (j.contains("background_color")) {
        auto bg = j["background_color"];
        config.background_color = Color(bg[0], bg[1], bg[2]);
    }

    if (j.contains("camera")) {
        auto cam = j["camera"];
        Point3 lookfrom = Point3(cam["lookfrom"][0], cam["lookfrom"][1], cam["lookfrom"][2]);
        Point3 lookat = Point3(cam["lookat"][0], cam["lookat"][1], cam["lookat"][2]);
        Vec3 vup = Vec3(cam["vup"][0], cam["vup"][1], cam["vup"][2]);
        float vfov = cam["vfov"];
        float aspect = float(config.image_width) / float(config.image_height);
        camera = Camera(lookfrom, lookat, vup, vfov, aspect);
    }

    if (j.contains("materials")) {
        for (const auto& mat_json : j["materials"]) {
            std::string type = mat_json["type"];
            if (type == "diffuse") {
                auto albedo = mat_json["albedo"];
                materials.push_back(Material::Diffuse(Texture(Color(albedo[0], albedo[1], albedo[2]))));
            } else if (type == "metal") {
                auto albedo = mat_json["albedo"];
                float fuzz = mat_json["fuzz"];
                materials.push_back(Material::Metal(Texture(Color(albedo[0], albedo[1], albedo[2])), fuzz));
            } else if (type == "dielectric") {
                float ior = mat_json["ior"];
                materials.push_back(Material::Dielectric(ior));
            } else if (type == "emissive") {
                auto emission = mat_json["emission"];
                materials.push_back(Material::Emissive(Color(emission[0], emission[1], emission[2])));
            }
        }
    }

    if (j.contains("objects")) {
        for (const auto& obj_json : j["objects"]) {
            std::string type = obj_json["type"];
            int mat_idx = obj_json["material_index"];
            if (type == "sphere") {
                auto c = obj_json["center"];
                float r = obj_json["radius"];
                primitives.push_back(Primitive(Sphere(Point3(c[0], c[1], c[2]), r, mat_idx)));
            } else if (type == "triangle") {
                auto v0j = obj_json["v0"];
                auto v1j = obj_json["v1"];
                auto v2j = obj_json["v2"];
                Point3 p0 = Point3(v0j[0], v0j[1], v0j[2]);
                Point3 p1 = Point3(v1j[0], v1j[1], v1j[2]);
                Point3 p2 = Point3(v2j[0], v2j[1], v2j[2]);
                primitives.push_back(Primitive(Triangle(p0, p1, p2, mat_idx)));
            }
        }
    }
    
    return true;
}

void SceneEnv::build_bvh() {
    BVHBuilder builder(primitives);
    bvh_nodes = builder.nodes;
}
