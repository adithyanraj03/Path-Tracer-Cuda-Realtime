#include <iostream>
#include <string>
#include <cstdlib>
#include "Scene.h"
#include "Renderer.h"

int main(int argc, char** argv) {
    std::string scene_filename = "config/cornell_box.json";
    std::string output_filename = "renders/output.png";

    if (argc > 1) {
        scene_filename = argv[1];
    }
    if (argc > 2) {
        output_filename = argv[2];
    }

    SceneEnv scene;
    if (!scene.load_from_json(scene_filename)) {
        std::cerr << "Failed to load scene." << std::endl;
        return 1;
    }

    scene.build_bvh(); // Ensure CPU BVH is ready to serialize

    render_scene(scene, output_filename);

    return 0;
}
