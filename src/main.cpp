#include <iostream>
#include <string>
#include <cstdlib>
#include "Scene.h"
#include "Renderer.h"
#include "Preview.h"

int main(int argc, char** argv) {
    std::string scene_filename = "config/cornell_box.json";
    std::string output_filename = "renders/output.png";
    bool preview = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--preview") {
            preview = true;
        } else if (scene_filename == "config/cornell_box.json" && arg.find(".json") != std::string::npos) {
            scene_filename = arg;
        } else if (output_filename == "renders/output.png" && arg.find(".png") != std::string::npos) {
            output_filename = arg;
        }
    }

    SceneEnv scene;
    if (!scene.load_from_json(scene_filename)) {
        std::cerr << "Failed to load scene." << std::endl;
        return 1;
    }

    scene.build_bvh(); // Ensure CPU BVH is ready to serialize

    if (preview) {
        preview_loop(scene);
    } else {
        render_scene(scene, output_filename);
    }

    return 0;
}
