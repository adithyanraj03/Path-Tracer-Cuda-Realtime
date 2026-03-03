#pragma once
#include "Scene.h"
#include <vector>
#include <string>
#include <cuda_runtime.h>

struct RenderState {
    int nx, ny;
    Vec3* d_framebuffer = nullptr;
    Vec3* d_accumulation_buffer = nullptr;
    Primitive* d_prims = nullptr;
    int num_prims = 0;
    Material* d_mats = nullptr;
    int num_mats = 0;
    unsigned long long frame_seed = 42;
};

void render_scene(const SceneEnv& scene, const std::string& output_filename);

// Interactive preview hooks
void init_renderer(const SceneEnv& scene, RenderState& state);
void render_frame(const SceneEnv& scene, RenderState& state, cudaArray_t target_texture, int current_spp);
void save_render_state(RenderState& state, const std::string& output_filename, int current_spp);
void cleanup_renderer(RenderState& state);
