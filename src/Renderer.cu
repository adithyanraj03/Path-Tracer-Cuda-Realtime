#include "Renderer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "stb_image_write.h"
#include "utils.h"

// Device function to find closest primitive hit
__device__ bool hit_world(const Ray& r, const Primitive* d_prims, int num_prims, float t_min, float t_max, HitRecord& rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_t = t_max;

    for (int i = 0; i < num_prims; i++) {
        if (d_prims[i].hit(r, t_min, closest_t, temp_rec)) {
            hit_anything = true;
            closest_t = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

// Global Path Tracing Kernel (1 SPP per call, accumulates)
__global__ void render_accum_kernel(
    Vec3* d_accumulation_buffer, 
    int image_width, 
    int image_height, 
    int max_depth,
    Camera camera, 
    Primitive* d_prims, 
    int num_prims, 
    Material* d_mats,
    Color bg_color,
    unsigned long long rng_seed,
    bool reset
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) return;

    int pixel_index = j * image_width + i;

    curandState local_rand_state;
    curand_init(rng_seed + pixel_index, 0, 0, &local_rand_state);

    float u = (i + random_float(&local_rand_state)) / (image_width - 1);
    float v = (j + random_float(&local_rand_state)) / (image_height - 1);
    
    // Ensure UV origin is bottom-left
    Ray cur_ray = camera.get_ray(u, 1.0f - v);
    
    Color cur_attenuation(1.0f, 1.0f, 1.0f);
    Color radiance(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        if (hit_world(cur_ray, d_prims, num_prims, 0.001f, PT_INFINITY, rec)) {
            Ray scattered;
            Color attenuation;
            Color emitted;
            
            Material mat = d_mats[rec.material_idx];
            bool scatters = scatter(cur_ray, rec, mat, &local_rand_state, attenuation, scattered, emitted);
            
            radiance += cur_attenuation * emitted;
            
            if (scatters) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
                
                if (depth > 3) {
                    float p = fmax(cur_attenuation.x(), fmax(cur_attenuation.y(), cur_attenuation.z()));
                    if (random_float(&local_rand_state) > p) {
                        break;
                    }
                    cur_attenuation /= p;
                }
            } else {
                break;
            }
        } else {
            radiance += cur_attenuation * bg_color;
            break;
        }
    }

    if (reset) {
        d_accumulation_buffer[pixel_index] = radiance;
    } else {
        d_accumulation_buffer[pixel_index] += radiance;
    }
}

// Convert accumulated HDR buffer to standard 8-bit RGBA for OpenGL/saving
__global__ void format_output_kernel(
    Vec3* d_accumulation_buffer,
    uchar4* d_output_texture, // Used for OpenGL interop
    Vec3* d_output_framebuffer, // Used for saving to disk
    int image_width,
    int image_height,
    int accumulated_frames
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) return;
    int pixel_index = j * image_width + i;

    Vec3 col = d_accumulation_buffer[pixel_index] / float(accumulated_frames);
    col.e[0] = sqrt(fclamp(col.e[0], 0.0f, 0.999f));
    col.e[1] = sqrt(fclamp(col.e[1], 0.0f, 0.999f));
    col.e[2] = sqrt(fclamp(col.e[2], 0.0f, 0.999f));

    if (d_output_texture) {
        uchar4 rgba;
        rgba.x = static_cast<unsigned char>(255.99f * col.e[0]);
        rgba.y = static_cast<unsigned char>(255.99f * col.e[1]);
        rgba.z = static_cast<unsigned char>(255.99f * col.e[2]);
        rgba.w = 255;
        // OpenGL expects lower-left origin, CUDA interop writes top-left to surface/array
        // We handle coordinate matching in the shader, so we just write linearly or appropriately
        // Actually for a cudaArray_t, OpenGL coordinates are top-down depending on how we map it.
        // We'll write to a raw buffer, or directly via surface write. But cudaArray doesn't map easily as flat pointer.
        // Wait, for this we'll need surface writes if using cudaArray.
    }
    
    if (d_output_framebuffer) {
        d_output_framebuffer[pixel_index] = col;
    }
}

// Write to surface object (for OpenGL interop)
__global__ void write_surface_kernel(
    Vec3* d_accumulation_buffer,
    cudaSurfaceObject_t surface,
    int image_width,
    int image_height,
    int accumulated_frames
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) return;
    int pixel_index = j * image_width + i;

    Vec3 col = d_accumulation_buffer[pixel_index] / float(accumulated_frames);
    col.e[0] = sqrt(fclamp(col.e[0], 0.0f, 0.999f));
    col.e[1] = sqrt(fclamp(col.e[1], 0.0f, 0.999f));
    col.e[2] = sqrt(fclamp(col.e[2], 0.0f, 0.999f));

    uchar4 rgba;
    rgba.x = static_cast<unsigned char>(255.99f * col.e[0]);
    rgba.y = static_cast<unsigned char>(255.99f * col.e[1]);
    rgba.z = static_cast<unsigned char>(255.99f * col.e[2]);
    rgba.w = 255;

    // Flip Y: OpenGL (0,0) = bottom-left, kernel j=0 = top row
    surf2Dwrite(rgba, surface, i * sizeof(uchar4), (image_height - 1 - j));
}

void init_renderer(const SceneEnv& scene, RenderState& state) {
    state.num_prims = scene.primitives.size();
    if (state.num_prims > 0) {
        CUDA_CHECK(cudaMalloc((void**)&state.d_prims, state.num_prims * sizeof(Primitive)));
        CUDA_CHECK(cudaMemcpy(state.d_prims, scene.primitives.data(), state.num_prims * sizeof(Primitive), cudaMemcpyHostToDevice));
    }

    state.num_mats = scene.materials.size();
    if (state.num_mats > 0) {
        CUDA_CHECK(cudaMalloc((void**)&state.d_mats, state.num_mats * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(state.d_mats, scene.materials.data(), state.num_mats * sizeof(Material), cudaMemcpyHostToDevice));
    }

    size_t fb_size = state.nx * state.ny * sizeof(Vec3);
    CUDA_CHECK(cudaMalloc((void**)&state.d_framebuffer, fb_size));
    CUDA_CHECK(cudaMalloc((void**)&state.d_accumulation_buffer, fb_size));
    state.frame_seed = 42;
}

void render_frame(const SceneEnv& scene, RenderState& state, cudaArray_t target_texture, int current_spp) {
    int tx = 8, ty = 8;
    dim3 blocks(state.nx / tx + 1, state.ny / ty + 1);
    dim3 threads(tx, ty);

    bool reset = (current_spp == 1);
    state.frame_seed += state.nx * state.ny;

    render_accum_kernel<<<blocks, threads>>>(
        state.d_accumulation_buffer, state.nx, state.ny, scene.config.max_depth,
        scene.camera, state.d_prims, state.num_prims, state.d_mats, scene.config.background_color,
        state.frame_seed, reset
    );

    if (target_texture != nullptr) {
        // Create surface object from array
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = target_texture;

        cudaSurfaceObject_t surface;
        CUDA_CHECK(cudaCreateSurfaceObject(&surface, &resDesc));

        write_surface_kernel<<<blocks, threads>>>(
            state.d_accumulation_buffer, surface, state.nx, state.ny, current_spp
        );
        CUDA_CHECK(cudaDestroySurfaceObject(surface));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void save_render_state(RenderState& state, const std::string& output_filename, int current_spp) {
    if (!state.d_framebuffer || !state.d_accumulation_buffer) return;

    int tx = 8, ty = 8;
    dim3 blocks(state.nx / tx + 1, state.ny / ty + 1);
    dim3 threads(tx, ty);

    format_output_kernel<<<blocks, threads>>>(
        state.d_accumulation_buffer, nullptr, state.d_framebuffer, 
        state.nx, state.ny, current_spp
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_pixels = state.nx * state.ny;
    std::vector<Vec3> h_framebuffer(num_pixels);
    CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), state.d_framebuffer, num_pixels * sizeof(Vec3), cudaMemcpyDeviceToHost));

    std::vector<uint8_t> image_data(num_pixels * 3);
    for (int j = 0; j < state.ny; j++) {
        for (int i = 0; i < state.nx; i++) {
            Vec3 col = h_framebuffer[j * state.nx + i];
            int idx = (j * state.nx + i) * 3;
            image_data[idx + 0] = static_cast<uint8_t>(255.99f * col.e[0]);
            image_data[idx + 1] = static_cast<uint8_t>(255.99f * col.e[1]);
            image_data[idx + 2] = static_cast<uint8_t>(255.99f * col.e[2]);
        }
    }

    stbi_write_png(output_filename.c_str(), state.nx, state.ny, 3, image_data.data(), state.nx * 3);
}

void cleanup_renderer(RenderState& state) {
    if (state.d_framebuffer) cudaFree(state.d_framebuffer);
    if (state.d_accumulation_buffer) cudaFree(state.d_accumulation_buffer);
    if (state.d_prims) cudaFree(state.d_prims);
    if (state.d_mats) cudaFree(state.d_mats);
}

void render_scene(const SceneEnv& scene, const std::string& output_filename) {
    RenderState state;
    state.nx = scene.config.image_width;
    state.ny = scene.config.image_height;

    std::cerr << "Rendering " << state.nx << "x" << state.ny << " image with " 
              << scene.config.samples_per_pixel << " spp..." << std::endl;

    init_renderer(scene, state);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Render loop
    for (int i = 1; i <= scene.config.samples_per_pixel; i++) {
        render_frame(scene, state, nullptr, i);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Format output to frame buffer for saving
    int tx = 8, ty = 8;
    dim3 blocks(state.nx / tx + 1, state.ny / ty + 1);
    dim3 threads(tx, ty);
    format_output_kernel<<<blocks, threads>>>(
        state.d_accumulation_buffer, nullptr, state.d_framebuffer, 
        state.nx, state.ny, scene.config.samples_per_pixel
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Stats
    float seconds = milliseconds / 1000.0f;
    long long total_rays = (long long)state.nx * state.ny * scene.config.samples_per_pixel;
    double mrays_per_sec = (total_rays / 1e6) / seconds;
    
    std::cerr << "\n=== Render Statistics ===" << std::endl;
    std::cerr << "  Resolution:    " << state.nx << "x" << state.ny << std::endl;
    std::cerr << "  Samples/pixel: " << scene.config.samples_per_pixel << std::endl;
    std::cerr << "  Max depth:     " << scene.config.max_depth << std::endl;
    std::cerr << "  Total rays:    " << total_rays << std::endl;
    std::cerr << "  Render time:   " << seconds << " seconds" << std::endl;
    std::cerr << "  Throughput:    " << mrays_per_sec << " Mrays/sec" << std::endl;
    std::cerr << "========================\n" << std::endl;

    // Save
    int num_pixels = state.nx * state.ny;
    std::vector<Vec3> h_framebuffer(num_pixels);
    CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), state.d_framebuffer, num_pixels * sizeof(Vec3), cudaMemcpyDeviceToHost));

    std::vector<uint8_t> image_data(num_pixels * 3);
    for (int j = 0; j < state.ny; j++) {
        for (int i = 0; i < state.nx; i++) {
            Vec3 col = h_framebuffer[j * state.nx + i];
            int idx = (j * state.nx + i) * 3;
            image_data[idx + 0] = static_cast<uint8_t>(255.99f * col.e[0]);
            image_data[idx + 1] = static_cast<uint8_t>(255.99f * col.e[1]);
            image_data[idx + 2] = static_cast<uint8_t>(255.99f * col.e[2]);
        }
    }

    stbi_write_png(output_filename.c_str(), state.nx, state.ny, 3, image_data.data(), state.nx * 3);
    std::cerr << "Saved image to " << output_filename << std::endl;

    cleanup_renderer(state);
}
