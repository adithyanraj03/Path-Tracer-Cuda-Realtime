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

// Global Path Tracing Kernel
__global__ void render_kernel(
    Vec3* d_framebuffer, 
    int image_width, 
    int image_height, 
    int samples_per_pixel, 
    int max_depth,
    Camera camera, 
    Primitive* d_prims, 
    int num_prims, 
    Material* d_mats,
    Color bg_color,
    unsigned long long rng_seed
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) return;

    int pixel_index = j * image_width + i;

    curandState local_rand_state;
    curand_init(rng_seed + pixel_index, 0, 0, &local_rand_state);

    Color pixel_color(0, 0, 0);

    for (int s = 0; s < samples_per_pixel; ++s) {
        float u = (i + random_float(&local_rand_state)) / (image_width - 1);
        float v = (j + random_float(&local_rand_state)) / (image_height - 1);
        
        // Ensure UV origin is bottom-left as common in tracing
        Ray cur_ray = camera.get_ray(u, 1.0f - v);
        
        Color cur_attenuation(1.0f, 1.0f, 1.0f);
        Color radiance(0.0f, 0.0f, 0.0f);
        bool done = false;

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
                    
                    // Russian roulette termination after 3 bounces
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
        pixel_color += radiance;
    }

    d_framebuffer[pixel_index] = pixel_color / float(samples_per_pixel);
}

void render_scene(const SceneEnv& scene, const std::string& output_filename) {
    int nx = scene.config.image_width;
    int ny = scene.config.image_height;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering " << nx << "x" << ny << " image with " 
              << scene.config.samples_per_pixel << " spp..." << std::endl;

    // Allocate frame buffer
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Vec3);
    Vec3* d_framebuffer;
    CUDA_CHECK(cudaMalloc((void**)&d_framebuffer, fb_size));

    // Allocate geometry and materials
    int num_prims = scene.primitives.size();
    Primitive* d_prims = nullptr;
    if (num_prims > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_prims, num_prims * sizeof(Primitive)));
        CUDA_CHECK(cudaMemcpy(d_prims, scene.primitives.data(), num_prims * sizeof(Primitive), cudaMemcpyHostToDevice));
    }

    int num_mats = scene.materials.size();
    Material* d_mats = nullptr;
    if (num_mats > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_mats, num_mats * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(d_mats, scene.materials.data(), num_mats * sizeof(Material), cudaMemcpyHostToDevice));
    }

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    // Render Kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    render_kernel<<<blocks, threads>>>(
        d_framebuffer, nx, ny, scene.config.samples_per_pixel, scene.config.max_depth, 
        scene.camera, d_prims, num_prims, d_mats, scene.config.background_color, 42
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0f;
    long long total_rays = (long long)nx * ny * scene.config.samples_per_pixel;
    double mrays_per_sec = (total_rays / 1e6) / seconds;
    
    std::cerr << "\n=== Render Statistics ===" << std::endl;
    std::cerr << "  Resolution:    " << nx << "x" << ny << std::endl;
    std::cerr << "  Samples/pixel: " << scene.config.samples_per_pixel << std::endl;
    std::cerr << "  Max depth:     " << scene.config.max_depth << std::endl;
    std::cerr << "  Total rays:    " << total_rays << std::endl;
    std::cerr << "  Render time:   " << seconds << " seconds" << std::endl;
    std::cerr << "  Throughput:    " << mrays_per_sec << " Mrays/sec" << std::endl;
    std::cerr << "========================\n" << std::endl;

    // Read back framebuffer
    std::vector<Vec3> h_framebuffer(num_pixels);
    CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer, fb_size, cudaMemcpyDeviceToHost));

    // Gamma correction and write out
    std::vector<uint8_t> image_data(num_pixels * 3);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            Vec3 col = h_framebuffer[pixel_index];

            // Gamma 2.2 correction
            col.e[0] = sqrt(fclamp(col.e[0], 0.0f, 0.999f));
            col.e[1] = sqrt(fclamp(col.e[1], 0.0f, 0.999f));
            col.e[2] = sqrt(fclamp(col.e[2], 0.0f, 0.999f));

            int idx = (j * nx + i) * 3;
            image_data[idx + 0] = static_cast<uint8_t>(255.99f * col.e[0]);
            image_data[idx + 1] = static_cast<uint8_t>(255.99f * col.e[1]);
            image_data[idx + 2] = static_cast<uint8_t>(255.99f * col.e[2]);
        }
    }

    stbi_write_png(output_filename.c_str(), nx, ny, 3, image_data.data(), nx * 3);
    std::cerr << "Saved image to " << output_filename << std::endl;

    // PPM Output 
    std::string ppm_filename = "renders/output.ppm";
    if (output_filename.find_last_of('.') != std::string::npos) {
        ppm_filename = output_filename.substr(0, output_filename.find_last_of('.')) + ".ppm";
    }
    std::ofstream ppm(ppm_filename);
    ppm << "P3\n" << nx << ' ' << ny << "\n255\n";
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = (j * nx + i) * 3;
            ppm << (int)image_data[idx] << ' ' << (int)image_data[idx+1] << ' ' << (int)image_data[idx+2] << '\n';
        }
    }
    std::cerr << "Saved image to " << ppm_filename << std::endl;

    // Cleanup
    cudaFree(d_framebuffer);
    if (d_prims) cudaFree(d_prims);
    if (d_mats) cudaFree(d_mats);
}
