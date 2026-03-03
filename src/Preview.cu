#include "Preview.h"
#include "Renderer.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

using namespace std;

// ---- Input State ----
struct InputState {
    bool keys[1024] = {};
    double lastX = 0, lastY = 0;
    bool firstMouse = true;
    bool secondMouse = true; // skip 2 frames of mouse input
    float yaw = 0, pitch = 0;
    float movementSpeed = 5.0f;
    float mouseSensitivity = 0.1f;
    Point3 camPos;
};

static InputState input;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) input.keys[key] = true;
        else if (action == GLFW_RELEASE) input.keys[key] = false;
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (input.firstMouse) {
        input.lastX = xpos;
        input.lastY = ypos;
        input.firstMouse = false;
        return;
    }
    // Skip the second frame too (avoids the massive delta from cursor warp)
    if (input.secondMouse) {
        input.lastX = xpos;
        input.lastY = ypos;
        input.secondMouse = false;
        return;
    }

    float xoffset = (float)(xpos - input.lastX);
    float yoffset = (float)(input.lastY - ypos);
    input.lastX = xpos;
    input.lastY = ypos;

    // Clamp large deltas (cursor warp artifacts)
    if (fabs(xoffset) > 100.0f || fabs(yoffset) > 100.0f) return;

    xoffset *= input.mouseSensitivity;
    yoffset *= input.mouseSensitivity;

    input.yaw += xoffset;
    input.pitch += yoffset;
    if (input.pitch > 89.0f) input.pitch = 89.0f;
    if (input.pitch < -89.0f) input.pitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    input.movementSpeed *= (1.0f + 0.15f * (float)yoffset);
    if (input.movementSpeed < 0.1f) input.movementSpeed = 0.1f;
    if (input.movementSpeed > 2000.0f) input.movementSpeed = 2000.0f;
}

// ---- Shaders ----
const char* vsrc = R"(
#version 330 core
layout(location=0) in vec2 p;
layout(location=1) in vec2 t;
out vec2 uv;
void main(){ gl_Position=vec4(p,0,1); uv=t; }
)";
const char* fsrc = R"(
#version 330 core
in vec2 uv;
out vec4 c;
uniform sampler2D tex;
void main(){ c=texture(tex,uv); }
)";

void preview_loop(SceneEnv& scene) {
    int nx = scene.config.image_width;
    int ny = scene.config.image_height;

    // Extract look direction from camera geometry
    Vec3 lookdir = normalize(
        scene.camera.lower_left_corner 
        + 0.5f * scene.camera.horizontal 
        + 0.5f * scene.camera.vertical 
        - scene.camera.origin
    );

    // Use the JSON camera position directly
    input.camPos = scene.camera.origin;

    // Extract yaw/pitch from look direction
    input.yaw = atan2(lookdir.z(), lookdir.x()) * 180.0f / PT_PI;
    input.pitch = asin(fclamp(lookdir.y(), -1.0f, 1.0f)) * 180.0f / PT_PI;

    // Estimate FOV from camera geometry
    float half_height = scene.camera.vertical.length() / 2.0f;
    float focal_length = (scene.camera.lower_left_corner + 0.5f * scene.camera.horizontal + 0.5f * scene.camera.vertical - scene.camera.origin).length();
    float vfov = 2.0f * atan2(half_height, focal_length) * 180.0f / PT_PI;
    float aspect = (float)nx / (float)ny;

    // Auto-scale movement speed based on scene size
    // Use distance from camera to lookat as reference
    float scene_scale = (scene.camera.origin - (scene.camera.origin + lookdir)).length();
    // Use the camera-to-origin distance as a rough scene scale
    float cam_dist = scene.camera.origin.length();
    input.movementSpeed = fmax(cam_dist * 0.3f, 1.0f);

    // ---- GLFW ----
    if (!glfwInit()) { cerr << "GLFW init failed\n"; return; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(nx, ny, "CUDA Path Tracer - Interactive Preview", NULL, NULL);
    if (!window) { cerr << "GLFW window failed\n"; glfwTerminate(); return; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Uncap framerate
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { cerr << "GLAD failed\n"; return; }

    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // Force-center cursor to avoid initial jump
    glfwSetCursorPos(window, nx / 2.0, ny / 2.0);

    // ---- OpenGL setup ----
    GLuint vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &vsrc, NULL); glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &fsrc, NULL); glCompileShader(fs);
    GLuint prog = glCreateProgram(); glAttachShader(prog, vs); glAttachShader(prog, fs); glLinkProgram(prog);
    glDeleteShader(vs); glDeleteShader(fs);

    float verts[] = { 1,1, 1,1,  1,-1, 1,0,  -1,-1, 0,0,  -1,1, 0,1 };
    unsigned int idx[] = { 0,1,3, 1,2,3 };
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO); glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float))); glEnableVertexAttribArray(1);

    GLuint tex;
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // ---- CUDA-OpenGL interop ----
    cudaGraphicsResource* cuda_tex;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cuda_tex, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    // ---- Renderer state ----
    RenderState rstate;
    rstate.nx = nx; rstate.ny = ny;
    init_renderer(scene, rstate);

    int accumulated_frames = 0;
    float lastFrame = (float)glfwGetTime();
    float prevYaw = input.yaw, prevPitch = input.pitch;
    Point3 prevPos = input.camPos;

    cerr << "Preview started. WASD=move, Mouse=look, Scroll=speed, ESC=quit" << endl;
    cerr << "Initial speed: " << input.movementSpeed << " units/sec" << endl;

    while (!glfwWindowShouldClose(window)) {
        float now = (float)glfwGetTime();
        float dt = now - lastFrame;
        lastFrame = now;

        glfwPollEvents();

        // ---- Camera direction from yaw/pitch ----
        float yr = input.yaw * PT_PI / 180.0f;
        float pr = input.pitch * PT_PI / 180.0f;
        Vec3 front(cos(yr)*cos(pr), sin(pr), sin(yr)*cos(pr));
        front = normalize(front);
        Vec3 right = normalize(cross(front, Vec3(0, 1, 0)));
        Vec3 up = normalize(cross(right, front));

        // ---- Movement ----
        float vel = input.movementSpeed * dt;
        bool moved = false;
        if (input.keys[GLFW_KEY_W])          { input.camPos += front * vel; moved = true; }
        if (input.keys[GLFW_KEY_S])          { input.camPos -= front * vel; moved = true; }
        if (input.keys[GLFW_KEY_A])          { input.camPos -= right * vel; moved = true; }
        if (input.keys[GLFW_KEY_D])          { input.camPos += right * vel; moved = true; }
        if (input.keys[GLFW_KEY_E] || input.keys[GLFW_KEY_SPACE])      { input.camPos += up * vel; moved = true; }
        if (input.keys[GLFW_KEY_Q] || input.keys[GLFW_KEY_LEFT_SHIFT]) { input.camPos -= up * vel; moved = true; }

        // ---- Wall collision: clamp to room bounds ----
        // Room: x[-10,10], y[0.2,11.8], z[-9.8,17.8] (with margin)
        float margin = 0.3f;
        input.camPos.e[0] = fclamp(input.camPos.e[0], -10.0f + margin, 10.0f - margin);
        input.camPos.e[1] = fclamp(input.camPos.e[1], 0.0f + margin, 12.0f - margin);
        input.camPos.e[2] = fclamp(input.camPos.e[2], -10.0f + margin, 18.0f - margin);

        // Check if look direction changed
        if (fabs(input.yaw - prevYaw) > 0.01f || fabs(input.pitch - prevPitch) > 0.01f) {
            moved = true;
            prevYaw = input.yaw;
            prevPitch = input.pitch;
        }

        if (moved) {
            accumulated_frames = 0;
            prevPos = input.camPos;
            scene.camera = Camera(input.camPos, input.camPos + front, Vec3(0,1,0), vfov, aspect);
        }

        accumulated_frames++;

        // ---- Render frame ----
        cudaArray_t arr;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_tex, 0));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&arr, cuda_tex, 0, 0));
        render_frame(scene, rstate, arr, accumulated_frames);
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_tex, 0));

        // ---- Draw ----
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(window);

        // ---- Title bar ----
        char title[256];
        sprintf(title, "CUDA Path Tracer - %d SPP | Speed: %.1f u/s | %.0f FPS", 
                accumulated_frames, input.movementSpeed, 1.0f / fmax(dt, 0.001f));
        glfwSetWindowTitle(window, title);
    }

    cerr << "Saving final accumulated image (" << accumulated_frames << " SPP)..." << endl;
    save_render_state(rstate, "renders/preview_output.png", accumulated_frames);
    cerr << "Saved to renders/preview_output.png" << endl;

    cleanup_renderer(rstate);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
}
