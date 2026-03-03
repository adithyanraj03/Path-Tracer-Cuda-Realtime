#pragma once
#include "Primitive.h"
#include <vector>
#include <algorithm>

struct BVHNode {
    AABB box;
    int left;    // Index of left child node or start primitive
    int right;   // Index of right child node or primitives count
    bool is_leaf;
};

class BVHBuilder {
public:
    std::vector<BVHNode> nodes;
    std::vector<Primitive> flat_primitives;

    BVHBuilder(std::vector<Primitive>& prims) {
        flat_primitives = prims;
        if (prims.empty()) return;
        
        nodes.reserve(prims.size() * 2);
        build_recursive(0, flat_primitives.size());
    }

private:
    int build_recursive(int start, int end) {
        BVHNode node;
        node.is_leaf = false;
        
        // Compute bounding box
        AABB bounds;
        if (start < end) {
            bounds = flat_primitives[start].bounding_box();
            for (int i = start + 1; i < end; i++) {
                bounds = surrounding_box(bounds, flat_primitives[i].bounding_box());
            }
        }
        node.box = bounds;

        int num_prims = end - start;

        if (num_prims <= 2) {
            node.is_leaf = true;
            node.left = start;       // first primitive idx
            node.right = num_prims;  // primitive count
            nodes.push_back(node);
            return nodes.size() - 1;
        }

        // Midpoint split on longest axis
        Vec3 extent = bounds.max() - bounds.min();
        int axis = 0;
        if (extent.y() > extent.x()) axis = 1;
        if (extent.z() > extent.e[axis]) axis = 2;

        std::sort(flat_primitives.begin() + start, flat_primitives.begin() + end, 
            [axis](const Primitive& a, const Primitive& b) {
                return a.bounding_box().min().e[axis] < b.bounding_box().min().e[axis];
            }
        );

        int mid = start + num_prims / 2;
        
        int node_idx = nodes.size();
        nodes.push_back(node); // placeholder
        
        int left_child = build_recursive(start, mid);
        int right_child = build_recursive(mid, end);
        
        nodes[node_idx].left = left_child;
        nodes[node_idx].right = right_child;
        nodes[node_idx].box = surrounding_box(nodes[left_child].box, nodes[right_child].box);

        return node_idx;
    }
};

// GPU Traversal (Iterative Stack)
__device__ inline bool hit_bvh(const BVHNode* d_nodes, const Primitive* d_prims, const Ray& r, float t_min, float t_max, HitRecord& rec) {
    if (!d_nodes) return false;

    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Root node is 0

    bool hit_anything = false;
    float closest_t = t_max;
    HitRecord temp_rec;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = d_nodes[node_idx];

        if (node.box.hit(r, t_min, closest_t)) {
            if (node.is_leaf) {
                for (int i = 0; i < node.right; i++) {
                    int prim_idx = node.left + i;
                    if (d_prims[prim_idx].hit(r, t_min, closest_t, temp_rec)) {
                        hit_anything = true;
                        closest_t = temp_rec.t;
                        rec = temp_rec;
                    }
                }
            } else {
                stack[stack_ptr++] = node.left;
                stack[stack_ptr++] = node.right;
            }
        }
    }

    return hit_anything;
}
