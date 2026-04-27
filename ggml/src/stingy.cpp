#include "ggml.h"
#include "stingy.h"

#include <vector>
#include <string>

int s_use_stingy;
int s_nl;
int s_ngl;
int s_ngls;
int s_n_B_start;
int s_n_C_start;

std::vector<std::pair<std::string, struct ggml_tensor *>> * s_tensors_by_name;

void init_stingy(int nl, int ngl) {
    s_nl = nl;
    s_ngl = ngl;
    // if invalid, load from env
    const char * n_gpu_layers_stingy_raw = std::getenv("N_GPU_LAYERS_STINGY");
    if (n_gpu_layers_stingy_raw == nullptr) {
        printf("[STINGY] fail: env not set\n");
        s_use_stingy = false;
        return;
    }
    s_ngls = std::stoi(n_gpu_layers_stingy_raw);
    if (s_ngls < 0) {
        printf("[STINGY] fail: ngls < 0\n");
        s_use_stingy = false;
        return;
    }
    if (ngl - s_ngls < 2) {
        printf("[STINGY] fail: ngl-ngls < 2\n");
        s_use_stingy = false;
        return;
    }
    // stingy: ngl--n_gpu_layers, ngls--n_gpu_layers_stingy
    // stingy: A={0, 1, ..., -ngl-1} store/calc on cpu, B={-ngl, -ngl+1, ..., -ngls-1} store on cpu but calc on gpu, C={-ngls, ..., -1} store/calc on gpu
    s_n_B_start = std::max(int(nl) + 1 - ngl, 0);
    s_n_C_start = std::max(int(nl) + 1 - s_ngls, 0);
    printf("[STINGY] open\n");
    s_use_stingy = true;
}

bool use_stingy() {
    return s_use_stingy;
}

///////////////////////////////////////////////////////

// stingy: load fewer than need
int blk_id(const char * tensor_name) {
    int ret = -1;
    if (strncmp(tensor_name, "blk.", 4) != 0 || sscanf(tensor_name + 4, "%d", &ret) != 1) {
        return -1;
    }
    return ret;
}

// stingy: load fewer than need
std::string name_with_bid(const char * tensor_name, int bid) {
    char buf[GGML_MAX_NAME];
    
    int layer_idx = 0;
    char suffix[GGML_MAX_NAME];
    
    if (sscanf(tensor_name, "blk.%d.%s", &layer_idx, suffix) == 2) {
        snprintf(buf, sizeof(buf), "blk.%d.%s", bid, suffix);
        return std::string(buf);
    }
    
    return std::string(tensor_name);
}

// stingy: load fewer than need
void share_mem(ggml_tensor * t, const ggml_tensor * src) {
    t->buffer = src->buffer;

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        t->nb[i] = src->nb[i];
    }

    t->flags = src->flags;

    t->data = src->data;
    t->extra = src->extra;
    strcpy(t->padding, src->padding);
}

// stingy: load fewer than need
bool is_backup(const char* tensor_name) {
    if (tensor_name == nullptr) return false;

    size_t len = std::strlen(tensor_name);
    const char* suffix = ".backup";
    size_t suffix_len = std::strlen(suffix);

    if (len >= suffix_len) {
        return std::strcmp(tensor_name + len - suffix_len, suffix) == 0;
    }
    return false;
}

void print_stingy_data() {
    printf("\n----------- [STINGY] Configuration -----------\n");
    printf("Status:        %s\n", s_use_stingy ? "ENABLED" : "DISABLED");
    
    if (s_use_stingy) {
        printf("Layers (nl):   %d\n", s_nl);
        printf("GPU Layers:    %d (Total Offloaded)\n", s_ngl);
        printf("Stingy NGLS:   %d (Direct GPU storage)\n", s_ngls);
        printf("----------------------------------------------\n");
        
        // 这里的逻辑对应你代码中的注释：
        // A: 0 到 s_n_B_start - 1 (纯 CPU)
        // B: s_n_B_start 到 s_n_C_start - 1 (CPU 存储，GPU 计算)
        // C: s_n_C_start 到 s_nl (纯 GPU)
        
        printf("Zone A (CPU):      [0, %d)\n", s_n_B_start);
        printf("Zone B (CPU->GPU): [%d, %d)\n", s_n_B_start, s_n_C_start);
        printf("Zone C (GPU):      [%d, %d]\n", s_n_C_start, s_nl);
        
    }
    printf("----------------------------------------------\n\n");
}

// Getter and Setter functions for global variables

int get_s_use_stingy() {
    return s_use_stingy;
}

void set_s_use_stingy(int value) {
    s_use_stingy = value;
}

int get_s_nl() {
    return s_nl;
}

void set_s_nl(int value) {
    s_nl = value;
}

int get_s_ngl() {
    return s_ngl;
}

void set_s_ngl(int value) {
    s_ngl = value;
}

int get_s_ngls() {
    return s_ngls;
}

void set_s_ngls(int value) {
    s_ngls = value;
}

int get_s_n_B_start() {
    return s_n_B_start;
}

void set_s_n_B_start(int value) {
    s_n_B_start = value;
}

int get_s_n_C_start() {
    return s_n_C_start;
}

void set_s_n_C_start(int value) {
    s_n_C_start = value;
}

const void* get_s_tensors_by_name() {
    return s_tensors_by_name;
}

void set_s_tensors_by_name(void * value) {
    s_tensors_by_name = (std::vector<std::pair<std::string, struct ggml_tensor *>> *)value;
}