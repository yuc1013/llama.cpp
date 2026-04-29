#ifndef STINGY_H
#define STINGY_H

#include "ggml.h"

#ifdef _WIN32
    #ifdef LLAMA_BUILD
        #define STINGY_API __declspec(dllexport)
    #else
        #define STINGY_API __declspec(dllimport)
    #endif
#else
    #define STINGY_API
#endif

const int DISABLE = 0;
const int MODEA = 1;
const int MODEB = 2;

const int MODE = MODEB;

#define STINGY

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations

STINGY_API void init_stingy(int nl, int ngl);
STINGY_API bool use_stingy();
STINGY_API int blk_id(const char * tensor_name);
STINGY_API void share_mem(ggml_tensor * t, const ggml_tensor * src);
STINGY_API bool is_backup(const char* tensor_name);
STINGY_API void print_stingy_data();

// Getter and Setter functions
STINGY_API int get_s_use_stingy();
STINGY_API void set_s_use_stingy(int value);
STINGY_API int get_s_nl();
STINGY_API void set_s_nl(int value);
STINGY_API int get_s_ngl();
STINGY_API void set_s_ngl(int value);
STINGY_API int get_s_ngls();
STINGY_API void set_s_ngls(int value);
STINGY_API int get_s_n_B_start();
STINGY_API void set_s_n_B_start(int value);
STINGY_API int get_s_n_C_start();
STINGY_API void set_s_n_C_start(int value);
STINGY_API const void * get_s_tensors_by_name();
STINGY_API void set_s_tensors_by_name(void* value);

#ifdef __cplusplus
}
#endif

#endif // STINGY_H