#ifndef STINGY_H
#define STINGY_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations

void init_stingy(int nl, int ngl);
bool use_stingy();
int blk_id(const char * tensor_name);
void share_mem(ggml_tensor * t, const ggml_tensor * src);
bool is_backup(const char* tensor_name);
void print_stingy_data();

// Getter and Setter functions
int get_s_use_stingy();
void set_s_use_stingy(int value);
int get_s_nl();
void set_s_nl(int value);
int get_s_ngl();
void set_s_ngl(int value);
int get_s_ngls();
void set_s_ngls(int value);
int get_s_n_B_start();
void set_s_n_B_start(int value);
int get_s_n_C_start();
void set_s_n_C_start(int value);
const void * get_s_tensors_by_name();
void set_s_tensors_by_name(void* value);

#ifdef __cplusplus
}
#endif

#endif // STINGY_H