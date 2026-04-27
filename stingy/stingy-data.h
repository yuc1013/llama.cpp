#ifndef STINGY_H
#define STINGY_H

#include "ggml.h"
#include <vector>
#include <string>
#include <utility>

// --- 全局变量声明 (使用 extern，定义在 .cpp 中) ---

extern int s_use_stingy;
extern int s_nl;
extern int s_ngl;
extern int s_ngls;
extern int s_n_B_start;
extern int s_n_C_start;

// 注意：此处声明指针本身
extern std::vector<std::pair<std::string, struct ggml_tensor *>> * s_tensors_by_name;

#endif // STINGY_H