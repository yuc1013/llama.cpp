结论：
+ 首先程序会为权重分配设备，若!no_alloc，则会真的卸载到设备，我需要创建一个容量相当小的设备（恰好为一个缓冲区的大小），并把多余权重都分配到这个设备，若!no_alloc，则要确实进行第一个来的张量的卸载，但是其它的都不进行卸载，同时属于这个设备的张量也不能进行unmap；
+ 然后程序会分配kvcache这个空白区域，注意最好是能把特殊设备和gpu一起计算，不然自己实现分配kvcache的工作会很麻烦；
+ 之后程序还会创建计算图，如果!no_alloc，这个计算图只会包含最大的buffer，注意这一步最好也是把特殊设备和gpu一起计算，但是计算图的COPY节点必须要准确插入，不然数据肯定是不对的；【但是split会导致程序自动插入不必要的结果拷贝】
+ 过后程序会进行计算图的编译，这个时候要保证计算图中准确存储了COPY节点；

新想法：
+ 首先程序会为权重分配设备，若!no_alloc，则会真的卸载到设备，我需要更改gpu的权重加载行为，n-gpu-store-layers!=n-gpu-run-layers，多余的层会被分配到gpu，但是会被分配到相同的地址空间，并且不会被运行时加载，它们的本体也不会进行unmap；
+ 然后程序会分配kvcache这个空白区域，这时正常分配；【层数*词义维度*上下文长度】
+ 之后程序还会创建计算图，如果!no_alloc，这个计算图只会包含最大的buffer，这里也没有问题；也不会产生不必要的结果拷贝；但是需要注意要在多余的层之前插入拷贝节点；
+ 过后程序会进行计算图的编译，这个时候要保证计算图中准确存储了COPY节点；

-> 入口：src->llama.cpp->llama_model_load_from_file(+1069)
    -> 调用：src->llama.cpp->llama_model_load(+830)
        -> 调用：load_arch
        -> 调用：load_hparams
        -> 调用：load_vocab
        -> 调用：load_stats
        -> 调用：src->llama-model.cpp->load_tensors[important!](+2612)
            -> pass 1: 获取后端剩余空间大小
            -> pass 2: 计算拆分点并分配(load_tensors: layer %3d assigned to device %s)(+2677)
            -> pass 3: 根据arch创建张量并引用文件偏移量（大约5000行）
            -> pass 4: 如果no_alloc=0，则调用load_all_data
            -> 调用：src->llama-model-loader.cpp->load_all_data[important!](+1399)
                -> 两个搬运线程负责搬运数据
                -> 离线数据的拷贝会被释放

-> 入口：src->llama-context.cpp->llama_init_from_model(+2922)
	-> 调用：new llama_context()--src->llama-context.cpp(+22)
		-> 调用：src->llama-context.cpp->sched_reserve(+387)
【暂存——graph-reserve和graphsplits不会编译算子】

-> 入口：src->llama-context.cpp->llama_memory_breakdown_print(+3477)【在这之前似乎都没有实际占用内存 】

【这里会编译算子，但是我不确定这是不是入口：ggml->src->ggml_metal->ggml-metal-ops.cpp->ggml_metal_op_encode(+494)】

-> 入口：src->llama-context.cpp->decode(+1239)【需要封装一个llama-batch结构体】
-> 入口：src->llama-context.cpp->synchronize(+630)
-> 入口：src->llama-context.cpp->get_logits(+766)

-> ...采样（set_sampler() & sampling_info）等