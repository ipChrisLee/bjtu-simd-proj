/**
 * @file alex-net.c
 * @brief alex-net impl, with rand input and weight generated. Rand seed can be specified in command line arguments.
Usage:
		alex-net "N" "randseed" "workspace"
	where:
		"N" is the batch number to run.
		"randseed" is rand seed used to generate data.
		"workspace" is workspace of alex-net.
 * 
 */

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "tensor.h"
#include "see.h"

// ============ see ============
See * see;

static void * see_mallocer(size_t size) {
	return see_malloc(see, size);
}

// ============ alex-net ============
// NOLINTBEGIN silence naming warning
// clang-format off
Tensor * input;

Tensor * c1_conv2d_src, * c1_conv2d_kernel;
Tensor * c1_relu_src;
Tensor * c1_maxpool2d_src;

Tensor * c2_conv2d_src, * c2_conv2d_kernel;
Tensor * c2_relu_src;
Tensor * c2_maxpool2d_src;

Tensor * c3_conv2d_src, * c3_conv2d_kernel;
Tensor * c3_relu_src;

Tensor * c4_conv2d_src, * c4_conv2d_kernel;
Tensor * c4_relu_src;

Tensor * c5_conv2d_src, * c5_conv2d_kernel;
Tensor * c5_relu_src;
Tensor * c5_maxpool2d_src;

Tensor * c6_conv2d_src, * c6_conv2d_kernel;
Tensor * c6_relu_src;

Tensor * fc7_fc_src, * fc7_fc_weight;
Tensor * fc7_relu_src;

Tensor * fc8_fc_src, * fc8_fc_weight;
Tensor * fc8_softmax_src;

Tensor * output;
// clang-format on
// NOLINTEND

void dump_feat(const char * workspace) {
	static char s_buffer[1024];
#define doit(t)                                        \
	if (true) {                                        \
		sprintf(s_buffer, "%s/" #t ".txt", workspace); \
		tensor_save_to_file(t, s_buffer, false);       \
	}
	doit(input);
	doit(c1_conv2d_src);
	doit(c1_relu_src);
	doit(c1_maxpool2d_src);
	doit(c2_conv2d_src);
	doit(c2_relu_src);
	doit(c2_maxpool2d_src);
	doit(c3_conv2d_src);
	doit(c3_relu_src);
	doit(c4_conv2d_src);
	doit(c4_relu_src);
	doit(c5_conv2d_src);
	doit(c5_relu_src);
	doit(c5_maxpool2d_src);
	doit(c6_conv2d_src);
	doit(c6_relu_src);
	doit(fc7_fc_src);
	doit(fc7_relu_src);
	doit(fc8_fc_src);
	doit(fc8_softmax_src);
	doit(output);
#undef doit
}

Tensor * new_tensor_helper(TensorRander rander, int32_t dim, ...) {
	int32_t shape[MAX_DIM];
	va_list args;
	va_start(args, dim);

	for (int32_t i = 0; i < dim; ++i) {
		int32_t len = va_arg(args, int32_t);
		shape[i] = len;
	}

	va_end(args);

	Tensor * p = tensor_new_with_macllocer(see_mallocer, dim, shape);
	if (rander) {
		tensor_fill_rand_data(p, rander);
	}
	return p;
}

static float rand_float(float mxVal) {
	assert(sizeof(int) == sizeof(float));
	int x = rand();
	int sign = rand();
	return ((sign & 1) == 1 ? -1.0f : 1.0f) * (float) x / (float) RAND_MAX / mxVal;
}

static float rand_float_1() {
	return rand_float(1);
}

static float rand_float_10() {
	return rand_float(10);
}

static float rand_float_30() {
	return rand_float(30);
}

static float rand_float_100() {
	return rand_float(100);
}

void alex_net_gen_data(unsigned int seed, const int32_t N) {
	srand(seed);
	input = new_tensor_helper(rand_float_1, 4, N, 3, 227, 227);
	c1_conv2d_kernel = new_tensor_helper(rand_float_10, 4, 96, 3, 11, 11);
	c2_conv2d_kernel = new_tensor_helper(rand_float_10, 4, 256, 96, 5, 5);
	c3_conv2d_kernel = new_tensor_helper(rand_float_30, 4, 384, 256, 3, 3);
	c4_conv2d_kernel = new_tensor_helper(rand_float_10, 4, 384, 384, 3, 3);
	c5_conv2d_kernel = new_tensor_helper(rand_float_10, 4, 256, 384, 3, 3);
	c6_conv2d_kernel = new_tensor_helper(rand_float_30, 4, 4096, 256, 6, 6);
	fc7_fc_weight = new_tensor_helper(rand_float_30, 2, 4096, 4096);
	fc8_fc_weight = new_tensor_helper(rand_float_100, 2, 4096, 4096);
}

#ifdef do_timeit
clock_t g_timeSentinel;
	#define start_timeit() \
		if (true) { g_timeSentinel = time(NULL); }
	#define timeit()                                                                          \
		if (true) {                                                                           \
			clock_t cur = time(NULL);                                                         \
			printf("At line {%d} time cost {%f}\n", __LINE__, difftime(cur, g_timeSentinel)); \
			g_timeSentinel = cur;                                                             \
		}
#else
	#define start_timeit()
	#define timeit()
#endif

void alex_net_infer(const int32_t N) {
	start_timeit();
	{
		// input = new_tensor_helper(rand_float, 4, N, 3, 227, 227);
	}
	{
		c1_conv2d_src = input;
		// c1_conv2d_kernel = new_tensor_helper(rand_float, 4, 3, 96, 11, 11);
		ConstDimArray Conv2d_Padding = {0, 0};
		ConstDimArray Conv2d_Stride = {4, 4};
		c1_relu_src = tensor_conv2d_layer(see_mallocer, c1_conv2d_src, c1_conv2d_kernel, Conv2d_Padding, Conv2d_Stride);
		c1_maxpool2d_src = tensor_relu_layer(see_mallocer, c1_relu_src);
		ConstDimArray Maxpool2d_Kernel_Size = {3, 3};
		ConstDimArray Maxpool2d_Stride = {2, 2};
		ConstDimArray Maxpool2d_Padding = {0, 0};
		c2_conv2d_src = tensor_maxpool2d_layer(see_mallocer, c1_maxpool2d_src, Maxpool2d_Kernel_Size, Maxpool2d_Stride, Maxpool2d_Padding);
	}
	timeit();
	{
		// c2_conv2d_kernel = new_tensor_helper(rand_float, 4, 96, 256, 5, 5);
		ConstDimArray Conv2d_Padding = {2, 2};
		ConstDimArray Conv2d_Stride = {1, 1};
		c2_relu_src = tensor_conv2d_layer(see_mallocer, c2_conv2d_src, c2_conv2d_kernel, Conv2d_Padding, Conv2d_Stride);
		c2_maxpool2d_src = tensor_relu_layer(see_mallocer, c2_relu_src);
		ConstDimArray Maxpool2d_Kernel_Size = {3, 3};
		ConstDimArray Maxpool2d_Stride = {2, 2};
		ConstDimArray Maxpool2d_Padding = {0, 0};
		c3_conv2d_src = tensor_maxpool2d_layer(see_mallocer, c2_maxpool2d_src, Maxpool2d_Kernel_Size, Maxpool2d_Stride, Maxpool2d_Padding);
	}
	timeit();
	{
		// c3_conv2d_kernel = new_tensor_helper(rand_float, 4, 256, 384, 3, 3);
		ConstDimArray Conv2d_Padding = {1, 1};
		ConstDimArray Conv2d_Stride = {1, 1};
		c3_relu_src = tensor_conv2d_layer(see_mallocer, c3_conv2d_src, c3_conv2d_kernel, Conv2d_Padding, Conv2d_Stride);
		c4_conv2d_src = tensor_relu_layer(see_mallocer, c3_relu_src);
	}
	timeit();
	{
		// c4_conv2d_kernel = new_tensor_helper(rand_float, 4, 384, 384, 3, 3);
		ConstDimArray Conv2d_Padding = {1, 1};
		ConstDimArray Conv2d_Stride = {1, 1};
		c4_relu_src = tensor_conv2d_layer(see_mallocer, c4_conv2d_src, c4_conv2d_kernel, Conv2d_Padding, Conv2d_Stride);
		c5_conv2d_src = tensor_relu_layer(see_mallocer, c4_relu_src);
	}
	timeit();
	{
		// c5_conv2d_kernel = new_tensor_helper(rand_float, 4, 384, 256, 3, 3);
		ConstDimArray Conv2d_Padding = {1, 1};
		ConstDimArray Conv2d_Stride = {1, 1};
		c5_relu_src = tensor_conv2d_layer(see_mallocer, c5_conv2d_src, c5_conv2d_kernel, Conv2d_Padding, Conv2d_Stride);
		c5_maxpool2d_src = tensor_relu_layer(see_mallocer, c5_relu_src);
		ConstDimArray Maxpool2d_Kernel_Size = {3, 3};
		ConstDimArray Maxpool2d_Stride = {2, 2};
		ConstDimArray Maxpool2d_Padding = {0, 0};
		c6_conv2d_src = tensor_maxpool2d_layer(see_mallocer, c5_maxpool2d_src, Maxpool2d_Kernel_Size, Maxpool2d_Stride, Maxpool2d_Padding);
	}
	timeit();
	{
		// c6_conv2d_kernel = new_tensor_helper(rand_float, 4, 256, 4096, 6, 6);
		ConstDimArray Conv2d_Padding = {0, 0};
		ConstDimArray Conv2d_Stride = {1, 1};
		c6_relu_src = tensor_conv2d_layer(see_mallocer, c6_conv2d_src, c6_conv2d_kernel, Conv2d_Padding, Conv2d_Stride);
		fc7_fc_src = tensor_relu_layer(see_mallocer, c6_relu_src);
		const int32_t Reshape_New_Dim = 2;
		DimArray reshapeNewShape = {N, 4096};
		tensor_reshape_inplace(fc7_fc_src, Reshape_New_Dim, reshapeNewShape);
	}
	timeit();
	{
		// fc7_fc_weight = new_tensor_helper(rand_float, 2, 4096, 4096);
		fc7_relu_src = tensor_fc_layer(see_mallocer, fc7_fc_src, fc7_fc_weight);
		fc8_fc_src = tensor_relu_layer(see_mallocer, fc7_relu_src);
	}
	timeit();
	{
		// fc8_fc_weight = new_tensor_helper(rand_float, 2, 4096, 4096);
		fc8_softmax_src = tensor_fc_layer(see_mallocer, fc8_fc_src, fc8_fc_weight);
		output = tensor_softmax_layer(see_mallocer, fc8_softmax_src, 1);
	}
	timeit();
}

// ============ main program ============
int g_n;
unsigned int g_seed;
const char * workspace;
char g_outputFilePath[1024];

void init(int argc, const char ** argv) {
	assert(argc == 4 && "Missing argv.");
	char * endptr;
	g_n = (int) strtol(argv[1], &endptr, 10);
	g_seed = (int) strtol(argv[2], &endptr, 10);
	workspace = argv[3];

	see = see_new(/*1G=2^30 Byte*/ (1 << 30));
}

int main(int argc, char const ** argv) {
	init(argc, argv);

	alex_net_gen_data(g_seed, g_n);
	clock_t st = time(NULL);
	alex_net_infer(g_n);
	clock_t ed = time(NULL);
	printf("%f sec used.\n", difftime(ed, st));

	dump_feat(workspace);

	return 0;
}