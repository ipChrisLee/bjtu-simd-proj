#include "test-helper.h"
#include "tensor.h"

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TestResultInfo * test_result_info_new() {
	TestResultInfo * p = malloc(sizeof(TestResultInfo));
	p->testResult = Result_Unkown;
	p->info = NULL;
	return p;
}

void test_result_info_delete(TestResultInfo * p) {
	free(p);
}

static Tensor * read_tensor(FILE * fp, Msg_t * msg) {
	int32_t dim;
	fscanf(fp, "%" PRId32, &dim);
	if (dim < 0 || dim > MAX_DIM) {
		if (msg) *msg = "read_tensor: dim is not in legal range.";
		return NULL;
	}
	int32_t shape[MAX_DIM];
	for (int32_t i = 0; i < dim; ++i) {
		fscanf(fp, "%" PRId32, shape + i);
	}
	Tensor * p = tensor_new(dim, shape);
	int32_t len = tensor_get_len(p);
	for (int32_t i = 0; i < len; ++i) {
		fscanf(fp, "%f", p->data + i);
	}
	return p;
}

static DimArray * read_dim_array(FILE * fp, Msg_t * msg) {
	DimArray * a = malloc(sizeof(DimArray));
	for (int32_t i = 0; i < MAX_DIM; ++i) {
		fscanf(fp, "%" PRId32, (*a) + i);
	}
	return a;
}

static int32_t * read_i32(FILE * fp, Msg_t * msg) {
	int32_t * a = malloc(sizeof(int32_t));
	fscanf(fp, "%" PRId32, a);
	return a;
}

static float * read_f32(FILE * fp, Msg_t * msg) {
	float * f = malloc(sizeof(float));
	fscanf(fp, "%f", f);
	return f;
}

TestInfo * test_info_read_from(const char * filePath, Msg_t * msg) {
	TestInfo * p = malloc(sizeof(TestInfo));
	memset((void *) p, 0, sizeof(TestInfo));
	FILE * fp = fopen(filePath, "r");
	if (fp == NULL) {
		return NULL;
	}
	static char s_buffer[1024];

#define fun(x, y) x##y
#define process_part(field, type)                 \
	else if (strcmp(s_buffer, #field) == 0) {     \
		p->field = fun(read_, type)(fp, msg);     \
		if (p->field == NULL) { goto read_fail; } \
	}

	while (true) {
		fscanf(fp, "%s", s_buffer);
		// clang-format off
		if (strcmp("end", s_buffer) == 0) {
			break;
		}
		process_part(src, tensor)
		process_part(goldenDst, tensor)
		process_part(kernel, tensor)
		process_part(kernelSize, dim_array)
		process_part(padding, dim_array)
		process_part(stride, dim_array)
		process_part(axis, i32)
		process_part(atol, f32)
		process_part(rtol, f32)
		else {
			*msg = "Can't get field in tInfo.";
			goto read_fail;
		}
		// clang-format on
	}
#undef process_part
#undef fun

	if (p->atol && *p->atol < 0) {
		*msg = "Atol should be greater thant 0";
		goto read_fail;
	}
	if (p->rtol && *p->rtol < 0) {
		*msg = "Rtol should be greater thant 0";
		goto read_fail;
	}

	return p;
read_fail:
#define f(x)         \
	if (p->x) {      \
		free(p->x);  \
		p->x = NULL; \
	}
	f(src);
	f(goldenDst);
	f(kernel);
	f(kernelSize);
	f(padding);
	f(stride);
	f(axis);
	f(atol);
	f(rtol);
#undef f
	free(p);
	return NULL;
}

const char * tensor_compare_result_str(TensorCompareResult tensorCompareResult) {
#define enum_case(e) \
	case e: {        \
		return #e;   \
	}
	switch (tensorCompareResult) {
		enum_case(Same_In_Tol);
		enum_case(Diff_On_Shape);
		enum_case(Out_Of_Tol);
		default: {
			assert(0 && "tensor_compare_result_str: unimplemented.");
		}
	}
#undef enum_case
}

TensorCompareResult tensor_compare_passed(const Tensor * dst, const Tensor * goldenDst, float * rtol, float * atol, DimArray * diffOn) {
	assert(diffOn == NULL && "tensor_compare_passed: diffOn != NULL is to be supported.");
	if (!tensor_same_shape(dst, goldenDst)) {
		return Diff_On_Shape;
	}
	const int32_t LEN = tensor_get_len(dst);
	const float RTOL = rtol == NULL ? 0 : *rtol;
	const float ATOL = atol == NULL ? 0 : *atol;
	for (int32_t i = 0; i < LEN; ++i) {
		float c = dst->data[i];
		float g = goldenDst->data[i];
		if (fabsf(c - g) > ATOL + RTOL * fabsf(g)) {
			return Out_Of_Tol;
		}
	}
	return Same_In_Tol;
}