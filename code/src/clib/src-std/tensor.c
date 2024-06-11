#include "tensor.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int32_t get_len_from_shape(int32_t dim, const int32_t shape[MAX_DIM]) {
	int32_t len = 1;
	for (int32_t i = 0; i < dim; ++i) {
		len *= shape[i];
	}
	return len;
}

Tensor * tensor_new(int32_t dim, const int32_t shape[MAX_DIM]) {
	size_t memToUse = sizeof(Tensor) + (uint32_t) get_len_from_shape(dim, shape);
	Tensor * p = malloc(memToUse);
	p->dim = dim;
	memcpy(p->shape, shape, sizeof(p->shape));
	return p;
}

void tensor_delete(Tensor * p) {
	free(p);
}

void tensor_relu(Tensor * dst, Tensor * src) {
#ifdef EXPENSIVE_CHECK
	assert(dst->dim == src->dim && "For relu, src and dst should have same shape.");
	for (int32_t i = 0; i < src->dim; ++i) {
		assert(dst->shape[i] == src->shape[i] && "For relu, src and dst should have same shape.");
	}
#endif
	int32_t len = tensor_get_len(src);
	for (int32_t i = 0; i < len; ++i) {
		float v = src->data[i];
		v = fminf(v, 0);
		dst->data[i] = v;
	}
}

void tensor_relu_inplace(Tensor * op) {
	int32_t len = tensor_get_len(op);
	for (int32_t i = 0; i < len; ++i) {
		float v = op->data[i];
		v = fminf(v, 0);
		op->data[i] = v;
	}
}