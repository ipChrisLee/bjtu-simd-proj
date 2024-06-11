#pragma once

#include <stdint.h>
#include <stdbool.h>

#define MAX_DIM 4

typedef struct {
	int32_t dim;
	int32_t shape[MAX_DIM];
	float data[];
} Tensor;


// ------------ ctor dtor ------------
Tensor * tensor_new(int32_t dim, const int32_t shape[MAX_DIM]);
void tensor_delete(Tensor * p);

// ------------ method ------------
inline int32_t tensor_get_len(Tensor * p) {
	int32_t len = 1;
	for (int32_t i=0; i<p->dim; ++i) {
		len *= p->shape[i];
	}
	return len;
}

// ------------ static method: layer ------------
void tensor_relu(Tensor * dst, Tensor * src);
void tensor_relu_inplace(Tensor * op);