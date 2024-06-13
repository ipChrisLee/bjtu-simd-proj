#include "tensor.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <arm_neon.h>

static inline int32_t r4(int32_t x) {
	return (x) & (int32_t) 0xFFFFFFFC;
}

static inline float reduce_sum(float32x4_t f) {
	float32x2_t r = vadd_f32(vget_high_f32(f), vget_low_f32(f));
	return vget_lane_f32(vpadd_f32(r, r), 0);
}

static inline float reduce_max(float32x4_t f) {
	float32x2_t r = vmax_f32(vget_high_f32(f), vget_low_f32(f));
	return vget_lane_f32(vpmax_f32(r, r), 0);
}

Tensor * tensor_new(int32_t dim, const int32_t shape[MAX_DIM]) {
	size_t memToUse = sizeof(Tensor) + sizeof(float) * (size_t) get_len_from_shape(dim, shape);
	Tensor * p = malloc(memToUse);
	p->dim = dim;
	memcpy(p->shape, shape, sizeof(p->shape));
	return p;
}

void tensor_delete(Tensor * p) {
	free(p);
}

Tensor * tensor_new_with_macllocer(TensorMallocer mallocer, int32_t dim, const int32_t shape[MAX_DIM]) {
	size_t memToUse = sizeof(Tensor) + sizeof(float) * (size_t) get_len_from_shape(dim, shape);
	Tensor * p = mallocer(memToUse);
	p->dim = dim;
	memcpy(p->shape, shape, sizeof(p->shape));
	return p;
}

void tensor_delete_with_freer(TensorFreer freer, Tensor * p) {
	freer(p);
}

bool tensor_same_shape(const Tensor * lhs, const Tensor * rhs) {
	if (lhs->dim != rhs->dim) {
		return false;
	}
	const int32_t D = lhs->dim;
	for (int32_t i = 0; i < D; ++i) {
		if (lhs->shape[i] != rhs->shape[i]) {
			return false;
		}
	}
	return true;
}

void tensor_memcpy(Tensor * dst, const Tensor * src) {
	int32_t lenDst = tensor_get_len(dst);
	int32_t lenSrc = tensor_get_len(src);
	assert(lenDst == lenSrc && "tensor_memcpy requires src and dst data length same.");
	memcpy(dst->data, src->data, sizeof(float) * (size_t) lenSrc);
}

void tensor_relu(Tensor * dst, const Tensor * src) {
	tensor_relu_check(dst, src);
	const int32_t Len = tensor_get_len(src);
	float32x4_t zero = vdupq_n_f32(0.0f);
	for (int32_t i = 0; i + 4 <= Len; i += 4) {
		float32x4_t v = vld1q_f32(src->data + i);
		v = vmaxnmq_f32(v, zero);
		vst1q_f32(dst->data + i, v);
	}
	for (int32_t i = r4(Len); i < Len; ++i) {
		float v = src->data[i];
		v = fmaxf(v, 0.0f);
		dst->data[i] = v;
	}
}

void tensor_relu_inplace(Tensor * op) {
	tensor_relu_inplace_check(op);
	const int32_t Len = tensor_get_len(op);
	float32x4_t zero = vdupq_n_f32(0.0f);
	for (int32_t i = 0; i + 4 <= Len; i += 4) {
		float32x4_t v = vld1q_f32(op->data + i);
		v = vmaxnmq_f32(v, zero);
		vst1q_f32(op->data + i, v);
	}
	for (int32_t i = r4(Len); i < Len; ++i) {
		float v = op->data[i];
		v = fmaxf(v, 0.0f);
		op->data[i] = v;
	}
}

void tensor_conv2d(Tensor * dst, const Tensor * src, const Tensor * kernel, const int32_t padding[MAX_DIM], const int32_t stride[MAX_DIM]) {
	tensor_conv2d_check(dst, src, kernel, padding, stride);
	const int32_t N = src->shape[0];
	const int32_t C_IN = src->shape[1];
	const int32_t H_IN = src->shape[2];
	const int32_t W_IN = src->shape[3];
	const int32_t C_OUT = dst->shape[1];
	const int32_t H_KER = kernel->shape[2];
	const int32_t W_KER = kernel->shape[3];
	const int32_t H_OUT = dst->shape[2];
	const int32_t W_OUT = dst->shape[3];
	const float PAD_VAL = 0;
	for (int32_t b = 0; b < N; ++b) {
		for (int32_t oc = 0; oc < C_OUT; ++oc) {
			for (int32_t h = 0; h < H_OUT; ++h) {
				for (int32_t w = 0; w < W_OUT; ++w) {
					float s = 0;
					const int32_t IND_DST[MAX_DIM] = {b, oc, h, w};
					// cross-correlation
					for (int32_t ic = 0; ic < C_IN; ++ic) {
						for (int32_t hKer = 0; hKer < H_KER; ++hKer) {
							for (int32_t wKer = 0; wKer < W_KER; ++wKer) {
								const int32_t IND_SRC[MAX_DIM] = {b, ic, h * stride[0] + hKer - padding[0], w * stride[1] + wKer - padding[1]};
								float vSrc = tensor_get_or_default_const(src, IND_SRC, PAD_VAL);
								const int32_t IND_KER[MAX_DIM] = {oc, ic, hKer, wKer};
								float vKer = *tensor_access_const(kernel, IND_KER);
								s += vSrc * vKer;
							}
						}
					}
					*tensor_access(dst, IND_DST) = s;
				}
			}
		}
	}
}

void tensor_maxpool2d(Tensor * dst, const Tensor * src, const int32_t kernelSize[MAX_DIM], const int32_t stride[MAX_DIM], const int32_t padding[MAX_DIM]) {
	tensor_maxpool2d_check(dst, src, kernelSize, stride, padding);
	const int32_t N = src->shape[0];
	const int32_t C = src->shape[1];
	const int32_t H_OUT = dst->shape[2];
	const int32_t W_OUT = dst->shape[3];
	const int32_t H_IN = src->shape[2];
	const int32_t W_IN = src->shape[3];
	const int32_t H_KER = kernelSize[0];
	const int32_t W_KER = kernelSize[1];
	const float PAD_VAL = FLT_MIN;
	for (int32_t i = 0; i < N; ++i) {
		for (int32_t j = 0; j < C; ++j) {
			for (int32_t h = 0; h < H_OUT; ++h) {
				for (int32_t w = 0; w < W_OUT; ++w) {
					float maxVal = FLT_MIN;
					for (int32_t m = 0; m < H_KER; ++m) {
						for (int32_t n = 0; n < W_KER; ++n) {
							const int32_t IND_SRC[MAX_DIM] = {i, j, stride[0] * h + m - padding[0], stride[1] * w + n - padding[0]};
							maxVal = fmaxf(maxVal, tensor_get_or_default_const(src, IND_SRC, PAD_VAL));
						}
					}
					const int32_t IND_DST[MAX_DIM] = {i, j, h, w};
					*tensor_access(dst, IND_DST) = maxVal;
				}
			}
		}
	}
}

static void dfs_for_softmax(Tensor * dst, int32_t curDim, int32_t axis, DimArray curIndex) {
	if (curDim == dst->dim) {
		if (axis == dst->dim - 1) {
			const int32_t L = dst->shape[axis];
			curIndex[axis] = 0;
			float * dstValPtrBase = tensor_access(dst, curIndex);
			float * dstValPtr;

			float32x4_t maxValVec = vdupq_n_f32(FLT_MIN);
			dstValPtr = dstValPtrBase;
			for (int32_t i = 0; i + 4 <= L; i += 4, dstValPtr += 4) {
				float32x4_t v = vld1q_f32(dstValPtr);
				maxValVec = vmaxnmq_f32(maxValVec, v);
			}
			float maxVal = reduce_max(maxValVec);
			for (int32_t i = r4(L); i < L; ++i, ++dstValPtr) {
				float v = *dstValPtr;
				maxVal = fmaxf(maxVal, v);
			}

			float denominator = 0.0f;
			dstValPtr = dstValPtrBase;
			for (int32_t i = 0; i < L; ++i, ++dstValPtr) {
				float v = *dstValPtr;
				v = expf(v - maxVal);
				*dstValPtr = v;
				denominator += v;
			}

			const float32x4_t Denom_Val_Vec = vdupq_n_f32(denominator);
			dstValPtr = dstValPtrBase;
			for (int32_t i = 0; i + 4 <= L; i += 4, dstValPtr += 4) {
				float32x4_t v = vld1q_f32(dstValPtr);
				v = vdivq_f32(v, Denom_Val_Vec);
				vst1q_f32(dstValPtr, v);
			}
			for (int32_t i = r4(L); i < L; ++i, ++dstValPtr) {
				float v = *dstValPtr;
				v = v / denominator;
				*dstValPtr = v;
			}
		} else {
			float maxVal = FLT_MIN;
			for (int32_t i = 0; i < dst->shape[axis]; ++i) {
				curIndex[axis] = i;
				float v = *tensor_access_const(dst, curIndex);
				maxVal = fmaxf(v, maxVal);
			}
			float denominator = 0.0f;
			for (int32_t i = 0; i < dst->shape[axis]; ++i) {
				curIndex[axis] = i;
				float v = *tensor_access_const(dst, curIndex) - maxVal;
				v = expf(v);
				*tensor_access(dst, curIndex) = v;
				denominator += v;
			}
			for (int32_t i = 0; i < dst->shape[axis]; ++i) {
				curIndex[axis] = i;
				*tensor_access(dst, curIndex) /= denominator;
			}
		}
	} else if (curDim == axis) {
		dfs_for_softmax(dst, curDim + 1, axis, curIndex);
	} else {
		for (int32_t i = 0; i < dst->shape[curDim]; ++i) {
			curIndex[curDim] = i;
			dfs_for_softmax(dst, curDim + 1, axis, curIndex);
		}
	}
}

void tensor_softmax(Tensor * dst, const Tensor * src, int32_t axis) {
	tensor_softmax_check(dst, src, axis);
	tensor_memcpy(dst, src);
	int32_t curIndex[MAX_DIM] = {};
	dfs_for_softmax(dst, 0, axis, curIndex);
}

static void dfs_for_fc(Tensor * dst, const Tensor * src, const Tensor * weight, int32_t curDim, DimArray curIndex) {
	if (curDim == src->dim - 1) {
		//  (..., srcDimLen) dot (dstDimLen, srcDimLen)
		int32_t dstDimLen = weight->shape[0];
		int32_t srcDimLen = weight->shape[1];
		curIndex[curDim] = 0;
		const float * srcValPtrBase = tensor_access_const(src, curIndex);
		const float * weightValPtr = weight->data;
		float * dstValPtr = tensor_access(dst, curIndex);
		for (int32_t i = 0; i < dstDimLen; ++i, ++dstValPtr) {
			float32x4_t vs = vdupq_n_f32(0.0f);
			const float * srcValPtr = srcValPtrBase;
			for (int32_t j = 0; j + 4 <= srcDimLen; j += 4, srcValPtr += 4, weightValPtr += 4) {
				float32x4_t srcVal = vld1q_f32(srcValPtr);
				float32x4_t weightVal = vld1q_f32(weightValPtr);
				float32x4_t mulRes = vmulxq_f32(srcVal, weightVal);
				vs = vaddq_f32(vs, mulRes);
			}
			float s = 0;
			for (int32_t j = r4(srcDimLen); j < srcDimLen; ++j, ++srcValPtr, ++weightValPtr) {
				float srcVal = *srcValPtr;
				float weightVal = *weightValPtr;
				float mulRes = srcVal * weightVal;
				s += mulRes;
			}
			s += reduce_sum(vs);
			*dstValPtr = s;
		}
	} else {
		const int32_t D = dst->shape[curDim];
		for (int32_t i = 0; i < D; ++i) {
			curIndex[curDim] = i;
			dfs_for_fc(dst, src, weight, curDim + 1, curIndex);
		}
	}
}

void tensor_fc(Tensor * dst, const Tensor * src, const Tensor * weight) {
	tensor_fc_check(dst, src, weight);
	int32_t curIndex[MAX_DIM];
	dfs_for_fc(dst, src, weight, 0, curIndex);
}

void tensor_reshape_inplace(Tensor * op, int32_t newDim, DimArray newShape) {
	tensor_reshape_inplace_check(op, newDim, newShape);
	int32_t oldLen = tensor_get_len(op);
	int32_t newLen = 1;
	int32_t minusOneIndex = -1;
	for (int32_t i = 0; i < newDim; ++i) {
		if (newShape[i] == -1) {
			assert(minusOneIndex == -1 && "tensor_reshape_inplace get more than one -1 in newShape.");
			minusOneIndex = i;
		} else {
			newLen *= newShape[i];
		}
	}
	if (minusOneIndex != -1) {
		assert(oldLen % newLen == 0 && "tensor_reshape_inplace shape deduce failed.");
		newShape[minusOneIndex] = oldLen / newLen;
		newLen = oldLen;
	}
	assert(newLen == oldLen && "tensor_reshape_inplace reshape requires same data length.");
	op->dim = newDim;
	memcpy(op->shape, newShape, sizeof(op->shape));
}