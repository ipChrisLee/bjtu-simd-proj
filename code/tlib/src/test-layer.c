#include "test-layer.h"

#include <stdlib.h>

#include "tensor.h"
#include "test-helper.h"


#define get_and_check_not_null(field)                       \
	__auto_type field = tInfo->field;                       \
	if (field == NULL) {                                    \
		result->testResult = Missing_Parameter;             \
		result->info = "Missing field " #field " in tInfo"; \
		return;                                             \
	}

void test_relu(TestResultInfo * result, const TestInfo * tInfo) {
	assert(result && tInfo && "test_relu: missing result or tInfo pointer.");
	get_and_check_not_null(src);
	get_and_check_not_null(goldenDst);
	Tensor * dst = tensor_new(goldenDst->dim, goldenDst->shape);
	tensor_relu(dst, src);
	DimArray diffOn = {};
	TensorCompareResult r = tensor_compare_passed(dst, goldenDst, tInfo->rtol, tInfo->atol, &diffOn);
	if (r != Same_In_Tol) {
		result->testResult = Test_Failed;
		result->info = tensor_compare_result_str(r);
	} else {
		result->testResult = Test_Passed;
	}
}

void test_softmax(TestResultInfo * result, const TestInfo * tInfo) {
	assert(result && tInfo && "test_softmax: missing result or tInfo pointer.");
	get_and_check_not_null(src);
	get_and_check_not_null(goldenDst);
	get_and_check_not_null(axis);
	Tensor * dst = tensor_new(goldenDst->dim, goldenDst->shape);
	tensor_softmax(dst, src, *axis);
	DimArray diffOn = {};
	TensorCompareResult r = tensor_compare_passed(dst, goldenDst, tInfo->rtol, tInfo->atol, &diffOn);
	if (r != Same_In_Tol) {
		result->testResult = Test_Failed;
		result->info = tensor_compare_result_str(r);
	} else {
		result->testResult = Test_Passed;
	}
}

void test_conv2d(TestResultInfo * result, const TestInfo * tInfo) {
	assert(result && tInfo && "test_softmax: missing result or tInfo pointer.");
	get_and_check_not_null(src);
	get_and_check_not_null(goldenDst);
	get_and_check_not_null(kernel);
	get_and_check_not_null(padding);
	get_and_check_not_null(stride);
	Tensor * dst = tensor_new(goldenDst->dim, goldenDst->shape);
	ConstDimArray Zero_Dim_Array = {};
	DimArray onesDimArray = {};
	for (int32_t i=0;i<MAX_DIM; ++i) {
		onesDimArray[i] = 1;
	}
	tensor_conv2d(dst, src, kernel, *padding, *stride);
	DimArray diffOn = {};
	TensorCompareResult r = tensor_compare_passed(dst, goldenDst, tInfo->rtol, tInfo->atol, &diffOn);
	if (r != Same_In_Tol) {
		result->testResult = Test_Failed;
		result->info = tensor_compare_result_str(r);
	} else {
		result->testResult = Test_Passed;
	}
}