#include "test-layer.h"

#include <stdlib.h>

#include "tensor.h"
#include "test-helper.h"


#define get_and_check_not_null(field)                       \
	__auto_type field = tInfo->field;               \
	if (field == NULL) {                                    \
		result->testResult = Missing_Parameter;             \
		result->info = "Missing field " #field " in tInfo"; \
		return;                                             \
	}

void test_relu(TestResultInfo * result, const TestInfo * tInfo) {
	assert(result && tInfo && "test_reul: missing result or tInfo pointer.");
	get_and_check_not_null(src);
	get_and_check_not_null(goldenDst);
	Tensor * dst = tensor_new(goldenDst->dim, goldenDst->shape);
	tensor_relu(dst, src);
	TensorCompareResult r = tensor_compare_passed(dst, goldenDst, tInfo->rtol, tInfo->atol, NULL);
	if (r != Same_In_Tol) {
		result->testResult = Test_Failed;
		result->info = tensor_compare_result_str(r);
	} else {
		result->testResult = Test_Passed;
	}
}