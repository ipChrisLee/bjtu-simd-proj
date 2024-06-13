#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test-helper.h"
#include "test-layer.h"

#define GREEN "\033[0;32m"
#define RED "\033[0;31m"
#define NC "\033[0m"

/**
 * Usage: run-test "relu" "tInfoPath"
 * 		test `relu` layer, with info path in `tInfoPath`
 */
int main(int argc, char const ** argv) {
	const char * layerName = argv[1];
	const char * tInfoPath = argv[2];

	Msg_t msg;
	TestInfo * tInfo = test_info_read_from(tInfoPath, &msg);
	if (tInfo == NULL) {
		printf(RED "Test failed since reading tInfo failed with msg {%s}.\n" NC, msg);
		exit(EXIT_FAILURE);
	}

	TestResultInfo * result = test_result_info_new();
#define concat(x, y) x##y
#define layer_case(layer) \
	else if (strcmp(layerName, #layer) == 0) { concat(test_, layer)(result, tInfo); }

	// clang-format off
	if(false){

	}
	layer_case(relu)
	layer_case(softmax)
	layer_case(conv2d)
	layer_case(fc)
	else {
		printf(RED "Unsupported layer %s\n" NC, layerName);
		exit(EXIT_FAILURE);
	}
	// clang-format on

	if (result->testResult != Test_Passed) {
		printf(RED "Test failed when comparing, with info {%s}.\n" NC, result->info);
		exit(EXIT_FAILURE);
	} else {
		printf(GREEN "Test passed.\n" NC);
	}

	return 0;
}