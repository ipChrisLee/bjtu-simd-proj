#pragma once

#include "test-helper.h"

void test_relu(TestResultInfo * result, const TestInfo * tInfo);
void test_softmax(TestResultInfo * result, const TestInfo * tInfo);
void test_conv2d(TestResultInfo * result, const TestInfo * tInfo);
void test_fc(TestResultInfo * result, const TestInfo * tInfo);
void test_maxpool2d(TestResultInfo * result, const TestInfo * tInfo);
