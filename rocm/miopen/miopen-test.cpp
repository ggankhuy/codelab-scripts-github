/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
//#include "test.hpp"
//#include <array>
#include <iostream>
#include <iterator>
//#include <limits>
//#include <memory>
//#include <miopen/convolution.hpp>
#include <miopen/miopen.h>
//#include <miopen/tensor.hpp>
//#include <miopen/tensor_ops.hpp>
//#include <utility>

//#include "driver.hpp"
//#include "get_handle.hpp"
//#include "tensor_holder.hpp"
//#include "verify.hpp"
//#include "tensor.hpp"

#define MIO_OPS_DEBUG 0

struct s1    \
{            \
}; typedef struct s1* s1_t;


int main(int argc, const char* argv[]) { 
    
    //tensor<int> a;
    miopenTensorDescriptor_t tensor{};

    //s1_t s1i{};
    //printf("size of s1: %d.\n", sizeof(s1i));

    printf("size of miopenTensorDescriptor_t: %d 0x0%x.\n", sizeof(*tensor), tensor);
    miopenCreateTensorDescriptor(&tensor);
    printf("size of miopenTensorDescriptor_t: %d 0x0%x.\n", sizeof(*tensor), tensor);
    miopenSet4dTensorDescriptor(tensor, miopenFloat, 100, 32, 8, 8);
    printf("size of miopenTensorDescriptor_t: %d 0x0%x.\n", sizeof(*tensor), tensor);
    return 0;
}
