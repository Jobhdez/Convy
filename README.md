# llm compiler
The start of an llm compiler

## Running the examples
So, far I have generated C code for a 2d convolutional layer.

On Linux or macOS you type the following to run the example. Running the example will generste a C code file contsing the generated code for the CNN in the `conv2d.py` file in the `examples` directory.

```
$ pip install -r requirements.txt
$ pip install -e .
$ cd src/frontend
$ python conv2d.py
$ gcc ../backend/conv.c
$ cd ../backend
$ ./a.out
```

## Goals

My main goal is not to build the next tvm. That would be silly.

But I do want to make a deep learning compiler capable of compiling a VGG block to C code and Cuda call and do some optimizations. Thats what I want to build.
