# convolutional layer to C code
a little compiler that takes a neural net defined with Pytorch and lowers to C code that runs with GCC.

## Running the examples

On Linux or macOS you type the following to run the example. 

```
$ pip install -r requirements.txt
$ pip install -e .
$ cd src/frontend
$ python conv2d.py
$ gcc ../backend/conv.c
$ cd ../backend
$ ./a.out
```
