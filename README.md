# convy
Convy is a deep learning compiler proof of concept. So far this compiler lowers a convolution layer defined in Pytorch to C code.

Hopefully, I can end up compiling a VGG convolutional neural net :-)

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
