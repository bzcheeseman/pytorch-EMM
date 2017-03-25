## pytorch-EMM ##

EMM stands for External Memory Module.

The initial implementation, EMM_NTM is very similar to a normal NTM except
it's more self-contained. The other differences include an arbitrary number
of read heads - which only requires minimal changes to the controller
network to actually make use of. I have also implemented multiple memory banks
which was inspired by [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228)
by Kaiser and Sutskever.

I'm currently working also on EMM_GPU which is an external memory module
even more inspired by the Neural GPU. I'm not sure how to do this, but I'm 
trying to create a 3D memory that's addressed by convolution filters
to read in and out.

My original NTM project is [here](https://github.com/bzcheeseman/pytorch-NTM)

Contributing: Fork the repo and play with it! I'm always happy to hear suggestions!