## pytorch-EMM ##

EMM stands for External Memory Module.

I've been doing a lot of reading up on Neural GPUs and synthetic gradients
and I decided to first make a small change to my NTM. The first thing
I did was to modularize the head and memory so that you can (almost)
plug in a network to this module and go. Then, taking some inspiration from
[Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) by Kaiser and Sutskever
I added extra memory banks that are connected with batch normalization (thinking about other methods too). The 
eventual idea is to combine this with something else like a Neural GPU
and/or synthetic gradients (this model lends itself well to unlocking
updates for example).

My original NTM project is [here](https://github.com/bzcheeseman/pytorch-NTM)