# named-tensors-asserts


## General idea

We write: 
```python
T &nt // "batch=1024, channels=3, height=32, width=32" 
```

to assert that `T`'s shape is `(1024,3,32,32)` and to update the global
named dimensions batch, channels,height, width to these values.

We can access these with nt.batch, nt.width etc
In future declarations we can write expressions such as: 
```python
Q &nt // "batch, channels*(height+1), width"
```

To say that a model maps tensors with dimensions `['batch','width','height','channels']` to tensors with dimensions `['batch','output']`

We we write: 

```python
model &nt // "batch, width, height, channels -> batch, output"
```

## How to use

At the moment, the way to use is just to run the file `named_asserts.py`. It will define the singleton `nt` which is the main object we use.
See [this colab notebook](https://githubtocolab.com/boazbk/named-tensors-asserts/blob/main/cifar10_example.ipynb) for an example