# PyTorch with Flask Demo

A simple, bare-bones example on how to serve an ML model through a web application, using
[PyTorch](https://pytorch.org/) and [Flask](https://flask.palletsprojects.com/en/2.2.x/).
Based off the official [classifier tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

Images for testing can be found in the `images/` directory.

**Installing dependencies**

```
$ pip install -r requirements.txt
```

**Training the model**

```
$ python ./model.py train
```

**Testing the model**

```
$ python ./model.py test
```

**Running the server**

```
$ python ./app.py
```
