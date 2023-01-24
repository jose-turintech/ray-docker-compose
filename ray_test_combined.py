from collections import Counter
import socket
import time
import ray

ray.init(address='ray://localhost:10001')

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))


# Tensorflow
@ray.remote(resources={"tensorflow": 1})
def tensorflow_call():
    time.sleep(0.001)
    # Return IP address.
    import tensorflow as tf
    version = tf.__version__
    return socket.gethostbyname(socket.gethostname()), version


object_ids = [tensorflow_call.remote() for _ in range(10)]
results = ray.get(object_ids)

print('Tensorflow tasks executed')
for result, num_tasks in Counter(results).items():
    print('    {} tasks on {} with tensorflow version {}'.format(num_tasks, result[0], result[1]))


# Torch
@ray.remote(resources={"pytorch": 1})
def pytorch_call():
    time.sleep(0.001)
    # Return IP address.
    import torch
    version = torch.__version__
    return socket.gethostbyname(socket.gethostname()), version


object_ids = [pytorch_call.remote() for _ in range(10)]
results = ray.get(object_ids)

print('Pytorch tasks executed')
for result, num_tasks in Counter(results).items():
    print('    {} tasks on {} with pytorch version {}'.format(num_tasks, result[0], result[1]))


# Xgboost
@ray.remote(resources={"xgboost": 1})
def xgboost_call():
    time.sleep(0.001)
    # Return IP address.
    import xgboost as xg
    version = xg.__version__
    return socket.gethostbyname(socket.gethostname()), version


object_ids = [xgboost_call.remote() for _ in range(10)]
results = ray.get(object_ids)

print('Tensorflow tasks executed')
for result, num_tasks in Counter(results).items():
    print('    {} tasks on {} with tensorflow version {}'.format(num_tasks, result[0], result[1]))
