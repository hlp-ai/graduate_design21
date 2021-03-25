import tensorflow as tf
import numpy as np
input=np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4]])
print(len(input))
print(input)
input=tf.data.Dataset.from_tensor_slices(input)
print(input)
input=input.shuffle(100).batch(2,drop_remainder=True)
print(input)
for(batch,inp) in enumerate(input.take(3)):
    print(batch)
    print(inp)