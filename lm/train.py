import tensorflow as tf
import os
import numpy as np
import rnn
import time
import english_adapter
pathname="test.csv"
dataset,datasetvalid,len1,len2,vocab_size=english_adapter.getdataset(16,1000,pathname)
model=rnn.rnn_model(vocab_size+1, 16, 4)
optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
checkpoint_dir='./training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,"ckpt")
checkpoint=tf.train.Checkpoint(optimizer=optimizer,
                               model=model)

input= np.zeros((2,5))
target=np.zeros((2,5))
pre=model(input)
print(pre)
print(loss_object(target,pre))
def loss_function(real,pred):
    loss = loss_object(real, pred)
    return tf.reduce_mean(loss)
@tf.function
def train_step(inp,tar):
    with tf.GradientTape() as tape:
        prediction=model(inp)
        loss=loss_function(tar,prediction)

    variables=model.trainable_variables
    gradients=tape.gradient(loss,variables)
    optimizer.apply_gradients(zip(gradients,variables))
    return loss

@tf.function
def test_step(inp,tar):
    prediction=model(inp)
    loss=loss_function(tar,prediction)
    return loss

steps_per_epoch=len1
EPOCH=10
for epoch in range(EPOCH):
    start=time.time()
    total_loss=0
    total_valid_loss=0
    count=0
    count_valid=0
    for(batch,(inp,targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss=train_step(inp,targ)
        total_loss+=batch_loss
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
        count+=1
    for (batch, (inp, targ)) in enumerate(datasetvalid.take(steps_per_epoch)):
        batch_valid_loss = test_step(inp, targ)
        total_valid_loss += batch_valid_loss
        count_valid+=1
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / count))
    print('Epoch {} valid Loss {:.4f}'.format(epoch + 1,
                                        total_valid_loss / count_valid))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
