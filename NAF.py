from numpy.core.fromnumeric import shape
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

import pandas as pd
from tensorflow.python.ops import math_ops

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras
from tensorflow.python.keras import backend as K

from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class NonzeroNormWeights(tfk.constraints.Constraint):
    def __init__(self, axis=1):
        self.axis=axis

    def __call__(self, w):
        wnonzero = w * math_ops.cast(math_ops.greater_equal(w, 0.01), K.floatx())
        return wnonzero / (K.epsilon() + math_ops.reduce_sum(wnonzero, axis=self.axis, keepdims=True))

    def get_config(self):
        return {}

def invsigmoid(x):
    xclip = tf.clip_by_value(x, 1e-5, 0.99999)
    return tf.math.log(xclip/(1.0-xclip))
    #return tf.math.log(x/(1.0-x))

# construct NAF model
def NAF_denseflipout(inputdim, conddim, nafdim, depth=1):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    xcondin = xin[:, inputdim:]

    xfeatures = xin[:, :inputdim]
    outlist = []
    netout = None
    for iv in range(inputdim):
        xiv = tf.reshape(xfeatures[:, iv], [-1, 1])
        net = xiv
        for idepth in range(depth):
            condnet = xcondin
            condnet = tfpl.DenseFlipout(64, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfpl.DenseFlipout(64, activation=tf.nn.leaky_relu)(condnet)
            w1 = tfpl.DenseFlipout(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfpl.DenseFlipout(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfpl.DenseFlipout(64, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfpl.DenseFlipout(64, activation=tf.nn.leaky_relu)(condnet)
            w2 =  tfpl.DenseFlipout(nafdim, activation=tf.nn.softplus)(condnet)
            w2 = w2/ (1.0e-3 + tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))
        outlist.append(net)
        
        xcondin = tf.concat([xcondin, xiv], axis=1)
    outputlayer = tf.concat(outlist, axis=1)
    return tfk.Model(xin, outputlayer)



# construct NAF model
def NAF2(inputdim, conddim, nafdim, depth=1, permute=True):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    if conddim>0:
        xcondin = xin[:, inputdim:]
    else:
        xcondin = tf.zeros(shape=(tf.shape(xin)[0], 1), dtype=tf.float32)

    xfeatures = xin[:, :inputdim]
    netout = None
    nextfeature = xfeatures
    for idepth in range(depth):
        #permutation = tf.random.shuffle(tf.range(inputdim))
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{idepth}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
        else:
            permutation = tf.range(inputdim, dtype='int32',  name=f'permutation{idepth}')
        permuter = tfb.Permute(permutation=permutation, name=f'permute{idepth}')
        xfeatures_permuted = permuter.forward(nextfeature)
        outlist = []
        for iv in range(inputdim):
            xiv = tf.reshape(xfeatures_permuted[:, iv], [-1, 1])
            net = xiv
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            w1 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            w2 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            #w2 = w2/ (1.0e-7 + tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize
            w2 = w2/ (tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))
            outlist.append(net)
            xcondin = tf.concat([xcondin, xiv], axis=1)
        outputlayer_permuted = tf.concat(outlist, axis=1)
        outputlayer = permuter.inverse(outputlayer_permuted)
        nextfeature = outputlayer

    return tfk.Model(xin, outputlayer)

# construct NAF model
def NAF2gated(inputdim, conddim, nafdim, depth=1, permute=True):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    if conddim>0:
        xcondin = xin[:, inputdim:]
    else:
        xcondin = tf.zeros(shape=(tf.shape(xin)[0], 1), dtype=tf.float32)

    xfeatures = xin[:, :inputdim]
    netout = None
    nextfeature = xfeatures
    for idepth in range(depth):
        #permutation = tf.random.shuffle(tf.range(inputdim))
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{idepth}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
        else:
            permutation = tf.range(inputdim, dtype='int32',  name=f'permutation{idepth}')
        permuter = tfb.Permute(permutation=permutation, name=f'permute{idepth}')
        xfeatures_permuted = permuter.forward(nextfeature)
        outlist = []
        for iv in range(inputdim):
            xiv = tf.reshape(xfeatures_permuted[:, iv], [-1, 1])
            net = xiv
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            w1 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            w2 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            #w2 = w2/ (1.0e-7 + tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize
            w2 = w2/ (tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))
            outlist.append(net)
            xcondin = tf.concat([xcondin, xiv], axis=1)
        outputlayer_permuted = tf.concat(outlist, axis=1)
        outputlayer = permuter.inverse(outputlayer_permuted)
        nextfeature = outputlayer

    return tfk.Model(xin, xfeatures + outputlayer)

# find inverse of NAF2
# tf.Jacobian method is too slow
# try numerical method
@tf.function
def invNAF2(nafmodel, output, inputdim, cond):
    # trial
    npts = tf.shape(output)[0]
    #trialfeatinputv = tf.Variable(np.zeros(shape=tf.shape(output), dtype=np.float32), name='inv')
    #trialfeatinput = tf.constant(np.zeros(shape=tf.shape(output), dtype=np.float32))
    trialfeatinput = tf.zeros_like(output, dtype=np.float32)
    #trialfeatinput = tf.constant(np.zeros(shape=(npts, inputdim), dtype=np.float32))
    #trialfeatinput = tf.constant(output, dtype=np.float32)
    delta = 10.0*tf.ones(shape=(npts,), dtype=np.float32)
    epssq = 1.0e-6 
    maxtrial = 500 # maximum number of trials to find the inverse
    itrial = 0

    if cond is None:
        hascondition = False
        conddim = 0
    else:
        hascondition = True
        conddim = cond.shape[1]

    while tf.reduce_mean(delta)>epssq and itrial<maxtrial:
        itrial += 1
        with tf.GradientTape() as tape:
            tape.watch(trialfeatinput)
            if hascondition:
                trialinput = tf.concat([trialfeatinput, cond], axis=-1)
            else:
                trialinput = trialfeatinput
            trialoutput = nafmodel(trialfeatinput)
            delta = tf.reduce_sum(tf.math.squared_difference(trialoutput, output), axis=1)
            
        grad = tape.gradient(delta, trialfeatinput)
        trialfeatinput -= 0.01*grad
        #trialfeatinput = trialfeatinputv.value()
    
    #print(itrial)

    return trialfeatinput

# construct NAF model
def NAF3(inputdim, conddim, nafdim, depth=1, droprate=0.1, permute=True, gated=False):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    if conddim>0:
        xcondin = xin[:, inputdim:]
    else:
        xcondin = tf.zeros(shape=(tf.shape(xin)[0], 1), dtype=tf.float32)

    xfeatures = xin[:, :inputdim]
    netout = None
    nextfeature = xfeatures
    for idepth in range(depth):
        #permutation = tf.random.shuffle(tf.range(inputdim))
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{idepth}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
        else:
            permutation = tf.range(inputdim, dtype='int32',  name=f'permutation{idepth}')
        permuter = tfb.Permute(permutation=permutation, name=f'permute{idepth}')
        xfeatures_permuted = permuter.forward(nextfeature)
        outlist = []
        for iv in range(inputdim):
            xiv = tf.reshape(xfeatures_permuted[:, iv], [-1, 1])
            net = xiv
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dropout(droprate)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            w1 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dropout(droprate)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.leaky_relu)(condnet)
            w2 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            #w2 = w2/ (1.0e-7 + tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize
            w2 = w2/ (tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))
            outlist.append(net)
            xcondin = tf.concat([xcondin, xiv], axis=1)
        outputlayer_permuted = tf.concat(outlist, axis=1)
        outputlayer = permuter.inverse(outputlayer_permuted)
        nextfeature = outputlayer

    return tfk.Model(xin, outputlayer)


# Newton's method
# not good... do not use
def invNAF3(nafmodel, output, inputdim, cond):
    # Due to the conditional function construction,
    # start with finding the inverse with x0, then x1 and so on. 
    # x1 will depend on x0, but not the other way around.
    #ndim = tf.shape(output)[1] # when not compiled
    ndim = inputdim
    npts = tf.shape(output)[0]
    trialfeatinput = tf.zeros(shape=(npts, ndim), dtype=tf.float32)
    epssq = 1.0e-5 
    maxtrial = 500 # maximum number of trials to find the inverse
    itrial = 0
    delta = 10.0

    for idim in range(ndim):
        itrial = 0
        delta = 10.0
        maskmatrix = np.zeros(shape=(1, ndim), dtype=np.float32)
        maskmatrix[0, idim] = 1.0
        while delta>epssq and itrial<maxtrial:
            itrial += 1
            delta, trialfeatinput = loopeval(trialfeatinput, nafmodel, output, maskmatrix, idim, cond)
    
    return trialfeatinput

@tf.function
def loopeval(trialfeatinput, nafmodel, output, mask, idim, cond):
    with tf.GradientTape() as tape:
        tape.watch(trialfeatinput)
        if cond is not None:
            trialinput = tf.concat([trialfeatinput, cond], axis=-1)
        else:
            trialinput = trialfeatinput
        trialoutput = nafmodel(trialinput)
        delta = tf.reduce_mean(tf.math.squared_difference(trialoutput, output), axis=0)[idim]
    grad = tape.gradient(delta, trialfeatinput)*mask
    trialfeatinput -= 0.2* grad

    return delta, trialfeatinput

# bisection method
# this is the sure way to find the inverse
#@tf.function
def invNAF_bisect(nafmodel, output, inputdim, cond):
    ndim = inputdim
    npts = output.shape[0]
    epssq = 1.0e-4 
    left = -1.0
    right = 1.0

    maxtrial1 = 7
    maxtrial2 = 20

    if cond is None:
        hascondition = False
    else:
        hascondition = True

    trialfeatinput_left = np.zeros(shape=(npts, ndim), dtype=np.float32)
    trialfeatinput_right = np.zeros(shape=(npts, ndim), dtype=np.float32)

    # append conditions
    if cond is not None:
        trialfeatinput_left = np.concatenate([trialfeatinput_left, cond], axis=-1)
        trialfeatinput_right = np.concatenate([trialfeatinput_right, cond], axis=-1)

    for idim in range(ndim):
        trialfeatinput_left[:, idim] = left
        trialfeatinput_right[:, idim] = right

        trialoutput_left = nafmodel(trialfeatinput_left)[:,idim].numpy()
        trialoutput_right = nafmodel(trialfeatinput_right)[:,idim].numpy()

        outputi = output[:, idim]
        residual_left = trialoutput_left - outputi
        residual_right = trialoutput_right - outputi

        # first check whether the signs are opposite
        # increase boundary by factor 2 if not OK
        boundarycheckok = False
        itrial = 0
        while not boundarycheckok and itrial<maxtrial1:
            itrial += 1
            signs = residual_left * residual_right
            resizeboundary = (signs > 0.0)
            counttrue = np.count_nonzero(resizeboundary)
            if counttrue == 0:
                boundarycheckok = True
            else:
                trialfeatinput_left[resizeboundary,idim] = 2.0 * trialfeatinput_left[resizeboundary,idim]
                trialfeatinput_right[resizeboundary,idim] = 2.0 * trialfeatinput_right[resizeboundary,idim]
                trialoutput_left = nafmodel(trialfeatinput_left)[:,idim].numpy()
                trialoutput_right = nafmodel(trialfeatinput_right)[:,idim].numpy()
                residual_left = trialoutput_left - outputi
                residual_right = trialoutput_right - outputi
        
        if counttrue>0: # could not resolve then eliminate the data point
            selectrows = np.logical_not(resizeboundary)
            output = output[selectrows]
            trialfeatinput_left = trialfeatinput_left[selectrows]
            trialfeatinput_right = trialfeatinput_right[selectrows]
            outputi = outputi[selectrows]
            residual_left = residual_left[selectrows]
            residual_right = residual_right[selectrows]
        # apply bisection method now
        converged = False
        itrial = 0
        while not converged and itrial<maxtrial2:
            itrial += 1
            midpoint = (trialfeatinput_left + trialfeatinput_right)/2.0
            trialoutput_midpoint = nafmodel(midpoint)[:,idim].numpy()
            residual_midpoint = trialoutput_midpoint - outputi
            
            if np.count_nonzero(np.square(residual_midpoint) < epssq) == npts:
                converged = True

            leftboundaryOK = (residual_left * residual_midpoint <= 0.0)
            rightboundaryOK = (residual_right * residual_midpoint <= 0.0)
            
            trialfeatinput_right[leftboundaryOK, :] = midpoint[leftboundaryOK,:]
            residual_right[leftboundaryOK] = residual_midpoint[leftboundaryOK]
            trialfeatinput_left[rightboundaryOK, :] = midpoint[rightboundaryOK, :]
            residual_left[rightboundaryOK] = residual_midpoint[rightboundaryOK]
        trialfeatinput_left = midpoint
        trialfeatinput_right = midpoint.copy()
    #if midpoint.shape[0] < npts:
    #    print(f'{npts - midpoint.shape[0]} out of {npts} failed')
    return midpoint[:, :inputdim]
   

def invNAF_bisect2(nafmodel, output, inputdim, cond):
    ndim = inputdim
    npts = output.shape[0]
    epssq = 1.0e-4 
    left = -2.0
    right = 2.0

    maxtrial1 = 7
    maxtrial2 = 20

    if cond is None:
        hascondition = False
    else:
        hascondition = True

    trialfeatinput_left = left*np.ones(shape=(npts, ndim), dtype=np.float32)
    trialfeatinput_right = right*np.ones(shape=(npts, ndim), dtype=np.float32)    # append conditions

    if cond is not None:
        trialfeatinput_left = np.concatenate([trialfeatinput_left, cond], axis=-1)
        trialfeatinput_right = np.concatenate([trialfeatinput_right, cond], axis=-1)

    for idim in range(ndim):
        trialfeatinput_left[:, idim] = left
        trialfeatinput_right[:, idim] = right

        trialoutput_left = nafmodel(trialfeatinput_left)[:,idim].numpy()
        trialoutput_right = nafmodel(trialfeatinput_right)[:,idim].numpy()

        outputi = output[:, idim]
        residual_left = trialoutput_left - outputi
        residual_right = trialoutput_right - outputi

        # first check whether the signs are opposite
        # increase boundary by factor 2 if not OK
        boundarycheckok = False
        itrial = 0
        while not boundarycheckok and itrial<maxtrial1:
            itrial += 1
            signs = residual_left * residual_right
            resizeboundary = (signs > 0.0)
            counttrue = np.count_nonzero(resizeboundary)
            if counttrue == 0:
                boundarycheckok = True
            else:
                trialfeatinput_left[resizeboundary,idim] = 2.0 * trialfeatinput_left[resizeboundary,idim]
                trialfeatinput_right[resizeboundary,idim] = 2.0 * trialfeatinput_right[resizeboundary,idim]
                trialoutput_left = nafmodel(trialfeatinput_left)[:,idim].numpy()
                trialoutput_right = nafmodel(trialfeatinput_right)[:,idim].numpy()
                residual_left = trialoutput_left - outputi
                residual_right = trialoutput_right - outputi
        
        if counttrue>0: # could not resolve then eliminate the data point
            selectrows = np.logical_not(resizeboundary)
            output = output[selectrows]
            trialfeatinput_left = trialfeatinput_left[selectrows]
            trialfeatinput_right = trialfeatinput_right[selectrows]
            outputi = outputi[selectrows]
            residual_left = residual_left[selectrows]
            residual_right = residual_right[selectrows]

    converged = False    
    itrial = 0 
    trialoutput_left = nafmodel(trialfeatinput_left).numpy()
    trialoutput_right = nafmodel(trialfeatinput_right).numpy()
    residual_left = trialoutput_left - output
    residual_right = trialoutput_right - output
        
    while not converged and itrial<maxtrial2:
        itrial += 1
        midpoint = (trialfeatinput_left + trialfeatinput_right)/2.0
        trialoutput_midpoint = nafmodel(midpoint).numpy()
        residual_midpoint = trialoutput_midpoint - output
        
        if np.count_nonzero(np.sum(np.square(residual_midpoint), axis=1) < epssq) == npts:
            converged = True

        leftboundaryOK = (residual_left * residual_midpoint <= 0.0)
        rightboundaryOK = (residual_right * residual_midpoint <= 0.0)
        
        trialfeatinput_right[:,:ndim][leftboundaryOK] = midpoint[:,:ndim][leftboundaryOK]
        residual_right[leftboundaryOK] = residual_midpoint[leftboundaryOK]
        trialfeatinput_left[:,:ndim][rightboundaryOK] = midpoint[:,:ndim][rightboundaryOK]
        residual_left[rightboundaryOK] = residual_midpoint[rightboundaryOK]
    trialfeatinput_left = midpoint
    trialfeatinput_right = midpoint.copy()
    #if midpoint.shape[0] < npts:
    #    print(f'{npts - midpoint.shape[0]} out of {npts} failed')
    return midpoint[:, :inputdim]


    pass


# following not so good either, do not use
class InvNAF(object):
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    delta = 10.0
    epssq = 1.0e-6 
    maxtrial = 1000 # maximum number of trials to find the inverse

    def __init__(self):
        pass

    #@tf.function
    def invert(nafmodel, output, cond):
        # trial
        trialfeatinputv = tf.Variable(np.zeros(shape=tf.shape(output), dtype=np.float32), name='inv') # for non compiled
        trialfeatinput = tf.constant(np.zeros(shape=tf.shape(output), dtype=np.float32))
        #trialfeatinput = tf.zeros(shape=tf.shape(output), dtype=tf.float32, name='trialin')
        #delta = 10.0* tf.ones(shape=(npts, ndim), dtype=tf.float32, name='delta')
        delta = 10.0
        itrial = 0

        if cond is None:
            hascondition = False
            conddim = 0
        else:
            hascondition = True
            conddim = cond.shape[1]

        while tf.reduce_mean(delta)>InvNAF.epssq and itrial<InvNAF.maxtrial:
            itrial += 1
            with tf.GradientTape() as tape:
                tape.watch(trialfeatinput)
                if hascondition:
                    trialinput = tf.concat([trialfeatinput, cond], axis=-1)
                else:
                    trialinput = trialfeatinput
                trialoutput = nafmodel(trialinput)
                delta = tf.reduce_sum(tf.math.squared_difference(trialoutput, output), axis=1)
                #print(trialoutput)
                #print(delta)
                
            grad = tape.gradient(delta, [trialfeatinput])
            #InvNAF.optimizer.apply_gradients(zip(grad, [trialfeatinput]))
            InvNAF.optimizer.apply_gradients(zip(grad, [trialfeatinputv]))
            trialfeatinput = trialfeatinputv.value()
        
        #print(itrial)
        return trialfeatinput

# construct NAF model with Normal distribution as the prior
def NAF_DSF(inputdim, conddim, nafdim, depth=1, permute=False):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    if conddim>0:
        xcondin = xin[:, inputdim:]
    else:
        xcondin = tf.zeros(shape=(tf.shape(xin)[0], 1), dtype=tf.float32)


    xfeatures = xin[:, :inputdim]
    netout = None
    nextfeature = xfeatures
    for idepth in range(depth):
        #permutation = tf.random.shuffle(tf.range(inputdim))
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{idepth}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
        else:
            permutation = tf.range(inputdim, dtype='int32',  name=f'permutation{idepth}')
        permuter = tfb.Permute(permutation=permutation, name=f'permute{idepth}')
        xfeatures_permuted = permuter.forward(nextfeature)
        outlist = []
        for iv in range(inputdim):
            xiv = tf.reshape(xfeatures_permuted[:, iv], [-1, 1])
            net = xiv
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            w1 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            w2 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            w2 = w2/ (1.0e-6 + tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net2 = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))

            # second
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            w3 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b3 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net3 = tf.nn.sigmoid(w3 * net2 + b3)
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            w4 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            w4 = w4/ (1.0e-6 + tf.reduce_sum(w4, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net3 * w4, axis=1, keepdims=True))

            outlist.append(net)
            xcondin = tf.concat([xcondin, xiv], axis=1)
        outputlayer_permuted = tf.concat(outlist, axis=1)
        outputlayer = permuter.inverse(outputlayer_permuted)
        nextfeature = outputlayer

    return tfk.Model(xin, outputlayer)

def test_invNAF():
    nafmodel = NAF2(2,0,14)
    input = np.array([[0.1, 0.5]], np.float32)
    output = nafmodel(input)
    print(output)

    inputestimate = invNAF3(nafmodel, output, 2, None)
    print(inputestimate)
    pass


if __name__ == "__main__":
    test_invNAF()
    #tfk.utils.plot_model(nafmodel, to_file='NAF.png')
    pass