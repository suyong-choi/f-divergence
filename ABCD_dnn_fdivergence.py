import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

import matplotlib.pyplot as plt
import os
import pickle

from NAF import NAF2, NAF3, NAF2gated
from CBNAF import buildConditionalBNAF
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import fdivergence

class SawtoothSchedule(LearningRateSchedule):
    def __init__(self, start_learning_rate=0.0001, end_learning_rate=0.000001, cycle_steps=100, random_fluctuation = 0.0, name=None):
        super(SawtoothSchedule, self).__init__()
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.cycle_steps = cycle_steps
        self.random_fluctuation = random_fluctuation
        self.name = name
    pass

    def __call__(self, step):
        phase = step % self.cycle_steps
        lr = self.start_learning_rate + (self.end_learning_rate-self.start_learning_rate)* (phase/self.cycle_steps)
        if (self.random_fluctuation>0):
            lr *= np.random.normal(1.0, self.random_fluctuation)
        return lr

    def get_config(self):
        return {
            "start_learning_rate": self.start_learning_rate,
            "end_learning_rate": self.end_learning_rate,
            "cycle_steps": self.cycle_steps,
            "random_fluctuation": self.random_fluctuation,
            "name": self.name
        }



class ABCDdnn(object):
    def __init__(self, inputdim_categorical_list, inputdim, minibatch=128, netarch=[1024, 256, 64], nafdim=16, depth=2, LRrange=[0.0001, 0.0001, 1, 0.0], \
        conddim=0, droprate=0.1, beta1=0.5, beta2=0.9, lamb=1.0, retrain=False, seed=100, permute=False, knn=5, savedir='./abcdnn/', savefile='abcdnn.pkl', gated=False):
        self.inputdim_categorical_list = inputdim_categorical_list
        self.inputdim = inputdim
        self.inputdimcat = int(np.sum(inputdim_categorical_list))
        self.inputdimreal = inputdim - self.inputdimcat
        self.minibatch = minibatch
        self.netarch = netarch
        self.nafdim = nafdim
        self.depth = depth
        self.LRrange = LRrange
        self.conddim = conddim
        self.droprate = droprate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb = lamb
        self.retrain = retrain
        self.savedir = savedir
        self.savefile = savefile
        self.global_step = tf.Variable(0, name='global_step')
        self.monitor_record = []
        self.monitorevery = 50
        self.seed = seed
        self.permute = permute
        self.knn = knn
        self.gated = gated
        self.setup()

    def setup(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.createmodel()
        self.checkpoint = tf.train.Checkpoint(global_step=self.global_step, model = self.model, optimizer=self.optimizer)
        self.checkpointmgr = tf.train.CheckpointManager(self.checkpoint, directory=self.savedir, max_to_keep=5)
        if (not self.retrain) and os.path.exists(self.savedir):
            status = self.checkpoint.restore(self.checkpointmgr.latest_checkpoint)
            status.assert_existing_objects_matched()
            print('loaded model from checkpoint')
            if os.path.exists(os.path.join(self.savedir, self.savefile)):
                print("Reading monitor file")
                self.load_training_monitor()
            print("Resuming from step", self.global_step)
        elif not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        pass

    def createmodel(self):
        #inputlayer = layers.Input(shape=(self.inputdimreal+self.inputdimcat+self.conddim+self.conddim,))
        inputlayer = layers.Input(shape=(self.inputdimreal+self.inputdimcat+self.conddim,))
        net = inputlayer
        noutdim = self.inputdimreal + self.inputdimcat
        self.model = NAF2gated(self.inputdim, self.conddim, nafdim=self.nafdim, depth=self.depth, permute=self.permute)
        #self.model = NAF3(self.inputdim, self.conddim, nafdim=self.nafdim, depth=self.depth, permute=self.permute \
        #               , gated=self.gated, droprate=self.droprate)
        #self.model = buildConditionalBNAF(self.inputdim, self.conddim, nafdim=self.nafdim, depth=self.depth, \
        #                                   permute=self.permute, gated=self.gated, droprate=self.droprate)
        
        self.model.summary()
        #tf.keras.utils.plot_model(self.model, to_file=self.savedir+'ABCD_dnn.png')
        lr_fn = SawtoothSchedule(self.LRrange[0], self.LRrange[1], self.LRrange[2], self.LRrange[3])
        self.optimizer = tfk.optimizers.Adam(learning_rate = lr_fn,  beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-5, name='nafopt')
        pass

    def setrealdata(self, numpydata, eventweight=None):
        self.numpydata = numpydata
        self.ntotalevents = numpydata.shape[0]
        self.datacounter = 0
        self.randorder = np.random.permutation(self.numpydata.shape[0])
        if eventweight is not None:
            self.eventweight = eventweight
        else:
            self.eventweight = np.ones((self.ntotalevents, 1), np.float32)
        pass


    def savehyperparameters(self):
        """Write hyper parameters into file
        """
        params = [self.inputdim, self.conddim, self.LRrange, self.beta1, self.beta2, self.minibatch, self.nafdim, self.depth, self.knn]
        pickle.dump(params, open(os.path.join(self.savedir, 'hyperparams.pkl'), 'wb'))
        pass

    def monitor(self):
        self.monitor_record.append([self.checkpoint.global_step.numpy(), self.glossv.numpy()])

    def save_training_monitor(self):
        pickle.dump(self.monitor_record, open(os.path.join(self.savedir, self.savefile), 'wb'))
        pass

    def load_training_monitor(self):
        fullfile = os.path.join(self.savedir, self.savefile)
        if os.path.exists(fullfile):
            self.monitor_record = pickle.load(open(fullfile, 'rb'))
            self.epoch = self.monitor_record[-1][0] + 1
        pass

    def get_next_batch(self, size=None, cond=False):
        """Return minibatch from random ordered numpy data
        """
        if size is None:
            size = self.minibatch
        if self.datacounter + size >= self.ntotalevents:
            self.datacounter = 0
            self.randorder = np.random.permutation(self.numpydata.shape[0])

        batchbegin = self.datacounter
        if not cond:
            batchend = batchbegin + size
            self.datacounter += size
            nextbatch = self.numpydata[self.randorder[batchbegin:batchend], 0::]
        else:
            nextconditional = self.numpydata[self.randorder[batchbegin], self.inputdim:]
            # find all data 
            bigenough = False
            while(not bigenough):
                match = (self.numpydata[:, self.inputdim:] == nextconditional).all(axis=1)
                nsamples_incategory = np.count_nonzero(match)
                if nsamples_incategory>=self.minibatch:
                    bigenough = True
                batchbegin += 1
                if batchbegin >= self.ntotalevents:
                    batchbegin = 0
                
            matchingarr = self.numpydata[match, :]
            matchingwgt = self.eventweight[match, :]
            randorder = np.random.permutation(matchingarr.shape[0])
            nextbatch = matchingarr[randorder[0:self.minibatch], :]
            nextbatchwgt = matchingwgt[randorder[0:self.minibatch], :]

        return nextbatch

    @tf.function
    def train_step(self, sourcebatch, targetbatch):
        # update discriminator ncritics times
        if self.conddim>0:
            conditionals = targetbatch[:, -self.conddim:]

        # update generator
        gen_loss = tf.constant(0, dtype=tf.float32)
        with tf.GradientTape() as gtape:
            generated = self.model(tf.concat([sourcebatch[:, :self.inputdim], conditionals], axis=-1), training=True)
            generated = tf.concat([generated, conditionals], axis=-1)
            for knn in self.knn:
                #gen_loss += fdivergence.D_KL(targetbatch[:, :self.inputdim], generated[:, :self.inputdim], knn, knn)
                #gen_loss += fdivergence.D_KLsym(targetbatch[:, :self.inputdim], generated[:, :self.inputdim], knn, knn)
                #gen_loss += fdivergence.D_JS_fromDKL(targetbatch[:, :self.inputdim], generated[:, :self.inputdim], knn, knn)
                # alpha 0.5 doesn't work
                gen_loss += fdivergence.alphadiv(targetbatch[:, :self.inputdim], generated[:, :self.inputdim], knn, knn, 2.0)
        if tf.math.is_finite(gen_loss):
            gen_grad = gtape.gradient(gen_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gen_grad, self.model.trainable_variables))
        else:
            tf.debugging.check_numerics(gen_loss, message = 'NaN encountered')
        meangloss = tf.reduce_mean(gen_loss)

        return meangloss

    def train(self, steps=1000):
        for istep in range(steps):
            source = self.get_next_batch()
            target = self.get_next_batch(cond=True)
            self.glossv = self.train_step(source, target)
            # generator update
            if istep % self.monitorevery == 0:
                print(f'{self.checkpoint.global_step.numpy()} {self.glossv.numpy():.3e} ')
                self.monitor()
                self.checkpointmgr.save()
            self.checkpoint.global_step.assign_add(1) # increment counter
        self.checkpointmgr.save()
        self.save_training_monitor()

    def display_training(self):
        # Following section is for creating movie files from trainings

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        monarray = np.array(self.monitor_record)
        x = monarray[0::, 0]
        ax.plot(x, monarray[0::, 1], color='r', label='gloss')
        #ax.set_yscale('log')
        ax.legend()

        plt.grid()
        plt.draw()

        fig.savefig(os.path.join(self.savedir, 'trainingperf.pdf'))
        pass

    def generate_sample(self, condition, repeatfirst=False):
        xin = self.get_next_batch()

        if not repeatfirst:
            xin_nocond = xin[:, :self.inputdim]
        else:
            xin_nocond = np.repeat(xin[0, :self.inputdim], self.minibatch, axis=0).reshape((self.minibatch, self.inputdim))

        yin = np.repeat(condition, self.minibatch, axis=0) # copy the same
        netin = np.hstack((xin_nocond, yin))
        youthat = self.model(netin) # for distribution
        #youthat = self.model.predict(netin)
        return youthat


    pass