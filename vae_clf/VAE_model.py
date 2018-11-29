#!/usr/bin/env python
# encoding: utf-8

"""
@author: izyq
@file: VAE_model.py
@time: 2018/11/27 14:25
"""
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import os

class GCNN(Layer):
    def __init__(self,output_dim=None,residual=False,**kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim=output_dim
        self.residual=residual

    def build(self, input_shape):
        if self.output_dim==None:
            self.output_dim=input_shape[-1]
        self.kernel=self.add_weight(name='gcnn_kernel',
                                    shape=(3,input_shape[-1],self.output_dim*2),
                                    initializer='glorot_uniform',
                                    trainable=True)

    def call(self, inputs):
        _=K.conv1d(inputs,self.kernel,padding='same')
        _=_[:,:,:self.output_dim]*K.sigmoid(_[:,:,self.output_dim:])
        if self.residual:
            return inputs+_
        else:
            return _

class SetConfig:
    def __init__(self,config):
        self.sequence_length=config['sequence_length']
        self.latent_dim=config['latent_dim']


def sampling(args):
    z_mean,z_log_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0],getconfig.latent_dim),
                            mean=0.0,
                            stddev=1.0)
    return z_mean+K.exp(z_log_var/2)*epsilon


# load data
modelpath='/home/fhzhu/youqianzhu/newstitle/model/vae_model/vae_1/vae_model/'
opath='/home/fhzhu/youqianzhu/newstitle/data/nlpcc_data/processed_word/new_npy/'
tag_path='/home/fhzhu/youqianzhu/newstitle/data/nlpcc_data/processed_word/npy/'
# (bs,sl)
x_train=np.load(os.path.join(opath,'x_train.npy'))
y_train=np.load(os.path.join(tag_path,'train_tags.npy'))
# (bs,sl,1)->(bs,sl,1)
x_train_mask=Lambda(lambda x_train:K.cast(K.greater(K.expand_dims(x_train,-1),0),'float32'))(x_train)
x_train_converse_mask=Lambda(lambda x_train:K.cast(K.equal(K.expand_dims(x_train,-1),0),'float32'))(x_train)

x_dev=np.load(os.path.join(opath,'x_dev.npy'))
x_dev_mask=Lambda(lambda x_dev:K.cast(K.greater(K.expand_dims(x_dev,-1),0),'float32'))(x_dev)
x_dev_converse_mask=Lambda(lambda x_dev:K.cast(K.equal(K.expand_dims(x_dev,-1),0),'float32'))(x_dev)
y_dev=np.load(os.path.join(tag_path,'dev_tags.npy'))

config={}
config['sequence_length']=30
config['latent_dim']=256
getconfig=SetConfig(config)

# the model of vae
input_sentence=Input(shape=(getconfig.sequence_length,),dtype='int32')
embedding=np.load(os.path.join(opath,'embedding.npy'))
embedding_dim=embedding.shape[-1]
vocab_size=embedding.shape[0]

# (bs,sl,dim)
embedding_layer=Embedding(input_dim=vocab_size,
                          output_dim=embedding_dim,
                          input_length=getconfig.sequence_length,
                          weights=[embedding],
                          trainable=True)
# (bs,sl,out)
gcnn_1=GCNN(residual=True)
# (bs,sl,out)
gcnn_2=GCNN(residual=True)
# (bs,out)
h_pool=GlobalAveragePooling1D()
encoder_d1=Dense(getconfig.latent_dim,activation='relu')
encoder_d2=Dense(getconfig.latent_dim,activation='relu')
sample_z=Lambda(sampling)


input_vec=embedding_layer(input_sentence)
h=gcnn_1(input_vec)
h=gcnn_2(h)
h=h_pool(h)
# 均值 方差
# (bs,latent_dim)
z_mean=encoder_d1(h)
z_log_var=encoder_d2(h)
# (bs,latent_dim)
z=sample_z([z_mean,z_log_var])

# (bs,embedding*)
decoder_hidden=Dense(embedding_dim*getconfig.sequence_length)
decoder_cnn=GCNN(residual=True)
decoder_dense=Dense(vocab_size,activation='softmax')

h=decoder_hidden(z)
h=Reshape((K.shape(h)[0],getconfig.sequence_length,embedding_dim))(h)
h=decoder_cnn(h)
# (bs,sl,vocab)
output=decoder_dense(h)
output_mask=Lambda(lambda output,x_train_mask:output*x_train_mask)([output,x_train_mask])
# (bs,sl,vocab_size)
output_real=Lambda(lambda output_mask,x_train_converse_mask:output_mask+x_train_mask)([output_mask,x_train_converse_mask])


vae=Model(input_sentence,output_real)

# (bs,sl,vocab) (bs,sl)(bs,sl,vocab)=>(bs,sl)=>(bs,)
xent_loss=K.sum(K.sparse_categorical_crossentropy(output_real,input_sentence),1)
# (bs,)
kl_loss=-0.5*K.sum(1+z_log_var-K.square(z_mean)-K.exp(z_log_var),axis=1)
vae_loss=K.mean(xent_loss+kl_loss)

vae.add_loss(vae_loss)
checkpoint=ModelCheckpoint(os.path.join(modelpath,'vae_bestmodel'),verbose=1,save_best_only=True)
adam=Adam(lr=1e-3)
vae.compile(optimizer=adam)

# encoder
c_input=Input(shape=getconfig.sequence_length)
_=embedding_layer(c_input)
_=gcnn_1(_)
_=gcnn_2(_)
_output=GlobalAveragePooling1D()
_z_mean=encoder_d1(_output)
_z_log_var=encoder_d2(_output)
# (bs,latent_dim)
_z=sample_z([_z_mean,_z_log_var])
# (bs,18)
logits=Dense(18,activation='softmax')(_z)
cmodel=Model(c_input,logits)


def classifier(cmodel,modelpath):
    adam=Adam(lr=1e-3)
    checkpoint=ModelCheckpoint(os.path.join(modelpath,'cmodel_bestmodel'),verbose=1,save_best_only=True)
    cmodel.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['acc'])
    cmodel.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=[x_dev,y_dev],callbacks=[checkpoint])
    score_train=cmodel.evaluate(x_train,y_train)
    score_dev=cmodel.evaluate(x_dev,y_dev)
    print('score_train=%f'%score_train)
    print('score_dev=%f'%score_dev)

vae.fit(x_train,batch_size=64,epochs=100,callbacks=[checkpoint])
cmodelpath='/home/fhzhu/youqianzhu/newstitle/model/vae_model/vae_1/cmodel/model/'
classifier(cmodel,cmodelpath)





