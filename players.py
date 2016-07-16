
import numpy as np
import scipy
import players_net
import pylab as py
import theano
import theano.tensor as T
# import time
import scipy.ndimage as nd
import lasagne
import os

def try_moving_around():

    input_var = T.tensor4('inputs')
    inp_out=T.tensor4('inp_out')
    network = players_net.build_cnn(input_var)
    out_net=players_net.build_deconv_network(network,inp_out)
    out_im=lasagne.layers.get_output(out_net,deterministic=True)
    if (os.path.isfile('net_old.npy')):
        spars=np.load('net_old.npy')
        lasagne.layers.set_all_param_values(network,spars)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction=T.transpose(test_prediction,(0,2,3,1))
    dims=test_prediction.shape
    test_prediction=T.reshape(test_prediction,(dims[0]*dims[1]*dims[2],dims[3]))
    #test_prediction=T.reshape(test_prediction,(-1,3))
    test_prediction=T.nnet.softmax(test_prediction)
    test_prediction=T.reshape(test_prediction,dims)
    test_fn = theano.function([input_var], [test_prediction])
    out_fn=theano.function([inp_out],[out_im])
    dir='/Users/amit/Desktop/Dropbox/ACMilxLAZIO/'
    fls=os.listdir(dir)
    ii=np.int32(range(len(fls)))
    np.random.shuffle(ii)
    aaa=[]
    for i in ii[0:10]:
        fn=dir+fls[i]
        print(fn)
        if ('jpg' in fn):
            aaa.append(nd.imread(dir+fls[i]))


    aaa=np.array(aaa)
    aa=standardize_data(aaa)
    imb=np.float32(aa)
    print(imb.shape)
    imb=np.transpose(imb,(0,3,1,2))
    tt=test_fn(imb)
    ss=np.transpose(tt[0],(0,3,1,2))
    oo=out_fn(ss)
    ooo=np.squeeze(np.transpose(oo[0],(0,2,3,1)))

    for i in range(imb.shape[0]):

            py.figure(1)
            for j in range(4):
                py.subplot(1,5,j+1)
                py.imshow(ooo[i,:,:,j])
                py.axis('off')
                py.axis('equal')
            py.subplot(1,5,5)
            py.imshow(aaa[i,:,:,:])
            py.axis('off')
            py.axis('equal')
            py.show()



def load_from_jpg(dir,dosize=True):
    newdims=(61,31,3)
    lower=50
    higher=75
    dirs=os.listdir(dir)
    IM=[]
    for sdir in dirs[0:10]:
        if ('DS_Store' not in sdir):
            ims=os.listdir(dir+sdir)
            print(sdir)
            for im in ims:
                #print(im)
                if ('jpg' in im or 'JPG' in im):
                    #print(dir+sdir+'/'+im)
                    fn=dir+sdir+'/'+im
                    if (dosize):
                        lens=len(fn)
                        siz=fn[lens-6:lens-4]
                        size=int(siz)
                    else:
                        size=40.
                    if (size>=lower and size<=higher or not dosize):
                        #print(size)
                        aa=nd.imread(fn)
                        aa=scipy.misc.imresize(aa,np.double(40./size))
                        bb=np.zeros(newdims)
                        if (newdims[0]>=aa.shape[0]):
                            marg0=(newdims[0]-aa.shape[0])/2
                            marg1=(newdims[1]-aa.shape[1])/2
                            bb[marg0:(marg0+aa.shape[0]),marg1:(marg1+aa.shape[1]),:]=aa
                        else:
                            marg0=(aa.shape[0]-newdims[0])/2
                            marg1=(aa.shape[1]-newdims[1])/2
                            bb=aa[marg0:(marg0+newdims[0]),marg1:(marg1+newdims[1]),:]
                        # py.figure(1)
                        # py.imshow(bb)
                        # py.show()
                        if (bb.shape==newdims):
                            IM.append(bb)
                        else:
                            print("error",fn,bb.shape)
    return(IM)

def standardize_data(aa):
    m=np.mean(np.reshape(aa,(aa.shape[0],np.prod(aa.shape[1:]))),axis=1)
    m=m[:,np.newaxis,np.newaxis,np.newaxis]
    s=np.std(np.reshape(aa,(aa.shape[0],np.prod(aa.shape[1:]))),axis=1)
    s=s[:,np.newaxis,np.newaxis,np.newaxis]
    aa=(aa-m)/s
    return(aa)

def load_single_player_data(use_existing=False, num_train=0):
    aa=np.load('/Users/amit/Desktop/Dropbox/Markov/IMSPL.npy')
    bb=np.load('/Users/amit/Desktop/Dropbox/Markov/IMSBGD.npy')
    aa=standardize_data(aa)
    bb=standardize_data(bb)


    #ii=np.int32(np.floor(np.random.rand(100)*bb.shape[0]))
    # py.figure(1)
    # for j,i in enumerate(ii):
    #     py.subplot(10,10,j+1)
    #     py.imshow(bb[i,:,:,:])
    #     py.axis('off')
    #     py.axis('equal')
    # py.show()
    if (num_train==0):
        num=aa.shape[0]
    else:
        num=np.minimum(aa.shape[0],num_train)
    if (not use_existing):
         ii=range(num)
         np.random.shuffle(ii)
         np.save('ii.npy',ii)
         aa=aa[ii,]
    else:
        if (os.path.isfile('ii.npy')):
            ii=np.load('ii.npy')
            aa=aa[ii,]
    train_num=np.int32(num/2)
    val_num=np.int32(num/4)
    test_num=np.int32(num/4)
    head=aa[:,0:25,:,:]
    body=aa[:,20:45,:,:]
    legs=aa[:,35:60,:,:]
    bgd=bb[:,20:45,:,:]
    val_start=train_num
    val_end=val_num+val_start
    test_start=val_end
    test_end=test_num+test_start
    X_train=scipy.vstack((head[0:train_num,],body[0:train_num,],legs[0:train_num],bgd[0:train_num,]))
    X_val=scipy.vstack((head[val_start:val_end,],body[val_start:val_end,],
                        legs[val_start:val_end,],bgd[val_start:val_end,]))
    X_test=scipy.vstack((head[test_start:test_end,],
                         body[test_start:test_end,],
                         legs[test_start:test_end,],
                         bgd[test_start:test_end,]))

    X_train=X_train.transpose((0,3,1,2)) #/256.
    X_val=X_val.transpose((0,3,1,2)) #/256.
    X_test=X_test.transpose((0,3,1,2)) #/256.
    y_train=np.repeat(range(4),train_num)
    y_val=np.repeat(range(4),val_num)
    y_test=np.repeat(range(4),test_num)

    return (np.float32(X_train),np.uint8(y_train),np.float32(X_val),np.uint8(y_val),np.float32(X_test),np.uint8(y_test))


use_existing=True
time_step=.00001
#DATA=load_single_player_data(use_existing=use_existing, num_train=10000)

try_moving_around()
#players_net.main_old(num_epochs=100,num_train=0,DATA=DATA,use_existing=use_existing, time_step=time_step)



# dir='BURNLYxCFC/'
# IM=load_from_jpg(dir,dosize=False)
# IM=np.uint8(np.array(IM))
# np.save('IMSBGD.npy',IM)
