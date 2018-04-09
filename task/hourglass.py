import mxnet as mx

def conv(data, name, num_filter, kernel_shape=(3,3), stride=(1, 1), relu=True):
    pad = [(t-1)/2 for t in kernel_shape]
    conv_data = mx.symbol.Convolution(data=data, name=name, kernel=kernel_shape, stride=stride, pad=pad, num_filter=num_filter)
    if relu: 
        return mx.symbol.Activation(data=conv_data, act_type='relu')
    else: 
        return conv_data


def downsample(data, name=None, kernel=(2,2), stride=(2,2), pad=(0,0)):
    return mx.symbol.Pooling(data=data, name=name, kernel=kernel, stride=stride, pad=pad, pool_type='max')


def upsample(data, name=None, scale=2, sample_type='nearest'):
    return mx.symbol.UpSampling(data, name=name, scale=scale, sample_type=sample_type)


def hourglass(data, num_features, hg_id, n=4):
    num_features2 = num_features + 128

    up1 = conv(data, name='%d_%d_up1'%(hg_id,n), num_filter=num_features)

    pool = downsample(data, name='%d_%d_pool'%(hg_id,n))
    low1 = conv(pool, name='%d_%d_low1'%(hg_id,n), num_filter=num_features2)
    if n>1:
        low2 = hourglass(low1, num_features, hg_id, n-1)
    else:
        low2 = conv(low1, name='%d_%d_low2'%(hg_id,n), num_filter=num_features2)
    low3 = conv(low2, name='%d_%d_low3'%(hg_id,n), num_filter=num_features)
    up2 = upsample(low3, name='%d_%d_upsampling'%(hg_id,n))

    return mx.sym.ElementWiseSum(*[up2, up1], name="%d_%d_sum"%(hg_id,n))


def stacked_hourglass(data, npk, npl, num_features=256, num_stacks=8, mode='test', train_mode=0):
    """ Stacked Hourglass Network

    Args:
        data (sym.Variable): input 
        npk (int): nums of keypoints
        npl (int): nums of limps
        num_features (int): nums of internal features 
        num_stacks (int): nums of stacks
        mode (str): 'train' or 'test'
        train_mode (int): 0 or 1, for different method to calculate loss

    Returns: 
        "train": Group[loss1, loss2, [values]] 
        "test": Group[keypoints, limbs]

    """
    conv1 = conv(data, 'conv1', 64, kernel_shape=(7,7), stride=(2,2))
    conv2 = conv(conv1, 'conv2', 128)
    pool1 = downsample(conv2, 'pool1')
    conv3 = conv(pool1, 'conv3', 128)
    conv4 = conv(conv3, 'conv4', 128)
    conv5 = conv(conv4, 'conv5', num_features)

    predicts = []
    keypoints = []
    limbs = []

    inter = conv5
    for stack_ind in range(num_stacks):
        hg = hourglass(inter, num_features, stack_ind)

        conv6 = conv(hg, 'conv6_%d'%stack_ind, num_features)
        conv7 = conv(conv6, 'conv7_%d'%stack_ind, num_features, kernel_shape=(1,1))

        predicts  += [conv(conv7, 'out_%d'%stack_ind, npk+1+2*npl, kernel_shape=(1,1), relu=False)]
        keypoints += [mx.symbol.slice_axis(data=predicts[-1], axis=1, begin=0, end=npk+1)]
        limbs     += [mx.symbol.slice_axis(data=predicts[-1], axis=1, begin=npk+1, end=npk+2*npl+1)]

        if stack_ind < num_stacks-1:
            inter = inter +\
                    conv(conv7, 'tmp1_%d'%stack_ind, num_features, kernel_shape=(1,1), relu=False) +\
                    conv(predicts[-1], 'tmp2_%d'%stack_ind, num_features, kernel_shape=(1,1), relu=False)
    
    if mode == 'test':
        group = mx.symbol.Group([keypoints[-1], limbs[-1]])
    elif mode == 'train':
        heatmap_gt = mx.sym.Variable('heatmap_gt')
        partaffinity_gt = mx.sym.Variable('partaffinity_gt')
        heatweight = mx.sym.Variable('heatweight')
        vecweight = mx.sym.Variable('vecweight')
        loss_L2 = [] # heatmap loss
        loss_L1 = [] # part affinity loss

        if train_mode == 0:
            for i in range(num_stacks):
                loss_L2 += [mx.symbol.MakeLoss(mx.symbol.square(heatweight * (keypoints[i] - heatmap_gt)))]
                loss_L1 += [mx.symbol.MakeLoss(mx.symbol.square(vecweight * (limbs[i] - partaffinity_gt)))]
        elif train_mode == 1:
            np1 = npk + 1
            np2 = 2 * npl
            for i in range(num_stacks):
                heatmap_gt_slice = mx.symbol.slice_axis(data=heatmap_gt, axis=1, begin=np1*i, end=np1*(i+1))
                partaffinity_gt_slice = mx.symbol.slice_axis(data=partaffinity_gt, axis=1, begin=np2*i, end=np2*(i+1))
                loss_L2 += [mx.symbol.MakeLoss(mx.symbol.square(heatweight * (keypoints[i] - heatmap_gt)))]
                loss_L1 += [mx.symbol.MakeLoss(mx.symbol.square(vecweight * (limbs[i] - partaffinity_gt)))]

        group = mx.symbol.Group(loss_L1 + 
                                loss_L2 +
                                [mx.sym.BlockGrad(keypoints[-1]),
                                 mx.sym.BlockGrad(limbs[-1]),
                                 mx.sym.BlockGrad(heatmaplabel), 
                                 mx.sym.BlockGrad(partaffinityglabel),
                                 mx.sym.BlockGrad(heatweight), 
                                 mx.sym.BlockGrad(vecweight)])
    return group



def test_hourglass():
    data = mx.sym.Variable(name='data')
    data_array = mx.nd.zeros((1,3,256,256))
    net_symbol = hourglass(data, 256,  0)
    mod = mx.mod.Module(net_symbol, label_names=None)
    mod.bind(data_shapes=[('data', data_array.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([data_array]))
    print(mod.get_outputs()[0].asnumpy().shape)
    #mx.viz.plot_network(symbol=net_symbol, shape={'data':data_array.shape}, title='symbol', save_format='jpg').render()
    

def test_stacked_hourglass():
    data = mx.sym.Variable(name='data')
    data_array = mx.nd.zeros((1,3,256,256))
    net_symbol = stacked_hourglass(data, 16, 10, mode='test')
    mod = mx.mod.Module(net_symbol, label_names=None)
    mod.bind(data_shapes=[('data', data_array.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([data_array]))
    print(mod.get_outputs()[0].asnumpy().shape)
    print(mod.get_outputs()[1].asnumpy().shape)
    #mx.viz.plot_network(symbol=net_symbol, shape={'data':data_array.shape}, title='symbol', save_format='jpg').render()


if __name__ == "__main__":
    #test_hourglass()
    test_stacked_hourglass()
