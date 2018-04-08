import mxnet as mx

#HourglassPose

def cnv(inp, kernel_shape, name, num_filter,stride=(1, 1),dorelu=True):
    pad=[(t-1)/2 for t in kernel_shape]
    conv=mx.symbol.Convolution(data=inp, kernel=kernel_shape,stride=stride,name=name,pad=pad,num_filter=num_filter)
    if dorelu: return mx.symbol.Activation(data=conv,act_type='relu')
    else: return conv

def pool(inp, name=None, kernel=(2,2), stride=(2,2), pad=(0,0)):
    return mx.symbol.Pooling(data=inp, kernel=kernel, stride=stride, pad=pad, pool_type='max', name=name)

def hourglass(inp,n,f,hg_id):
    nf=f+128
    up1=cnv(inp,kernel_shape=(3,3),num_filter=f,name='%d_%d_up1'%(hg_id,n))
    pool1=pool(inp,name='%d_%d_pool'%(hg_id,n))
    low1=cnv(pool1,kernel_shape=(3,3),num_filter=nf,name='%d_%d_low1'%(hg_id,n))
    if n>1:
        low2=hourglass(low1,n-1,nf,hg_id)
    else:
        low2=cnv(low1,kernel_shape=(3,3),num_filter=nf,name='%d_%d_low2'%(hg_id,n))
    low3=cnv(low2,kernel_shape=(3,3),num_filter=f,name='%d_%d_low3'%(hg_id,n))

    up2 = mx.symbol.UpSampling(low3, scale=2, sample_type='nearest', name='%d_%d_upsampling'%(hg_id,n))
    #up2_clip = mx.symbol.Crop(*[up2, up1], name="%d_%d_clip"%(hg_id,n))
    return mx.sym.ElementWiseSum(*[up2, up1], name="%d_%d_sum"%(hg_id,n))

def HGSymbol(config):
    npk=config.npk
    npl=config.npl
    f=256
    data = mx.symbol.Variable(name='data')
    cnv1=cnv(data,kernel_shape=(7,7),num_filter=64,name='cnv1',stride=(2,2))
    cnv2=cnv(cnv1,kernel_shape=(3,3),num_filter=128,name='cnv2')
    pool1=pool(cnv2,'pool1')
    cnv2b=cnv(pool1,kernel_shape=(3,3),num_filter=128,name='cnv2b')
    cnv3=cnv(cnv2b,kernel_shape=(3,3),num_filter=128,name="cnv3")
    cnv4=cnv(cnv3,kernel_shape=(3,3),num_filter=f,name='cnv4')
    inter=cnv4
    preds=[]
    keypoints=[]
    limbs=[]
    for i in range(config.stage):
        hg=hourglass(inter,4,f,i)

        cnv5=cnv(hg,kernel_shape=(3,3),num_filter=f,name='cnv5_%d'%i)
        cnv6=cnv(cnv5,kernel_shape=(1,1),num_filter=f,name='cnv6_%d'%i)
        preds += [cnv(cnv6, kernel_shape=(1, 1),num_filter=npk+1+2*npl, name='out_%d' % i, dorelu=False)]
        keypoints += [mx.symbol.slice_axis(data=preds[-1], axis=1, begin=0, end=npk+1)]
        limbs     += [mx.symbol.slice_axis(data=preds[-1], axis=1, begin=npk+1, end=npk+2*npl+1)]

        if i<3:
            inter=inter\
                  +cnv(cnv6,kernel_shape=(1,1),num_filter=f,name='tmp_%d'%i,dorelu=False)\
                  +cnv(preds[-1],kernel_shape=(1,1),num_filter=f,name='tmp_out_%d'%i,dorelu=False)

    if (config.mode == 'TRAIN'):
        heatmaplabel = mx.sym.Variable("heatmaplabel")
        partaffinityglabel = mx.sym.Variable('partaffinityglabel')
        heatweight = mx.sym.Variable('heatweight')
        vecweight = mx.sym.Variable('vecweight')
        #pafmap loss
        loss_L1=[]
        #heatmap loss
        loss_L2=[]

        if config.TRAIN.mode==1:
            np1 = npk + 1
            np2 = 2 * npl
            for i in range(config.stage):
                heatmaplabel_slice=mx.symbol.slice_axis(data=heatmaplabel, axis=1, begin=np1*i, end=np1*(i+1))
                partaffinityglabel_slice = mx.symbol.slice_axis(data=partaffinityglabel, axis=1, begin=np2*i, end=np2*(i+1))
                loss_L1 +=[mx.symbol.MakeLoss(mx.symbol.square(vecweight*(limbs[i]-partaffinityglabel_slice)))]
                loss_L2 +=[mx.symbol.MakeLoss(mx.symbol.square(heatweight*(keypoints[i] - heatmaplabel_slice)))]

        elif config.TRAIN.mode==0:
            for i in range(config.stage):
                loss_L1+=[mx.symbol.MakeLoss(mx.symbol.square(vecweight * (limbs[i] - partaffinityglabel)))]
                loss_L2+=[mx.symbol.MakeLoss(mx.symbol.square(heatweight * (keypoints[i] - heatmaplabel)))]

        group = mx.symbol.Group(loss_L1+loss_L2+
                                [mx.sym.BlockGrad(keypoints[-1]),mx.sym.BlockGrad(limbs[-1]),
                                 mx.sym.BlockGrad(heatmaplabel), mx.sym.BlockGrad(partaffinityglabel),
                                 mx.sym.BlockGrad(heatweight), mx.sym.BlockGrad(vecweight)])
        return group
    elif config.mode == 'TEST':
        group = mx.symbol.Group([limbs[-1], keypoints[-1]])
        return group


def test_hourglass():
    num_stacks = 1
    f = 256
    hg_id = 0

    data = mx.sym.Variable(name='data')
    data_array = mx.nd.zeros((1,3,256,256))
    net_symbol = hourglass(data, num_stacks, f, hg_id)
    mod = mx.mod.Module(net_symbol, label_names=None)
    mod.bind(data_shapes=[('data', data_array.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([data_array]))
    print(mod.get_outputs()[0].asnumpy().shape)
    #mx.viz.plot_network(symbol=net_symbol, shape={'data':data_array.shape}, title='symbol', save_format='jpg').render()
    

if __name__ == "__main__":
    test_hourglass()
