import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn


def Conv(channels, kernel_size, **kwargs):
    out = nn.HybridSequential(**kwargs)
    padding = 1 if kernel_size == 3 else 0
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=1, padding=padding, use_bias=False),
            nn.BatchNorm(scale=True),
            nn.LeakyReLU(0.1)
        )
    return out


def Stage(channels, repeats, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        for _ in range((repeats-1)/2):
            out.add(Conv(channels, 3),
                    Conv(channels/2, 1))
        if channels == 1024: # ignore maxpooling for the last stage
            out.add(Conv(channels, 3))
        else:
            out.add(Conv(channels, 3),
                    nn.MaxPool2D(2, 2))
    return out


class Darknet19(nn.HybridSequential):
    def __init__(self, num_classes, **kwargs):
        super(Darknet19, self).__init__(**kwargs)

        self._setting = [
            #t, c, 
            [1, 32, "stage1_"],  
            [1, 64, "stage2_"],  
            [3, 128,  "stage3_"],  
            [3, 256,  "stage4_"],  
            [5, 512,  "stage5_"],  
            [5, 1024, "stage6_"], 
        ]
        with self.name_scope():
            self.features = nn.HybridSequential()
            for t, c, prefix in self._setting:
                self.features.add(Stage(c, t, prefix=prefix))
            self.features.add(Conv(num_classes, 1),
                              nn.GlobalAvgPool2D())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    x = nd.random_normal(shape=(1, 3, 224, 224))
    net = Darknet19(1000)
    net.collect_params().initialize()
    y = net(x)
    print(y.shape)

    sym = net(mx.sym.var('data'))
    mx.viz.plot_network(symbol=sym, shape={'data':(1,3,224,224)}, title='symbol', save_format='jpg').render()
