from multiprocessing import Pool
import os


def theano_test_wrapper(tuple_in):
    return theano_test(*tuple_in)
def theano_test(gpu,x1,x2):
    os.environ["THEANO_FLAGS"] = "device=%s,floatX=float32,nvcc.fastmath=True"%gpu
    import theano
    print theano.config.device
    x = theano.tensor.scalar()
    y = theano.tensor.scalar()
    z=x*y
    times = theano.function([x,y],z)
    print times(x1,x2)

def main():
    gpulist = ['gpu%d'%x for x in range(4)]
    x = range(4)
    y = range(4)
    args = zip(gpulist,x,y)
    print args[0]
    theano_test_wrapper(args[0])
    p=Pool(4)
    p.map(theano_test_wrapper,args)
if __name__ == '__main__':
    main()
