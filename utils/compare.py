import numpy as np 

def compare_npy(filename1,filename2):
    arr1 = np.load(filename1)
    arr2 = np.load(filename2)
    print('two array dtype are:',arr1.dtype,arr2.dtype)
    assert arr1.dtype==arr2.dtype
    diff = np.mean(abs(arr1-arr2))
    print('the diff between two array is: ',diff)
    return diff

def compare_arr(arr1,arr2):
    print('two array dtype are:',arr1.dtype,arr2.dtype)
    assert arr1.dtype==arr2.dtype
    diff = np.mean(abs(arr1-arr2))
    print('the diff between two array is: ',diff)
    return diff

def compare_bin(filename1,filename2,shape):
    if shape:
        arr1 = np.fromfile(filename1,np.float32).reshape(shape)
        arr2 = np.fromfile(filename2,np.float32).reshape(shape)
    else:
        arr1 = np.fromfile(filename1,np.float32)
        arr2 = np.fromfile(filename2,np.float32)      
    print('two array dtype are:',arr1.dtype,arr2.dtype)
    assert arr1.dtype==arr2.dtype
    diff = np.mean(abs(arr1-arr2))
    print('{} the diff between two array is: '.format(filename1),diff)
    return diff

if __name__=="__main__":
    # shape = (1, 25, 72, 120)
    shape = None
    # pytorch_bin = './pytorch_bins/'+'conv3a_r.bin'
    # ppl_bin = './script/'+'ppl_output-291-1_32_72_120.dat'
    # compare_bin(pytorch_bin,ppl_bin,shape)
    # print("-"*20)
    # pytorch_bin = './pytorch_bins/'+'conv3a_r.bin'
    # ppl_bin = './script/'+'ppl_output-291-1_32_72_120.dat'
    # compare_bin(pytorch_bin,ppl_bin,shape)
    print("-"*20)
    pytorch_bin = './pytorch_bins/'+'out_corr.bin'
    ppl_bin = './script/'+'ppl_output-292-1_25_72_120.dat'
    compare_bin(pytorch_bin,ppl_bin,shape)
    print("-"*20)
    pytorch_bin = './pytorch_bins/'+'out_pr0.bin'
    ppl_bin = './script/'+'ppl_output-428-1_1_576_960.dat'
    compare_bin(pytorch_bin,ppl_bin,shape)
    print("-"*20)
    pytorch_bin = './pytorch_bins/'+'output.bin'
    ppl_bin = './script/'+'ppl_output-428-1_1_576_960.dat'
    compare_bin(pytorch_bin,ppl_bin,shape)