
import numpy as np
from numba import njit, prange, set_num_threads

powers_of_two = 1 << np.arange(64, dtype=np.uint64)

@njit
def packbits(bool_array):
    ndim1, ndim2 = np.shape(bool_array)
    ndim2_out = int(np.ceil(ndim2/64))
    res = np.empty((ndim1, ndim2_out), dtype = np.uint64)
    for i in range(ndim1):
        for j in range(0, ndim2_out):
            tmp = bool_array[i, j*64:min((j+1)*64, ndim2)]
            res[i,j] = np.sum(powers_of_two[0:tmp.size] * tmp)
    return res

@njit
def unpackbits(int_array, nq):
    ndim1, ndim2 = np.shape(int_array)
    assert nq >= ndim2, 'Cannot unpack ' + str(ndim2) + ' integers into ' + str(nq) + ' qubits'
    res = np.empty((ndim1, nq), dtype = np.bool8)
    for i in range(ndim1):
        for j in range(0, nq, 64):
            res[i, j:min(j+64, nq)] = int_array[i][int(j/64)] & powers_of_two[0:min(64, nq-j)] 
    return res

@njit
def parityReprBits(n):
    r = n
    for _ in range(64):
        n<<=1
        r^=n
    return r

@njit
def parity_repr(a):
    sign = False
    b=np.zeros_like(a)
    for i in range(len(a)):
        b[i] = parityReprBits(a[i])
        if sign:
            b[i] = ~b[i]
        sign = ((b[i]>>64) and 1)
    return b

@njit(parallel=True)
def parity_repr_array(a):
    b=np.zeros_like(a)
    for i in prange(len(a)):
        b[i] = parity_repr(a[i])
    return b

@njit
def parity(a):
    ndim=len(a)
    s = np.zeros(ndim, dtype=np.bool8)
    for i in prange(ndim):
        s[i]^=np.mod(count_nonzero(a[i]), 2)
    return s

@njit
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count

@njit
def count_nonzero(a):
    s = 0
    for i in range(len(a)):
        s+=countSetBits(a[i])
    return s

@njit(parallel=True)
def count_nonzero_array(a):
    ndim = len(a)
    s = np.zeros(ndim, dtype=np.int32)
    for i in prange(ndim):
        s[i]+=count_nonzero(a[i])
    return s

@njit(parallel=True)
def count_and(a,b):
    c = np.bitwise_and(a, b)
    return count_nonzero_array(c)

@njit(parallel=True)
def count_or(a,b):
    c = np.bitwise_or(a, b)
    return count_nonzero_array(c)

@njit(parallel=True)
def count_and_array_bool(a, b):
    res = np.empty(len(a), dtype=np.bool8)
    for i in prange(len(a)):
        res[i] = np.mod(count_nonzero(np.bitwise_and(a[i, :], b[:])), 2)
    return res

@njit(parallel=True)
def bits_equal(a, b):
    c = np.empty(len(b), dtype=np.bool8)
    for i in prange(len(c)):
        c[i] = np.all(a[i, :] == b[i, :])
    return c

@njit(parallel=True)
def bits_equal_index(a, b, index):
    c = np.empty(len(b), dtype=np.bool8)
    for i in prange(len(c)):
        c[i] = ~np.any(a[index[i], :] != b[i, :])
    return c

@njit(parallel=True)
def inplace_xor(a,b):
    for i in prange(len(a)):
        a[i,:]^=b

@njit(parallel=True)
def a_lt_b(a, b, out):
    for i in prange(len(out)):
        out[i] = np.abs(a[i]) < b

@njit(parallel=True)
def a_gt_b(a, b, out):
    for i in prange(len(out)):
        out[i] = np.abs(a[i]) >= b

@njit(parallel=True)
def a_gt_b_and_not_c(a, b, c, out):
    for i in prange(len(out)):
        out[i] = (np.abs(a[i]) >= b) & ~c[i]

@njit(parallel=True)
def a_gt_b_or_c(a, b, c, out):
    for i in prange(len(out)):
        out[i] = (np.abs(a[i]) >= b) or c[i]

@njit(parallel=True)
def find_bit_index(a, b):
    size_1, size_2 = a.shape
    size_b = len(b)
    lower = np.repeat(0, size_b)
    upper = np.repeat(size_1, size_b)
    for j in prange(size_b):
        for i in range(size_2):
            if upper[j] == lower[j]:
                break
            lower[j] = lower[j] + np.searchsorted(a[lower[j]:upper[j], i], b[j, i], side='left')
            upper[j] = lower[j] + np.searchsorted(a[lower[j]:upper[j], i], b[j, i], side='right')
    return lower

@njit(parallel=True)
def insert_index(a,b, ac, bc, index):
    new_size = len(a) + len(b)
    res = np.empty((new_size, a.shape[1]), dtype=np.uint64)
    res_c = np.empty(new_size, dtype=np.complex128)
    ind = index+np.arange(len(index))
    res[:ind[0]] = a[:index[0]]
    res_c[:ind[0]] = ac[:index[0]]
    for i in prange(len(ind)):
        res[ind[i]] = b[i]
        res_c[ind[i]] = bc[i]
        if i==len(ind)-1:
            u = new_size
            ua = len(a)
        else:
            u = ind[i+1]
            ua = index[i+1]
        res[ind[i]+1:u] = a[index[i]:ua]
        res_c[ind[i]+1:u] = ac[index[i]:ua]
    return res, res_c

def insert_index_serial(a,b, ac, bc, index):
    new_size = len(a) + len(b)
    res = np.empty((new_size, a.shape[1]), dtype=np.uint64)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.zeros(new_size, dtype=np.bool8)
    mask[index+np.arange(len(index))] = True
    res[mask] = b[:]
    res_c[mask] = bc[:]
    mask = ~mask
    res[mask] = a[:]
    res_c[mask] = ac[:]
    return res, res_c

@njit(parallel=True)
def delete_index(a, ac, index):
    new_size = len(a) - len(index)
    res = np.empty((new_size, a.shape[1]), dtype=np.uint64)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.ones(len(a), dtype=np.bool8)
    mask[index] = False 
    ind = np.nonzero(mask)[0]
    for i in prange(len(ind)):
        res[i] = a[ind[i]]
        res_c[i] = ac[ind[i]]
    return res, res_c

@njit
def delete_index_serial(a, ac, index):
    mask = np.ones(len(a), dtype=np.bool8)
    mask[index] = False 
    res = a[mask]
    res_c = ac[mask]
    return res, res_c

@njit(parallel=True)
def pmult(a, b):
    a[:] = a[:] * b

@njit(parallel=True)
def pmult_array(a, b):
    a[:] = a[:] * b[:]

@njit(parallel=True)
def pmult_mask(a, b, mask):
    a[mask] = a[mask] * b

@njit(parallel=True)
def pmult_sign(a, b, sign):
    for i in prange(len(a)):
        a[i] = a[i] * b
        if sign[i]:
            a[i] = -a[i]

@njit(parallel=True)
def psum_index(a, b, index1):
    for i in prange(len(index1)):
        a[index1[i]] += b[i]

@njit
def remove_duplicates(a, ac):
    i=0
    while i < len(a)-1:
        c=1
        while (a[i, :] == a[i+c, :]).all():
            ac[i] += ac[i+c]
            ac[i+c] = 0
            c+=1
        i+=c