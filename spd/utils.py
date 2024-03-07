
import numpy as np
from numba import njit, prange, set_num_threads
from math import ceil

from tqdm.notebook import tqdm

powers_of_two = 1 << np.arange(64, dtype=np.uint64)

def my_tqdm(progress_bar, iterable):
    if progress_bar:
        return tqdm(iterable, leave=False)
    else:
        return iterable
    
@njit
def packbits(bool_array):
    ndim1, ndim2 = np.shape(bool_array)
    ndim2_out = ceil(ndim2/64)
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
    res = np.empty((ndim1, nq), dtype = np.bool_)
    for i in range(ndim1):
        for j in range(0, nq, 64):
            res[i, j:min(j+64, nq)] = int_array[i][int(j/64)] & powers_of_two[0:min(64, nq-j)] 
    return res

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
def not_equal(a,b):
    c = np.empty(len(a), dtype=np.bool_)
    c = (a != b)
    return c

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
    a[:,:] = np.bitwise_xor(a, b)

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
def find_bit_index(a, b, size_a, nq):
    lower = np.repeat(0, len(b))
    upper = np.repeat(size_a, len(b))
    for j in prange(len(b)):
        for i in range(2*nq):
            if upper[j] == lower[j]:
                break
            lower[j] = lower[j] + np.searchsorted(a[lower[j]:upper[j], i], b[j, i], side='left')
            upper[j] = lower[j] + np.searchsorted(a[lower[j]:upper[j], i], b[j, i], side='right')
    return lower

@njit(parallel=True)
def anticommutation_relation(a, b):
    res = np.empty(len(a), dtype=np.int16)
    for i in prange(len(a)):
        res[i] = np.mod(count_nonzero(np.bitwise_and(a[i,:], b[:])), 2)
    return res

@njit(parallel=True)
def anticommutation_relation_list(a, b):
    res = np.empty((len(a), len(b)), dtype=np.int16)
    for i in prange(len(a)):
        for j in range(len(b)):
            res[i, j] = np.mod(count_nonzero(np.bitwise_and(a[i,:], b[j,:])), 2)
    return res

@njit(parallel=True)
def update_phase(p1, p2, a, b):
    for i in prange(len(p1)):
        p1[i] = p1[i] + p2 + 2*count_nonzero(np.bitwise_and(a[i, :], b[:]))
    
@njit(parallel=True)
def insert_index(a,b, ap, bp, ac, bc, index, nq):
    new_size = len(a) + len(b)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    ind = index+np.arange(len(index))
    res[:ind[0]] = a[:index[0]]
    res_p[:ind[0]] = ap[:index[0]]
    res_c[:ind[0]] = ac[:index[0]]
    for i in prange(len(ind)):
        res[ind[i]] = b[i]
        res_p[ind[i]] = bp[i]
        res_c[ind[i]] = bc[i]
        if i==len(ind)-1:
            u = new_size
            ua = len(a)
        else:
            u = ind[i+1]
            ua = index[i+1]
        res[ind[i]+1:u] = a[index[i]:ua]
        res_p[ind[i]+1:u] = ap[index[i]:ua]
        res_c[ind[i]+1:u] = ac[index[i]:ua]
    return res, res_p, res_c

def insert_index_serial(a,b, ap, bp, ac, bc, index, nq):
    new_size = len(a) + len(b)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.zeros(new_size, dtype=np.bool_)
    mask[index+np.arange(len(index))] = True
    res[mask] = b[:]
    res_p[mask] = bp[:]
    res_c[mask] = bc[:]
    mask = ~mask
    res[mask] = a[:]
    res_p[mask] = ap[:]
    res_c[mask] = ac[:]
    return res, res_p, res_c

@njit(parallel=True)
def delete_index(a, ap, ac, index, nq):
    new_size = len(a) - len(index)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    mask = np.ones(len(a), dtype=np.bool_)
    mask[index] = False 
    ind = np.nonzero(mask)[0]
    for i in prange(len(ind)):
        res[i] = a[ind[i]]
        res_p[i] = ap[ind[i]]
        res_c[i] = ac[ind[i]]
    return res, res_p, res_c

@njit
def delete_index_serial(a, ap, ac, index):
    mask = np.ones(len(a), dtype=np.bool_)
    mask[index] = False 
    res = a[mask]
    res_p = ap[mask]
    res_c = ac[mask]
    return res, res_p, res_c

@njit(parallel=True)
def pmult(a, b):
    a[:] = a[:] * b

@njit(parallel=True)
def pmult_array(a, b):
    a[:] = a[:] * b[:]

@njit(parallel=True)
def mask_mult(a, b, mask):
    for i in prange(len(a)):
        for j in range(len(b)):
            if mask[i,j]:
                a[i] = a[i] * b[j]

@njit(parallel=True)
def compose_mask(a, ap, ac, b, bp, bc, mask, new_size_array, nq):
    new_size = sum(new_size_array)
    res = np.empty((new_size, 2*nq), dtype=np.uint64)
    res_p = np.empty(new_size, dtype=np.int32)
    res_c = np.empty(new_size, dtype=np.complex128)
    for i in prange(len(bp)):
        l = sum(new_size_array[:i])
        u = l + new_size_array[i]
        m = mask[:, i]
        res_p[l:u] = ap[m] + bp[i] + 2*count_nonzero_array(np.bitwise_and(a[m, :nq], b[i, nq:]))
        res[l:u] = np.bitwise_xor(a[m, :], b[i, :])
        res_c[l:u] = ac[m] * bc[i]
    return res, res_p, res_c

@njit
def remove_duplicates(a, ap, ac):
    i=0
    while i < len(a)-1:
        c=1
        while (a[i, :] == a[i+c, :]).all():
            ac[i] += ac[i+c] * (-1j)**(ap[i+c] - ap[i])
            ac[i+c] = 0
            c+=1
        i+=c

@njit(parallel=True)
def update_coeffs(coeffs1, coeffs2, c, s, p1, p2, index1, index_exists):
    tmp = coeffs2.copy()
    pmult_array(tmp, index_exists * s * (-1j)**(p2-p1))
    coeffs1[index1] = coeffs1[index1] * c + tmp

@njit(parallel=True)
def tmp_product(c, p, p1, index, found):
    out = np.empty(len(c), dtype=np.complex128)
    for i in prange(len(c)):
        if found[i]:
            out[i] = c[i]  * (-1j)**(p[i] - p1[index[i]])
        else:
            out[i] = 0
    return out

@njit
def add_to_array(a, b, index):
    for i in range(len(index)):
        a[index[i]] += b[i]
