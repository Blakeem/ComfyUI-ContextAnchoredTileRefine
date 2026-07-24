Source: https://docs.scipy.org/doc/scipy-1.15.1/reference/generated/scipy.ndimage.distance_transform_edt.html
Version: SciPy v1.15.1 Manual (scipy 1.15.1)
Retrieved: 2026-07-22

# distance_transform_edt

**scipy.ndimage.distance_transform_edt**(*input*, *sampling=None*, *return_distances=True*, *return_indices=False*, *distances=None*, *indices=None*)

[\[source\]](https://github.com/scipy/scipy/blob/v1.15.1/scipy/ndimage/_morphology.py#L0-L1)

Exact Euclidean distance transform.

This function calculates the distance transform of the *input*, by
replacing each foreground (non-zero) element, with its
shortest distance to the background (any zero-valued element).

In addition to the distance transform, the feature transform can
be calculated. In this case the index of the closest background
element to each foreground element is returned in a separate array.

**Parameters:**

- **input** : *array_like*
  Input data to transform. Can be any type but will be converted
  into binary: 1 wherever input equates to True, 0 elsewhere.
- **sampling** : *float, or sequence of float, optional*
  Spacing of elements along each dimension. If a sequence, must be of
  length equal to the input rank; if a single number, this is used for
  all axes. If not specified, a grid spacing of unity is implied.
- **return_distances** : *bool, optional*
  Whether to calculate the distance transform.
  Default is True.
- **return_indices** : *bool, optional*
  Whether to calculate the feature transform.
  Default is False.
- **distances** : *float64 ndarray, optional*
  An output array to store the calculated distance transform, instead of
  returning it.
  *return_distances* must be True.
  It must be the same shape as *input*.
- **indices** : *int32 ndarray, optional*
  An output array to store the calculated feature transform, instead of
  returning it.
  *return_indicies* must be True.
  Its shape must be `(input.ndim,) + input.shape`.

**Returns:**

- **distances** : *float64 ndarray, optional*
  The calculated distance transform. Returned only when
  *return_distances* is True and *distances* is not supplied.
  It will have the same shape as the input array.
- **indices** : *int32 ndarray, optional*
  The calculated feature transform. It has an input-shaped array for each
  dimension of the input. See example below.
  Returned only when *return_indices* is True and *indices* is not
  supplied.

### Notes

The Euclidean distance transform gives values of the Euclidean
distance:

```
              n
y_i = sqrt(sum (x[i]-b[i])**2)
              i
```

where b[i] is the background point (value 0) with the smallest
Euclidean distance to input points x[i], and n is the
number of dimensions.

### Examples

```pycon
>>> from scipy import ndimage
>>> import numpy as np
>>> a = np.array(([0,1,1,1,1],
...               [0,0,1,1,1],
...               [0,1,1,1,1],
...               [0,1,1,1,0],
...               [0,1,1,0,0]))
>>> ndimage.distance_transform_edt(a)
array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
       [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
       [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
```

With a sampling of 2 units along x, 1 along y:

```pycon
>>> ndimage.distance_transform_edt(a, sampling=[2,1])
array([[ 0.    ,  1.    ,  2.    ,  2.8284,  3.6056],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  3.    ],
       [ 0.    ,  1.    ,  2.    ,  2.2361,  2.    ],
       [ 0.    ,  1.    ,  2.    ,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
```

Asking for indices as well:

```pycon
>>> edt, inds = ndimage.distance_transform_edt(a, return_indices=True)
>>> inds
array([[[0, 0, 1, 1, 3],
        [1, 1, 1, 1, 3],
        [2, 2, 1, 3, 3],
        [3, 3, 4, 4, 3],
        [4, 4, 4, 4, 4]],
       [[0, 0, 1, 1, 4],
        [0, 1, 1, 1, 4],
        [0, 0, 1, 4, 4],
        [0, 0, 3, 3, 4],
        [0, 0, 3, 3, 4]]], dtype=int32)
```

With arrays provided for inplace outputs:

```pycon
>>> indices = np.zeros(((np.ndim(a),) + a.shape), dtype=np.int32)
>>> ndimage.distance_transform_edt(a, return_indices=True, indices=indices)
array([[ 0.    ,  1.    ,  1.4142,  2.2361,  3.    ],
       [ 0.    ,  0.    ,  1.    ,  2.    ,  2.    ],
       [ 0.    ,  1.    ,  1.4142,  1.4142,  1.    ],
       [ 0.    ,  1.    ,  1.4142,  1.    ,  0.    ],
       [ 0.    ,  1.    ,  1.    ,  0.    ,  0.    ]])
>>> indices
array([[[0, 0, 1, 1, 3],
        [1, 1, 1, 1, 3],
        [2, 2, 1, 3, 3],
        [3, 3, 4, 4, 3],
        [4, 4, 4, 4, 4]],
       [[0, 0, 1, 1, 4],
        [0, 1, 1, 1, 4],
        [0, 0, 1, 4, 4],
        [0, 0, 3, 3, 4],
        [0, 0, 3, 3, 4]]], dtype=int32)
```
