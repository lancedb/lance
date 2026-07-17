//Machine epsilon for f64:smallest value where 1.0 + EPS != 1.0
const EPS: f64 = 2.220_446_049_250_313e-16;
//Maximum number of Jacobi iteration sweeps before giving up
const MAX_JACOBI_SWEEPS: usize = 200;

//Creates an nxn identity matrix (row-major flat vector)
fn create_identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

//Applies a Givens rotation to a symmetric matrix S in-place
//Computes S ← G^T * S * G where G is the rotation in the (p,q) plane
fn jacobi_rotate(s: &mut [f64], n: usize, p: usize, q: usize, c: f64, sn: f64) {
    //Reads the current values of the 2x2 submatrix at (p,p), (q,q), and (p,q)
    let s_pp = s[p * n + p];
    let s_qq = s[q * n + q];
    let s_pq = s[p * n + q];

    //Updates the 2x2 submatrix using the closed-form Jacobi rotation formulas
    s[p * n + p] = c * c * s_pp - 2.0 * sn * c * s_pq + sn * sn * s_qq;
    s[q * n + q] = sn * sn * s_pp + 2.0 * sn * c * s_pq + c * c * s_qq;
    s[p * n + q] = 0.0;
    s[q * n + p] = 0.0;

    //Updates all other rows and columns that interact with p or q
    for r in 0..n {
        //Skips the 2x2 block already handles above
        if r == p || r == q {
            continue;
        }
        let s_rp = s[r * n + p];
        let s_rq = s[r * n + q];
        let new_rp = c * s_rp - sn * s_rq;
        let new_rq = sn * s_rp + c * s_rq;

        //Updates both (r,p) and (p,r) to maintain symmetry
        s[r * n + p] = new_rp;
        s[p * n + r] = new_rp;

        //Updates both (r,q) and (q,r) to maintain symmetry
        s[r * n + q] = new_rq;
        s[q * n + r] = new_rq;
    }
}

//Accumulates a Givens rotation into V from the right: V ← V * G
//Rotates columns p and q of V by angle (c, sn)
fn apply_givens_rotation_from_right(v: &mut [f64], n: usize, p: usize, q: usize, c: f64, sn: f64) {
    for r in 0..n {
        let vp = v[r * n + p];
        let vq = v[r * n + q];

        v[r * n + p] = c * vp - sn * vq;
        v[r * n + q] = sn * vp + c * vq;
    }
}

//Jacobi eigenvalue algorithm: Decomposes a symmetric matrix A into V * diag(eigenvalues) * V^T
//by repeatedly applying Givens rotation to zero out off-diagonal entries.
//Input: a - summetric nxn matrix (row-major flat vector)
//Output: (eigenvalues, V) where the columns of V are the eigenvectors
fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    //Copy of the matrix; this copy will be diagonalized in-place.
    let mut s = a.to_vec();
    //Accumulates the product of all Givens rotations
    let mut v = create_identity_matrix(n);

    for _ in 0..MAX_JACOBI_SWEEPS {
        //Finds the largest off-diagonal entry s[p,q]
        //This is the entry we will zero out in this iteration sweep.
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                let val = s[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        //If all off-diagonal entries are negligibly small, the matrix is
        //diagonal and the eigenvalues are on the diagonal.
        if max_val < EPS * 1e4 {
            break;
        }

        //Computes the Jacobi rotation angle theta that zeros out s[p,q]
        let s_pq = s[p * n + q];
        let diff = s[q * n + q] - s[p * n + p];
        let theta = if diff.abs() < EPS {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * s_pq / diff).atan()
        };
        let (sn, c) = theta.sin_cos();

        //Applies the Givens rotation: S ← G^T * S * G
        //This zeroes out s[p,q] and s[q,p] while updating the rest of the matrix.
        jacobi_rotate(&mut s, n, p, q, c, sn);

        //Accumulates the rotation into V: V ← V * G
        //At convergence, V's columns are the eigenvectors of the original matrix.
        apply_givens_rotation_from_right(&mut v, n, p, q, c, sn);
    }

    //Extract eigenvalues from the diagonal of the now-diagonalized matrix
    let eigenvalues: Vec<f64> = (0..n).map(|i| s[i * n + i]).collect();
    (eigenvalues, v)
}

//Computes C = A^T A where A is a mxn row-major flat vector
//and C is an nxn row-major flat vector
fn compute_ata(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0f64; n * n];
    for i in 0..n {
        for l in 0..m {
            let a_li = a[l * n + i];
            for j in 0..n {
                c[i * n + j] += a_li * a[l * n + j];
            }
        }
    }
    c
}

//Modified Gram-Schmidt orthonormalization
//Takes a list of column vectors and makes them orthonormal
//If a column is zero (e.g., from a zero singular value), it is
//replaced with a standard basis vector orthogonal to all prior columns.
fn gram_schmidt(cols: &mut Vec<Vec<f64>>) {
    let ncols = cols.len();
    for i in 0..ncols {
        //Subtracts the projection of cols[i] onto each already-orthonormal column
        for j in 0..i {
            let dot: f64 = cols[i].iter().zip(cols[j].iter()).map(|(&a, &b)| a * b).sum();
            let cj = cols[j].clone();
            for (a, b) in cols[i].iter_mut().zip(cj.iter()) {
                *a -= dot * b;
            }
        }

        //Normalizes the resulting vector
        let norm = cols[i].iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > EPS {
            for x in cols[i].iter_mut() {
                *x /= norm;
            }
        } else {
            //If the column is zero, finds a standard basis vector not in the span yet
            let dim = cols[i].len();
            'search: for k in 0..dim {
                let mut e = vec![0.0f64; dim];
                //Tries the k-th standard basis vector and 
                //orthogonalize it against all previous columns
                e[k] = 1.0;
                for j in 0..i {
                    let dot: f64 = e.iter().zip(cols[j].iter()).map(|(&a, &b)| a * b).sum();
                    let cj = cols[j].clone();
                    for (a, b) in e.iter_mut().zip(cj.iter()) {
                        *a -= dot * b;
                    }
                }
                let n2 = e.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if n2 > EPS {
                    //Found a valid replacement vector. Normalizes and uses it.
                    for x in e.iter_mut() {
                        *x /= n2;
                    }
                    cols[i] = e;
                    break 'search;
                }
            }
        }
    }
}

//Computes y = A * x where A is an mxn row-major flat vector,
//x is a vector of length n, and y is a vector of length m
fn multiply_a_by_vector(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
}

/// Computes the SVD of an `m x n` row-major matrix `a`, decomposing it into
/// `U * diag(sigma) * V^T`.
///
/// Returns `(u, sigma, vt)` where `u` is `m x m` (row-major), `sigma` has length
/// `min(m, n)` sorted descending, and `vt` is `n x n` (row-major, rows are right
/// singular vectors).
///
/// # Example
/// ```
/// let (u, sigma, vt) = svd(&[1.0, 0.0, 0.0, 1.0], 2, 2)?;
/// assert_eq!(u, vec![1.0, 0.0, 0.0, 1.0]);
/// assert_eq!(sigma, vec![1.0, 1.0]);
/// assert_eq!(vt, vec![1.0, 0.0, 0.0, 1.0]);
/// ```
pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    //Checks whether the input matrix A has at least 1 row and at least 1 column
    if a.len() == 0 || m == 0 || n == 0 {
        println!("Error: Matrix must have at least 1 row and at least 1 column.");
    }
    //Checks whether the data length of the input matrix A matches the
    //product of the specified number of rows and number of columns
    if a.len() != m * n {
        println!("Error: Data length of matrix must match the product of the specified number of rows and number of columns.");
    }
    //Checks whether the input matrix A contains null or infinite entries
    if !a.iter().all(|x| x.is_finite()) {
        println!("Error: Matrix must not contain null or infinite entries.");
    }
    //If any of the three input validation checks above fails,
    //all three output matrices are empty.
    if a.len() == 0 || m == 0 || n == 0 || a.len() != m * n || !a.iter().all(|x| x.is_finite()) {
        return (vec![], vec![], vec![]);
    }

    //Step 1: Forms A^T A (nxn symmetric positive semi-definite matrix)
    let ata = compute_ata(a, m, n);
    //Step 2: Eigendecomposes A^T A into eigenvalues λ and 
    //eigenvectors V using Jacobi iteration
    let (eigenvalues, v) = jacobi_eigen(&ata, n);
    
    //k is the number of singular values.
    let k = m.min(n);

    //Sorts all n eigenvalues in descending order
    let mut order: Vec<usize> = (0..n).collect();
    for i in 1..n {
        let mut j = i;
        while j > 0 && eigenvalues[order[j - 1]] < eigenvalues[order[j]] {
            order.swap(j - 1, j);
            j -= 1;
        }
    }

    //Step 3: Computes singular values σᵢ = √λᵢ for the top k eigenvalues
    let mut sigma: Vec<f64> = order[..k]
        .iter()
        .map(|&i| if eigenvalues[i] > 0.0 { eigenvalues[i].sqrt() } else { 0.0 })
        .collect();

    //Step 4: Computes the left singular vectors uᵢ = (1/σᵢ) * A * vᵢ
    //Builds U column by column using the sorted eigenvectors
    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(m);
    for index in 0..k {
        //eᵢ is the index of the idx-th largest eigenvector in V
        let ei = order[index];
        //Extracts the eigenvector eᵢ (column eᵢ of V, stored as a row-major flat nxn vector)
        let vi: Vec<f64> = (0..n).map(|r| v[r * n + ei]).collect();
        //Computes A * vᵢ to get the unnormalized left singular vector
        let av = multiply_a_by_vector(a, &vi, m, n);
        if sigma[index] > EPS * 10.0 {
            //Normalizes the left singular vector to get uᵢ
            u_cols.push(av.iter().map(|&x| x / sigma[index]).collect());
        } else {
            //The placeholder column of sigma is the zero singular value
            //This column is filled by the Gram-Schmidt algorithm.
            sigma[index] = 0.0;
            u_cols.push(vec![0.0; m]);
        }
    }

    //Step 5: If m > k, pad U with zero columns.
    //The Gram-Schmidt algorithm will fill these columns with
    //orthonormal vectors that complete the basis of R^m.
    for _ in k..m {
        u_cols.push(vec![0.0; m]);
    }

    //Orthonormalize all columns of U
    gram_schmidt(&mut u_cols);

    //Pack the columns of U into an mxm flat row-major matrix
    let mut u = vec![0f64; m * m];
    for (ci, col) in u_cols.iter().enumerate() {
        for (ri, &value) in col.iter().enumerate() {
            u[ri * m + ci] = value;
        }
    }

    //Step 6: Build V^T (nxn row-major flat vector)
    //Row i of V^T = column order[i] of V = the i-th right singular vector
    let mut vt = vec![0f64; n * n];
    for new_row in 0..n {
        let old_col = order[new_row];
        for c in 0..n {
            vt[new_row * n + c] = v[c * n + old_col];
        }
    }

    (u, sigma, vt)
}