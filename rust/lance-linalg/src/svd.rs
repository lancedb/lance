const EPS: f64 = 2.220_446_049_250_313e-16;
const MAX_JACOBI_SWEEPS: usize = 200;

fn create_identity_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn jacobi_rotate(s: &mut [f64], n: usize, p: usize, q: usize, c: f64, sn: f64) {
    let s_pp = s[p * n + p];
    let s_qq = s[q * n + q];
    let s_pq = s[p * n + q];

    s[p * n + p] = c * c * s_pp - 2.0 * sn * c * s_pq + sn * sn * s_qq;
    s[q * n + q] = sn * sn * s_pp + 2.0 * sn * c * s_pq + c * c * s_qq;
    s[p * n + q] = 0.0;
    s[q * n + p] = 0.0;

    for r in 0..n {
        if r == p || r == q {
            continue;
        }
        let s_rp = s[r * n + p];
        let s_rq = s[r * n + q];
        let new_rp = c * s_rp - sn * s_rq;
        let new_rq = sn * s_rp + c * s_rq;

        s[r * n + p] = new_rp;
        s[p * n + r] = new_rp;
        s[r * n + q] = new_rq;
        s[q * n + r] = new_rq;
    }
}

fn apply_givens_rotation_from_right(v: &mut [f64], n: usize, p: usize, q: usize, c: f64, sn: f64) {
    for r in 0..n {
        let vp = v[r * n + p];
        let vq = v[r * n + q];

        v[r * n + p] = c * vp - sn * vq;
        v[r * n + q] = sn * vp + c * vq;
    }
}

fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = a.to_vec();
    let mut v = create_identity_matrix(n);

    for _ in 0..MAX_JACOBI_SWEEPS {
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
        if max_val < EPS * 1e4 {
            break;
        }

        let s_pq = s[p * n + q];
        let diff = s[q * n + q] - s[p * n + p];
        let theta = if diff.abs() < EPS {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * s_pq / diff).atan()
        };
        let (sn, c) = theta.sin_cos();

        jacobi_rotate(&mut s, n, p, q, c, sn);
        apply_givens_rotation_from_right(&mut v, n, p, q, c, sn);
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| s[i * n + i]).collect();
    (eigenvalues, v)
}

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

fn gram_schmidt(cols: &mut Vec<Vec<f64>>) {
    let ncols = cols.len();
    for i in 0..ncols {
        for j in 0..i {
            let dot: f64 = cols[i].iter().zip(cols[j].iter()).map(|(&a, &b)| a * b).sum();
            let cj = cols[j].clone();
            for (a, b) in cols[i].iter_mut().zip(cj.iter()) {
                *a -= dot * b;
            }
        }

        let norm = cols[i].iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > EPS {
            for x in cols[i].iter_mut() {
                *x /= norm;
            }
        } else {
            let dim = cols[i].len();
            'search: for k in 0..dim {
                let mut e = vec![0.0f64; dim];
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

fn multiply_A_by_vector(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
}

pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert!(m > 0 && n > 0, "Matrix must have at least 1 row and at least 1 column.");
    assert_eq!(a.len(), m * n, "Data length of matrix must match the product of the specified number of rows and number of columns.", a.len(), m, n);
    assert!(a.iter().all(|x| x.is_finite()), "Matrix must not contain null or infinite entries.");

    let ata = compute_ata(a, m, n);
    let (eigenvalues, v) = jacobi_eigen(&ata, n);
    
    let k = m.min(n);

    let mut order: Vec<usize> = (0..n).collect();
    for i in 1..n {
        let mut j = i;
        while j > 0 && eigenvalues[order[j - 1]] < eigenvalues[order[j]] {
            order.swap(j - 1, j);
            j -= 1;
        }
    }

    let mut sigma: Vec<f64> = order[..k]
        .iter()
        .map(|&i| if eigenvalues[i] > 0.0 { eigenvalues[i].sqrt() } else { 0.0 })
        .collect();

    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(m);
    for index in 0..k {
        let ei = order[index];
        let vi: Vec<f64> = (0..n).map(|r| v[r * n + ei]).collect();
        let av = multiply_A_by_vector(a, &vi, m, n);
        if sigma[index] > EPS * 10.0 {
            u_cols.push(av.iter().map(|&x| x / sigma[index]).collect());
        } else {
            sigma[index] = 0.0;
            u_cols.push(vec![0.0; m]);
        }
    }

    for _ in k..m {
        u_cols.push(vec![0.0; m]);
    }
    gram_schmidt(&mut u_cols);

    let mut u = vec![0f64; m * m];
    for (ci, col) in u_cols.iter().enumerate() {
        for (ri, &value) in col.iter().enumerate() {
            u[ri * m + ci] = value;
        }
    }

    let mut vt = vec![0f64; n * n];
    for new_row in 0..n {
        let old_col = order[new_row];
        for c in 0..n {
            vt[new_row * n + c] = v[c * n + old_col];
        }
    }

    (u, sigma, vt)
}