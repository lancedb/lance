const EPS: f64 = 2.220_446_049_250_313e-16;
const MAX_JACOBI_SWEEPS: usize = 200;

fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = a.to_vec();
    let mut v = eye(n);

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
        apply_givens_right(&mut v, n, p, q, c, sn);
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| s[i * n + i]).collect();
    (eigenvalues, v)
}

fn mat_mul_atb(a: &[f64], m: usize, n: usize) -> Vec<f64> {
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
    let n_cols = cols.len();
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

fn mat_vec_mul(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect();
}

pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ata = mat_mul_atb(a, m, n);
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
        let av = mat_vec_mul(a, &vi, m, n);
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
}