const EPS: f64 = 2.220_446_049_250_313e-16;
const MAX_JACOBI_SWEEPS: usize = 200;

/// x² — just multiplication, listed for clarity
#[inline]
fn sq(x: f64) -> f64 { x * x }

/// Raise f64 to an integer power.
fn powi(mut x: f64, mut n: i32) -> f64 {
    if n == 0 { return 1.0; }
    if n < 0  { x = 1.0 / x; n = -n; }
    let mut result = 1.0f64;
    let mut base = x;
    let mut exp = n as u32;
    while exp > 0 {
        if exp & 1 == 1 { result *= base; }
        base *= base;
        exp >>= 1;
    }
    result
}

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

fn jacobi_rotate(s: &mut [f64], n: usize, p: usize, q: usize, c: f64, sn: f64) {
    let s_pp = s[p * n + p];
    let s_qq = s[q * n + q];
    let s_pq = s[p * n + q];

    s[p * n + p] = sq(c) * s_pp - 2.0 * sn * c * s_pq + sq(sn) * s_qq;
    s[q * n + q] = sq(sn) * s_pp + 2.0 * sn * c * s_pq + sq(c)  * s_qq;
    s[p * n + q] = 0.0;
    s[q * n + p] = 0.0;

    for r in 0..n {
        if r == p || r == q { continue; }
        let s_rp = s[r * n + p];
        let s_rq = s[r * n + q];
        let new_rp =  c * s_rp - sn * s_rq;
        let new_rq = sn * s_rp +  c * s_rq;
        s[r * n + p] = new_rp; s[p * n + r] = new_rp;
        s[r * n + q] = new_rq; s[q * n + r] = new_rq;
    }
}

fn apply_givens_right(v: &mut [f64], n: usize, p: usize, q: usize, c: f64, sn: f64) {
    for r in 0..n {
        let vp = v[r * n + p];
        let vq = v[r * n + q];
        v[r * n + p] =  c * vp - sn * vq;
        v[r * n + q] = sn * vp +  c * vq;
    }
}

// ============================================================
// Modified Gram-Schmidt orthonormalization
// ============================================================

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
        let norm = (cols[i].iter().map(|&x| sq(x)).sum::<f64>()).sqrt();
        if norm > EPS {
            for x in cols[i].iter_mut() { *x /= norm; }
        } else {
            // Replace with an orthonormal basis vector not already spanned
            let dim = cols[i].len();
            'search: for k in 0..dim {
                let mut e = vec![0.0f64; dim];
                e[k] = 1.0;
                for j in 0..i {
                    let dot: f64 = e.iter().zip(cols[j].iter()).map(|(&a, &b)| a * b).sum();
                    let cj = cols[j].clone();
                    for (a, b) in e.iter_mut().zip(cj.iter()) { *a -= dot * b; }
                }
                let n2 = (e.iter().map(|&x| sq(x)).sum::<f64>()).sqrt();
                if n2 > EPS {
                    for x in e.iter_mut() { *x /= n2; }
                    cols[i] = e;
                    break 'search;
                }
            }
        }
    }
}

// ============================================================
// Matrix helpers
// ============================================================

/// C = Aᵀ·A  where A is m×n → C is n×n
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

/// y = A·x  (A is m×n row-major, x length n → y length m)
fn mat_vec_mul(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
}

fn eye(n: usize) -> Vec<f64> {
    let mut m = vec![0f64; n * n];
    for i in 0..n { m[i * n + i] = 1.0; }
    m
}

pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ata = mat_mul_atb(a, m, n);
    let (eigenvalues, v) = jacobi_eigen(&ata, n);

    let k = m.min(n);
    let mut sigma: Vec<f64> = eigenvalues
        .iter()
        .map(|&l| (if l > 0.0 { l } else { 0.0 }).sqrt())
        .collect();

    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(m);
    for i in 0..k {
        let vi: Vec<f64> = (0..n).map(|r| v[r * n + i]).collect();
        let av = mat_vec_mul(a, &vi, m, n);
        if sigma[i] > EPS * 10.0 {
            u_cols.push(av.iter().map(|&x| x / sigma[i]).collect());
        } else {
            sigma[i] = 0.0;
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
            u[ri * m + ci] = value; // was: ri * m — m was undefined
        }
    }

    let mut order: Vec<usize> = (0..k).collect();
    for i in 1..k {
        let mut j = i;
        while j > 0 && sigma[order[j - 1]] < sigma[order[j]] {
            order.swap(j - 1, j);
            j -= 1;
        }
    }

    let sigma_sorted: Vec<f64> = order.iter().map(|&i| sigma[i]).collect(); // was: .map[(

    let mut u_sorted = vec![0f64; m * m];
    for (new_col, &old_col) in order.iter().enumerate() {
        for r in 0..m {
            u_sorted[r * m + new_col] = u[r * m + old_col]; // was: r * m (m undefined)
        }
    }
    for col in k..m {
        for r in 0..m {
            u_sorted[r * m + col] = u[r * m + col]; // was: k..m (m undefined)
        }
    }

    let mut vt = vec![0f64; n * n];
    for (new_row, &old_col) in order.iter().enumerate() {
        for c in 0..n {
            vt[new_row * n + c] = v[c * n + old_col]; // was: V_transpose, n undefined
        }
    }
    for row in k..n {
        for c in 0..n {
            vt[row * n + c] = v[c * n + row]; // was: n undefined
        }
    }

    (u_sorted, sigma_sorted, vt)
}

fn main() {
    let a = vec![3.0, 2.0, 2.0, 2.0, 3.0, -2.0];
    let (u, s, vt) = svd(&a, 2, 3);
    println!("{:?}", s);
}