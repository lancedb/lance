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

fn multiply_a_by_vector(a: &[f64], x: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
}

pub fn svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if a.len() == 0 || m == 0 || n == 0 {
        println!("Error: Matrix must have at least 1 row and at least 1 column.");
    }
    if a.len() != m * n {
        println!("Error: Data length of matrix must match the product of the specified number of rows and number of columns.");
    }
    if !a.iter().all(|x| x.is_finite()) {
        println!("Error: Matrix must not contain null or infinite entries.");
    }

    if m == 0 || n == 0 || a.len() != m * n || !a.iter().all(|x| x.is_finite()) {
        return (vec![], vec![], vec![]);
    }

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
        let av = multiply_a_by_vector(a, &vi, m, n);
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

fn main() {
    println!("========== UNIT TESTS ==========");
    unit_test_1_multiply_a_by_vector();
    unit_test_2_compute_ata();
    unit_test_3_create_identity_matrix();

    println!("\n========== INTEGRATION TESTS ==========");
    integration_test_1_svd_2x3();
    integration_test_2_svd_4x4();

    println!("\n========== MANUAL TESTS ==========");
    manual_test_1_svd_3x3();
    println!();
    manual_test_2_svd_empty_matrix();
    println!();
    manual_test_3_svd_length_mismatch();
    println!();
    manual_test_4_svd_nan_entry();
    println!();
    manual_test_5_svd_infinite_entry();
    println!();
    manual_test_6_jacobi_rotate();
    println!();
    manual_test_7_apply_givens_rotation_from_right();
    println!();
    manual_test_8_jacobi_eigen();
    println!();
    manual_test_9_gram_schmidt();

    println!("\nAll tests complete.");
}

// =========================================================================
// UNIT TESTS — automated
// =========================================================================

fn unit_test_1_multiply_a_by_vector() {
    let a = vec![5.0, 7.0, -1.0, 4.0];
    let x = vec![-9.0, -2.0];
    let m = 2;
    let n = 2;
    let result = multiply_a_by_vector(&a, &x, m, n);
    assert_eq!(result, vec![-59.0, 1.0],
        "unit_test_1 FAILED: got {:?}, expected [-59.0, 1.0]", result);
    println!("unit_test_1 PASSED");
}

fn unit_test_2_compute_ata() {
    let a = vec![3.0, 2.0, -4.0, 7.0, -2.0, -1.0, 5.0, 4.0];
    let m = 2;
    let n = 4;
    let result = compute_ata(&a, m, n);
    let expected = vec![
        13.0,   8.0, -22.0,  13.0,
         8.0,   5.0, -13.0,  10.0,
       -22.0, -13.0,  41.0,  -8.0,
        13.0,  10.0,  -8.0,  65.0,
    ];
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-9,
            "unit_test_2 FAILED at index {}: got {}, expected {}", i, r, e
        );
    }
    println!("unit_test_2 PASSED");
}

fn unit_test_3_create_identity_matrix() {
    let n = 5;
    let result = create_identity_matrix(n);
    let expected = vec![
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    assert_eq!(result, expected,
        "unit_test_3 FAILED: got {:?}", result);
    println!("unit_test_3 PASSED");
}

// =========================================================================
// INTEGRATION TESTS — automated
// =========================================================================

fn integration_test_1_svd_2x3() {
    let a = vec![3.0, 2.0, 2.0, 2.0, 3.0, -2.0];
    let m = 2;
    let n = 3;
    let (u, sigma, vt) = svd(&a, m, n);

    assert!(!sigma.is_empty(), "integration_test_1 FAILED: sigma is empty");
    assert!(
        (sigma[0] - 5.0).abs() < 1e-3,
        "integration_test_1 FAILED: sigma[0] = {}, expected 5.0", sigma[0]
    );
    assert!(
        (sigma[1] - 3.0).abs() < 1e-3,
        "integration_test_1 FAILED: sigma[1] = {}, expected 3.0", sigma[1]
    );

    // : Vec<f64> added — this is the only change, asserts are unchanged
    let expected_u: Vec<f64> = vec![0.7071, 0.7071, 0.7071, -0.7071];
    let expected_vt: Vec<f64> = vec![
         0.7071,  0.7071,  0.0000,
         0.2357, -0.2357,  0.9428,
         0.6667, -0.6667, -0.3333,
    ];
    for (i, (&r, &e)) in u.iter().zip(expected_u.iter()).enumerate() {
        assert!(
            (r.abs() - e.abs()).abs() < 1e-3,
            "integration_test_1 FAILED U at index {}: got {}, expected {}", i, r, e
        );
    }
    for (i, (&r, &e)) in vt.iter().zip(expected_vt.iter()).enumerate() {
        assert!(
            (r.abs() - e.abs()).abs() < 1e-3,
            "integration_test_1 FAILED V^T at index {}: got {}, expected {}", i, r, e
        );
    }
    println!("integration_test_1 PASSED: U = {:?}, sigma = {:?}, V^T = {:?}", u, sigma, vt);
}

fn integration_test_2_svd_4x4() {
    let a = vec![
         7.0, -4.0,  5.0,  5.0,
         8.0, -2.0,-10.0,  1.0,
        -1.0, -8.0,  9.0,  3.0,
         8.0,  7.0, -3.0,  4.0,
    ];
    let m = 4;
    let n = 4;
    let (u, sigma, vt) = svd(&a, m, n);

    assert!(!sigma.is_empty(), "integration_test_2 FAILED: sigma is empty");

    // : Vec<f64> added — asserts unchanged
    let expected_sigma: Vec<f64> = vec![17.834861, 13.682368, 8.433152, 0.769723];
    for (i, (&r, &e)) in sigma.iter().zip(expected_sigma.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-3,
            "integration_test_2 FAILED sigma at index {}: got {}, expected {}", i, r, e
        );
    }

    let expected_u: Vec<f64> = vec![
         0.1103,  0.7669,  0.1056, -0.6233,
        -0.5973,  0.3163, -0.7191,  0.1617,
         0.5994,  0.4553, -0.1537,  0.6402,
        -0.5214,  0.3233,  0.6695,  0.4189,
    ];
    let expected_vt: Vec<f64> = vec![
        -0.4921, -0.4312,  0.7560, -0.0187,
         0.7330, -0.3713,  0.2777,  0.4977,
         0.0588,  0.8219,  0.5131,  0.2402,
        -0.4659, -0.0249, -0.2968,  0.8332,
    ];
    for (i, (&r, &e)) in u.iter().zip(expected_u.iter()).enumerate() {
        assert!(
            (r.abs() - e.abs()).abs() < 1e-3,
            "integration_test_2 FAILED U at index {}: got {}, expected {}", i, r, e
        );
    }
    for (i, (&r, &e)) in vt.iter().zip(expected_vt.iter()).enumerate() {
        assert!(
            (r.abs() - e.abs()).abs() < 1e-3,
            "integration_test_2 FAILED V^T at index {}: got {}, expected {}", i, r, e
        );
    }
    println!("integration_test_1 PASSED: U = {:?}, sigma = {:?}, V^T = {:?}", u, sigma, vt);
}

// =========================================================================
// MANUAL TESTS
// =========================================================================

fn manual_test_1_svd_3x3() {
    let a = vec![-9.0, -5.0, -2.0, 4.0, -1.0, 6.0, 9.0, -2.0, -6.0];
    let m = 3;
    let n = 3;
    let (u, sigma, vt) = svd(&a, m, n);
    println!("manual_test_1 U:     {:?}", u);
    println!("manual_test_1 sigma: {:?}", sigma);
    println!("manual_test_1 V^T:    {:?}", vt);
    println!("Expected U ≈ [-0.6955, -0.4291, -0.5763, 0.2417, 0.6157, -0.7500, 0.6767, -0.6609, -0.3245], Expected sigma ≈ [13.5037, 8.9903, 4.5633], Expected V^T ≈ [0.9861, 0.1394, -0.0903, 0.0419, 0.3172, 0.9474, -0.1607, 0.9381, -0.3070]");
}

fn manual_test_2_svd_empty_matrix() {
    let a: Vec<f64> = vec![];
    let m = 0;
    let n = 0;
    let (u, sigma, vt) = svd(&a, m, n);
    println!("manual_test_2 U:     {:?}", u);
    println!("manual_test_2 sigma: {:?}", sigma);
    println!("manual_test_2 V^T:    {:?}", vt);
    println!("Expected: all empty vectors, error message: 'Error: Matrix must have at least 1 row and at least 1 column.'");
}

fn manual_test_3_svd_length_mismatch() {
    let a = vec![9.0, -8.0, 3.0, -1.0];
    let m = 1;
    let n = 2;
    let (u, sigma, vt) = svd(&a, m, n);
    println!("manual_test_3 sigma: {:?}", sigma);
    println!("manual_test_3 U:     {:?}", u);
    println!("manual_test_3 V^T:    {:?}", vt);
    println!("Expected: all empty vectors, error message: 'Error: Data length of matrix must match the product of the specified number of rows and number of columns.'");
}

fn manual_test_4_svd_nan_entry() {
    let a = vec![3.0, 1.0, f64::NAN, -4.0, -2.0, -1.0, 8.0, 3.0, 1.0];
    let m = 3;
    let n = 3;
    let (u, sigma, vt) = svd(&a, m, n);
    println!("manual_test_4 sigma: {:?}", sigma);
    println!("manual_test_4 U:     {:?}", u);
    println!("manual_test_4 V^T:    {:?}", vt);
    println!("Expected: all empty vectors, error message: 'Error: Matrix must not contain null or infinite entries.'");
}

fn manual_test_5_svd_infinite_entry() {
    let a = vec![9.0, f64::INFINITY, -1.0, 8.0];
    let m = 2;
    let n = 2;
    let (u, sigma, vt) = svd(&a, m, n);
    println!("manual_test_5 sigma: {:?}", sigma);
    println!("manual_test_5 U:     {:?}", u);
    println!("manual_test_5 V^T:    {:?}", vt);
    println!("Expected: all empty vectors, error message: 'Error: Matrix must not contain null or infinite entries.'");
}

fn manual_test_6_jacobi_rotate() {
    let mut a = vec![4.0, 2.0, 2.0, 3.0];
    let p = 0;
    let q = 1;
    let s_pq: f64 = a[p * 2 + q];
    let diff: f64 = a[q * 2 + q] - a[p * 2 + p];
    let theta: f64 = 0.5 * (2.0 * s_pq / diff).atan();
    let (sn, c) = theta.sin_cos();
    jacobi_rotate(&mut a, 2, p, q, c, sn);
    println!("manual_test_6 after jacobi_rotate: {:?}", a);
    println!("Off-diagonal a[0,1] = {:.6} (expected ~0.0)", a[p * 2 + q]);
    println!("Trace = {:.6} (expected 7.0)", a[0] + a[3]);
}

fn manual_test_7_apply_givens_rotation_from_right() {
    let mut b = vec![1.0, 0.0, 0.0, 1.0];
    let angle: f64 = std::f64::consts::FRAC_PI_4;
    let c = angle.cos();
    let s = angle.sin();
    apply_givens_rotation_from_right(&mut b, 2, 0, 1, c, s);
    println!("manual_test_7 after rotation: {:?}", b);
    println!("Expected: [0.7071, -0.7071, 0.7071, 0.7071]");
}

fn manual_test_8_jacobi_eigen() {
    let a = vec![4.0, 1.0, 1.0, 3.0];
    let (eigenvalues, eigenvectors) = jacobi_eigen(&a, 2);
    println!("manual_test_8 eigenvalues:  {:?}", eigenvalues);
    println!("manual_test_8 eigenvectors: {:?}", eigenvectors);
    println!("Expected eigenvalues ≈ [4.618, 2.382] (any order)");
}

fn manual_test_9_gram_schmidt() {
    let mut cols = vec![
        vec![1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
    ];
    gram_schmidt(&mut cols);
    println!("manual_test_9 after gram_schmidt:");
    for (i, col) in cols.iter().enumerate() {
        let norm: f64 = col.iter().map(|&x| x * x).sum::<f64>().sqrt();
        println!("  col[{}] = {:?}  norm = {:.6}", i, col, norm);
    }
    let dot01: f64 = cols[0].iter().zip(cols[1].iter()).map(|(&a, &b)| a * b).sum();
    let dot02: f64 = cols[0].iter().zip(cols[2].iter()).map(|(&a, &b)| a * b).sum();
    let dot12: f64 = cols[1].iter().zip(cols[2].iter()).map(|(&a, &b)| a * b).sum();
    println!("dot(col0,col1) = {:.6} (expected ~0.0)", dot01);
    println!("dot(col0,col2) = {:.6} (expected ~0.0)", dot02);
    println!("dot(col1,col2) = {:.6} (expected ~0.0)", dot12);
}