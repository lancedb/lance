use std::cmp::Ordering;

use arrow_schema::ArrowError;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};

/// Partition the `arr` for the first `topk` elements are the smallest
pub fn find_topk<T: Send, F>(values: &mut [T], indices: &mut [u64], topk: usize, compare: &F) -> Result<(), ArrowError>
    where F: Fn(&T, &T) -> Ordering + Sync {
    let mut rng = rand::rngs::SmallRng::from_entropy();
    assert!(topk > 0);
    if values.len() <= topk {
        return Ok(());
    }
    let mut low = 0;  // inclusive
    let mut high = values.len();  // exclusive
    // TODO handle worse case scenarios and fallback to nlog(n) sort
    loop {
        let pivot: usize = rng.gen_range(low..high);  // choose random pivot
        let left = partition(values, indices, low, high, pivot, compare)?;
        if left == topk {
            return Ok(());
        } else if left > topk {
            high = left;
        } else if left < topk {
            low = left;
        }
    }
}

/// Returns the number of elements that are smaller than the pivot value
pub(crate) fn partition<T: Send, F>(arr: &mut [T], indices: &mut [u64], low: usize, high: usize, pivot: usize, compare: &F) -> Result<usize, ArrowError>
    where F: Fn(&T, &T) -> Ordering, {
    let is_less = |l: &T, r: &T| compare(l, r) == Ordering::Less;
    if !(low <= pivot && pivot < high) {
        return Err(ArrowError::ComputeError("Pivot must be between low and high".to_string()));
    }
    arr.swap(high - 1, pivot);
    indices.swap(high - 1, pivot);
    let pivot = high - 1;
    if pivot == 0 {
        return Ok(0);
    }
    println!("Pivot is {}", pivot);

    // sweep from both ends til the two cursors meet
    let mut left_cursor = low;
    let mut right_cursor = pivot - 1;

    loop {
        // sweep from the left
        while is_less(&arr[left_cursor], &arr[pivot]) {
            left_cursor += 1;
        }

        // sweep from the right
        while right_cursor > left_cursor && is_less(&arr[pivot], &arr[right_cursor]) {
            right_cursor -= 1;
        }

        if left_cursor >= right_cursor {
            // if they meet then we're done
            break;
        } else {
            // otherwise we swap elements to the right partition and continue
            println!("Swapping {} with {}", left_cursor, right_cursor);
            arr.swap(left_cursor, right_cursor);
            indices.swap(left_cursor, right_cursor);
            left_cursor += 1;
            right_cursor -= 1;
        }
    }
    println!("Swapping {} with {}", left_cursor, pivot);
    arr.swap(left_cursor, pivot);
    indices.swap(left_cursor, pivot);
    Ok(left_cursor)
}


#[cfg(test)]
mod tests {
    use arrow_array::ArrowNativeTypeOp;
    use rand::{Rng, SeedableRng};
    use crate::index::ann::sort::{find_topk, partition};

    #[test]
    fn test_partition() {
        let compare = |a: &i32, b: &i32| a.compare(*b);
        let mut arr = vec![4,9,2,5,3,9,7,1];
        let mut indices: Vec<u64>= (0..arr.len() as u64).collect();
        let pivot = 3;
        let pivot_val = arr[pivot];
        let len = arr.len();
        let left = partition(&mut arr, &mut indices, 0, len, pivot, &compare).unwrap();
        assert_eq!(4, left);
        for v in arr[0..left].iter() {
            assert!(*v < pivot_val)
        }
        let mut indices: Vec<u64> = (0..arr.len() as u64).collect();

        arr = vec![1];
        assert_eq!(0, partition(&mut arr, &mut indices, 0, 1, 0, &compare).unwrap());

        arr = vec![1, 1, 1, 1, 1, 1, 1];
        let len = arr.len();
        assert_eq!(3, partition(&mut arr, &mut indices, 0, len, 4, &compare).unwrap());

        arr = vec![1, 2, 3, 4, 5, 6, 7];
        let len = arr.len();
        assert_eq!(4, partition(&mut arr, &mut indices, 0, len, 4, &compare).unwrap());

        arr = vec![7, 6, 5, 4, 3, 2, 1];
        let len = arr.len();
        assert_eq!(2, partition(&mut arr, &mut indices, 0, len, 4, &compare).unwrap());
    }

    #[test]
    fn test_find_topk() {
        let compare = |a: &f32, b: &f32| a.compare(*b);
        let mut rows: Vec<u64> = (0..100).collect();
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let mut scores:[f32; 100] = [0.0; 100];
        let mut sorted = scores.clone();
        sorted.sort_by(compare);

        rng.fill(&mut scores[..]);
        let topk = 10;
        let (top_scores, top_indices) = find_topk(&mut scores, &mut rows, topk, &compare).unwrap();

        for i in 0..topk {
            assert!(top_scores[i] <= sorted[topk])
        }
    }
}