extern crate whisky;
extern crate ndarray;

use whisky::*;
use ndarray::{Array1, Array2, ArrayView1};

struct Alloc {
    self_alloc: Vec<i32>, // 2^k * Basesize
    children: Vec<Alloc>,
}

fn collaps(a: &Alloc) -> Vec<i32> {
    let mut children: Vec<Vec<_>> = a.children.iter().map(|c| collaps(c)).collect();

    let mut alloc = a.self_alloc.clone();

    while let Some(max) = children.iter().filter_map(|c| c.last()).cloned().max() {
        alloc.push(max);

        for c in &mut children {
            // look how many components we can fit in the alloc
            let mut size = max;
            let mut count = 1;
            while count > 0 {
                if let Some(last) = c.last() {
                    while size > *last {
                        size -= 1;
                        count *= 2;
                    }
                }
                count -= 1;
                c.pop();
            }
        }
    }
    alloc.sort();
    alloc
}

pub fn main() {

    for i in 0..64 {
        //println!("{:?}", align_up(i));
    }

    let a = Alloc {
        self_alloc: vec![2,1,2],
        children: vec![
            Alloc {
                self_alloc: vec![1,3,1],
                children: vec![],
            },
            Alloc {
                self_alloc: vec![1,2,1,2,2],
                children: vec![],
            },
        ],
    };
    println!("{:?}", collaps(&a));
}
