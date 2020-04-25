use std::thread;
use std::time::Instant;

const SIZE: usize = 60_000;
const NUM_THREADS: usize = 6;
const NUM_TRIES: usize = 1024;
const A: [f64; SIZE] = [1.0; SIZE];
const B: [f64; SIZE] = [2.0; SIZE];
fn main() {
    let mut max_megamults = 0.0;
    for _ in 0..NUM_TRIES {
        let mut c = [0.0; SIZE];
        let start = Instant::now();
        println!("before thread");
        let t_1 = thread::spawn(move || {
            println!("In thread");
            let low = 0;
            let high = 10_000;
            let slice = &mut c[low..high];
            for num in 0..high {
                slice[num] = A[low + num] * B[low + num];
            }
        });
        // let t_2 = thread::spawn(move || {
        //     let low = 10_001;
        //     let high = 20_000;
        //     let slice = &mut c[low..high];
        //     for num in 0..slice.len() {
        //         slice[num] = A[low + num] * B[low + num];
        //     }
        // });
        // let t_3 = thread::spawn(move || {
        //     let low = 20_001;
        //     let high = 30_000;
        //     let slice = &mut c[low..high];
        //     for num in 0..slice.len() {
        //         slice[num] = A[low + num] * B[low + num];
        //     }
        // });
        // let t_4 = thread::spawn(move || {
        //     let low = 30_001;
        //     let high = 40_000;
        //     let slice = &mut c[low..high];
        //     for num in 0..slice.len() {
        //         slice[num] = A[low + num] * B[low + num];
        //     }
        // });
        // let t_5 = thread::spawn(move || {
        //     let low = 40_001;
        //     let high = 50_000;
        //     let slice = &mut c[low..high];
        //     for num in 0..slice.len() {
        //         slice[num] = A[low + num] * B[low + num];
        //     }
        // });
        // let t_6 = thread::spawn(move || {
        //     let low = 50_001;
        //     let high = 60_000;
        //     let slice = &mut c[low..high];
        //     for num in 0..slice.len() {
        //         slice[num] = A[low + num] * B[low + num];
        //     }
        // });

        t_1.join().unwrap();
        // t_2.join().unwrap();
        // t_3.join().unwrap();
        // t_4.join().unwrap();
        // t_5.join().unwrap();
        // t_6.join().unwrap();

        let end = Instant::now();
        let mults_per_nanosecond = SIZE as f64 / (end - start).as_nanos() as f64;
        let mults_per_second = mults_per_nanosecond * 1_000_000_000.0;
        let megamults_per_second = mults_per_second / 1_000_000.0;
        if megamults_per_second > max_megamults {
            max_megamults = megamults_per_second;
        }
    }

    println!("Peak Performance = {}", max_megamults);
}
