use std::{fs, env};
use waterbird::FuzzData;

fn main() {
    let path = env::args().nth(1).expect("file");
    let data = fs::read(path).unwrap();

    let input: FuzzData = bincode::deserialize(&data).unwrap();
    println!("{}", input.to_ast());
    println!("{:?}", input.to_ast().typecheck());
}
