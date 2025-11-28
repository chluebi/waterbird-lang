use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub parser); // synthesized by LALRPOP

mod parse_ast;
mod ast;
mod interpreter;
mod runner;

use std::env;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    
    let profile = args.contains(&String::from("--profile"));
    
    let args_cleaned: Vec<String> = args.iter()
        .filter(|arg| !arg.starts_with("--"))
        .cloned()
        .collect();

    if args_cleaned.len() < 2 {
        eprintln!("Usage: {} <filename> [--profile]", args[0]);
        std::process::exit(1);
    }
    
    let filename = args_cleaned[1].clone(); 

    runner::run(filename, profile);
}