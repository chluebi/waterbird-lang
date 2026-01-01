use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub parser); // synthesized by LALRPOP

use waterbird::runner;
use waterbird::parse_ast;

use std::env;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    
    let args_cleaned: Vec<String> = args.iter()
        .filter(|arg| !arg.starts_with("--"))
        .cloned()
        .collect();

    if args_cleaned.len() < 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }
    
    let filename = args_cleaned[1].clone(); 

    runner::typecheck(filename);
}