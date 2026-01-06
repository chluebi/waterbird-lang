use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub parser); // synthesized by LALRPOP

pub mod parse_ast;
pub mod ast;
pub mod interpreter;
pub mod runner;
pub mod typechecker;

pub mod fuzz_ast;