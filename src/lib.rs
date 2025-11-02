use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub parser); // synthesized by LALRPOP

mod parse_ast;
mod ast;
mod interpreter;
mod runner;