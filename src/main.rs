use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub parser); // synthesized by LALRPOP

mod parse_ast;
mod ast;
mod interpreter;
mod runner;





pub fn main() {
    runner::run("programs/test_for".to_string());
}