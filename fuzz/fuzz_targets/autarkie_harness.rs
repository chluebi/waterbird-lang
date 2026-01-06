#![no_main]
use libfuzzer_sys::fuzz_target;
use waterbird::fuzz_ast;

fuzz_target!(|data: &[u8]| {
    let Ok(program) = bincode::deserialize::<fuzz_ast::FuzzData>(data) else {
        return;
    };

    let _ = program.to_ast().typecheck().unwrap();
});
