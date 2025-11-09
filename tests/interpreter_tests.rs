extern crate waterbird_lang;

// 2. Use the items from your library
use waterbird_lang::runner::eval;
use waterbird_lang::interpreter;  



#[test]
fn tests_basic() {
    let programs: Vec<(&str, Result<interpreter::Value, interpreter::InterpreterErrorMessage>)> = vec![
        ("programs/test_bool", Ok(interpreter::Value::Bool(true))),
        ("programs/test_bool2", Ok(interpreter::Value::Bool(false))),
        ("programs/test_int", Ok(interpreter::Value::Int(2))),
        ("programs/test_int2", Ok(interpreter::Value::Int(13))),
        ("programs/test_if", Ok(interpreter::Value::Bool(true))),
        ("programs/test_if2", Ok(interpreter::Value::Bool(true))),
        ("programs/test_list", Ok(interpreter::Value::Int(3))),
        ("programs/test_list2", Ok(interpreter::Value::Int(5050))),
        ("programs/test_list3", Ok(interpreter::Value::Bool(true))),
        ("programs/test_list4", Ok(interpreter::Value::Bool(true))),
        ("programs/test_blocks", Ok(interpreter::Value::Int(4))),
        ("programs/test_precedence", Ok(interpreter::Value::Bool(true))),
        ("programs/test_shortcircuit", Ok(interpreter::Value::Int(3))),
        ("programs/test_readfile", Ok(interpreter::Value::Int(10))),
    ];

    for (path, res) in programs {
        assert!(eval(path.to_string()) == res);
    }
}


