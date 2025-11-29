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
        ("programs/test_assert", Ok(interpreter::Value::Void)),
        ("programs/test_for", Ok(interpreter::Value::Void)),
        ("programs/test_expr_return", Ok(interpreter::Value::Int(1))),
        ("programs/test_slice_list", Ok(interpreter::Value::Void)),
        ("programs/test_slice_string", Ok(interpreter::Value::Void)),
        ("programs/test_slice_tuple", Ok(interpreter::Value::Void)),
        ("programs/test_lambda", Ok(interpreter::Value::Void)),
        ("programs/test_lambda2", Ok(interpreter::Value::Void)),
        ("programs/test_listcomp", Ok(interpreter::Value::Void)),
        ("programs/test_listcomp2", Ok(interpreter::Value::Void)),
        ("programs/test_loop", Ok(interpreter::Value::Int(36))),
        ("programs/test_sqrt", Ok(interpreter::Value::Void)),
        ("programs/aoc_2023_1", Ok(interpreter::Value::Tuple(vec![interpreter::Value::Int(55123), interpreter::Value::Int(55260)]))),
        ("programs/aoc_2023_2", Ok(interpreter::Value::Tuple(vec![interpreter::Value::Int(2449), interpreter::Value::Int(63981)]))),
        ("programs/aoc_2023_3", Ok(interpreter::Value::Tuple(vec![interpreter::Value::Int(528819), interpreter::Value::Int(80403602)]))),
        ("programs/aoc_2023_4", Ok(interpreter::Value::Tuple(vec![interpreter::Value::Int(26426), interpreter::Value::Int(6227972)]))),
        ("programs/aoc_2023_5", Ok(interpreter::Value::Tuple(vec![interpreter::Value::Int(1181555926), interpreter::Value::Int(37806486)]))),
        ("programs/aoc_2023_6", Ok(interpreter::Value::Tuple(vec![interpreter::Value::Int(741000), interpreter::Value::Int(38220708)]))),
    ];

    for (path, res) in programs {
        assert!(eval(path.to_string()) == res, "{}", path.to_string());
    }
}

#[test]
fn tests_fail() {
    let programs: Vec<&str> = vec![
        ("programs/test_assert2"),
    ];

    for path in programs {
        assert!(matches!(eval(path.to_string()), Err(_)));
    }
}

