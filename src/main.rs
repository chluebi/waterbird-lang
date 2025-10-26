mod ast;
mod interpreter;

use ast::{Expr, LocExpr, LocStmt, Stmt, BinOp, UnOp};
use interpreter::{InterpreterError, Value};
use std::collections::HashMap;

fn loc_expr(expr: Expr) -> LocExpr {
    LocExpr { expr, loc: 0..0 }
}

fn loc_stmt(stmt: Stmt) -> LocStmt {
    LocStmt { stmt, loc: 0..0 }
}

fn build_test_ast() -> LocStmt {
    // {
    //   var x = 10;
    //   var y = 20;
    //   var z = x + (y + -5); // z = 10 + (20 - 5) = 25
    //   var my_list = [1, 2, 3];
    //   my_list[1] = z; // my_list = [1, 25, 3]
    //   var my_dict = {"a": 100, "b": 200};
    //   my_dict["a"] = my_list[1]; // my_dict = {"a": 25, "b": 200}
    //   var my_tuple = (my_dict["a"], "hello", 99); // my_tuple = (25, "hello", 99)
    //   return my_tuple;
    // }

    let statements = vec![
        // var x = 10;
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("x".to_string())),
            expression: loc_expr(Expr::Int(10)),
        }),
        // var y = 20;
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("y".to_string())),
            expression: loc_expr(Expr::Int(20)),
        }),
        // var z = x + (y + -5);
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("z".to_string())),
            expression: loc_expr(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(loc_expr(Expr::Variable("x".to_string()))),
                right: Box::new(loc_expr(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(loc_expr(Expr::Variable("y".to_string()))),
                    right: Box::new(loc_expr(Expr::Unop {
                        op: UnOp::Neg,
                        expr: Box::new(loc_expr(Expr::Int(5))),
                    })),
                })),
            }),
        }),
        // var my_list = [1, 2, 3];
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("my_list".to_string())),
            expression: loc_expr(Expr::List(vec![
                loc_expr(Expr::Int(1)),
                loc_expr(Expr::Int(2)),
                loc_expr(Expr::Int(3)),
            ])),
        }),
        // my_list[1] = z;
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Indexing {
                indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                indexer: Box::new(loc_expr(Expr::Int(1))),
            }),
            expression: loc_expr(Expr::Variable("z".to_string())),
        }),
        // var my_dict = {"a": 100, "b": 200};
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("my_dict".to_string())),
            expression: loc_expr(Expr::Dictionary(vec![
                (
                    loc_expr(Expr::Str("a".to_string())),
                    loc_expr(Expr::Int(100)),
                ),
                (
                    loc_expr(Expr::Str("b".to_string())),
                    loc_expr(Expr::Int(200)),
                ),
            ])),
        }),
        // my_dict["a"] = my_list[1];
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Indexing {
                indexed: Box::new(loc_expr(Expr::Variable("my_dict".to_string()))),
                indexer: Box::new(loc_expr(Expr::Str("a".to_string()))),
            }),
            expression: loc_expr(Expr::Indexing {
                indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                indexer: Box::new(loc_expr(Expr::Int(1))),
            }),
        }),
        // var my_tuple = (my_dict["a"], "hello", 99);
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("my_tuple".to_string())),
            expression: loc_expr(Expr::Tuple(vec![
                loc_expr(Expr::Indexing {
                    indexed: Box::new(loc_expr(Expr::Variable("my_dict".to_string()))),
                    indexer: Box::new(loc_expr(Expr::Str("a".to_string()))),
                }),
                loc_expr(Expr::Str("hello".to_string())),
                loc_expr(Expr::Int(99)),
            ])),
        }),
        // return my_tuple;
        loc_stmt(Stmt::Return {
            expression: loc_expr(Expr::Variable("my_tuple".to_string())),
        }),
    ];

    loc_stmt(Stmt::Block { statements })
}


fn print_value(value: &Value, heap: &interpreter::Heap) {
    match value {
        Value::Int(i) => print!("{}", i),
        Value::Bool(b) => print!("{}", b),
        Value::Str(ptr) => {
            if let Some(interpreter::HeapObject::Str(s)) = heap.get(*ptr) {
                print!("\"{}\"", s);
            } else {
                print!("[Invalid Str Ptr: {}]", ptr);
            }
        }
        Value::Tuple(values) => {
            print!("(");
            for (i, v) in values.iter().enumerate() {
                print_value(v, heap);
                if i < values.len() - 1 {
                    print!(", ");
                }
            }
            print!(")");
        }
        Value::List(ptr) => {
             if let Some(interpreter::HeapObject::List(l)) = heap.get(*ptr) {
                print!("[");
                for (i, v) in l.iter().enumerate() {
                    print_value(v, heap);
                    if i < l.len() - 1 {
                        print!(", ");
                    }
                }
                print!("]");
            } else {
                print!("[Invalid List Ptr: {}]", ptr);
            }
        }
        Value::Dictionary(ptr) => {
            if let Some(interpreter::HeapObject::Dictionary(d)) = heap.get(*ptr) {
                print!("{{");
                let mut first = true;
                for (k, v) in d.iter() {
                    if !first {
                        print!(", ");
                    }
                    print_value(k, heap);
                    print!(": ");
                    print_value(v, heap);
                    first = false;
                }
                print!("}}");
            } else {
                print!("[Invalid Dict Ptr: {}]", ptr);
            }
        }
        Value::Lambda(ptr) => print!("[Lambda Ptr: {}]", ptr),
    }
}

fn main() {
    println!("--- Interpreter Test ---");

    let mut state = interpreter::State::new();

    let program = ast::Program {
        functions: HashMap::new(),
    };

    let test_block = build_test_ast();

    match interpreter::run_statement(&mut state, &test_block, &program) {
        Ok(Some(return_value)) => {
            println!("\nTest block finished successfully.");
            print!("Return Value: ");
            print_value(&return_value, &state.heap);
            println!();
        }
        Ok(_) => {
            println!("\nTest block finished with no return value.");
        }
        Err(e) => {
            println!("\n--- INTERPRETER ERROR ---");
            match e.error {
                InterpreterError::Panic(msg) => {
                    println!("Panic: {}", msg);
                }
            }
            if let Some(loc) = e.loc {
                println!("Location: {:?}", loc);
            }
            println!("---------------------------");
        }
    }

    println!("\n--- Final Heap State ---");
    for (ptr, obj) in &state.heap.objects {
        print!("[{}]: ", ptr);
        match obj {
            interpreter::HeapObject::Str(s) => println!("Str(\"{}\")", s),
            interpreter::HeapObject::List(l) => {
                print!("List([");
                 for (i, v) in l.iter().enumerate() {
                    print_value(v, &state.heap);
                    if i < l.len() - 1 {
                        print!(", ");
                    }
                }
                println!("])");
            },
            interpreter::HeapObject::Dictionary(d) => {
                print!("Dictionary({{");
                let mut first = true;
                for (k, v) in d.iter() {
                    if !first {
                        print!(", ");
                    }
                    print_value(k, &state.heap);
                    print!(": ");
                    print_value(v, &state.heap);
                    first = false;
                }
                println!("}})");
            }
            interpreter::HeapObject::Lambda { .. } => println!("Lambda(...)"),
        }
    }
    println!("------------------------");

}
