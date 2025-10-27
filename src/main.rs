mod ast;
mod interpreter;

// Updated imports to include AST nodes for functions, calls, and control flow
use ast::{
    Expr, LocExpr, LocStmt, Stmt, BinOp, UnOp, CallArgument, LambdaArgument,
    Function, FunctionPrototype, Argument, TypeLiteral, LocTypeLiteral
};
use interpreter::{InterpreterError, Value};
use std::collections::HashMap;

fn loc_expr(expr: Expr) -> LocExpr {
    LocExpr { expr, loc: 0..0 }
}

fn loc_stmt(stmt: Stmt) -> LocStmt {
    LocStmt { stmt, loc: 0..0 }
}

fn build_test_ast() -> LocStmt {
    // This AST represents the following pseudo-code:
    // {
    //   // 1. Lambda Definition
    //   var my_add = fn(a, b) { a + b };
    //
    //   // 2. Lambda Call
    //   var z = my_add(10, 15); // z = 25
    //
    //   // 3. If/Else Control Flow
    //   var result_str = "";
    //   if (true) { // Using `true` since interpreter lacks comparison ops
    //     result_str = "big";
    //   } else {
    //     result_str = "small";
    //   } // result_str = "big"
    //
    //   // 4. While Control Flow
    //   var run_loop = true;
    //   var my_list = [100];
    //   while (run_loop) {
    //     my_list[0] = my_list[0] + z; // my_list[0] = 100 + 25 = 125
    //     run_loop = false; // Manually stop loop
    //   } // my_list = [125]
    //
    //   // 5. External Function Call (to 'test_func' defined in main)
    //   var func_result = test_func(my_list[0]); // func_result = test_func(125) = 125 + 125 = 250
    //
    //   // 6. Return all computed values
    //   return (z, result_str, my_list[0], func_result); // (25, "big", 125, 250)
    // }

    let statements = vec![
        // var my_add = fn(a, b) { a + b };
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("my_add".to_string())),
            expression: loc_expr(Expr::Lambda {
                arguments: vec![
                    LambdaArgument { name: "a".to_string(), loc: 0..0 },
                    LambdaArgument { name: "b".to_string(), loc: 0..0 },
                ],
                expr: Box::new(loc_expr(Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(loc_expr(Expr::Variable("a".to_string()))),
                    right: Box::new(loc_expr(Expr::Variable("b".to_string()))),
                })),
            }),
        }),

        // var z = my_add(10, 15);
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("z".to_string())),
            expression: loc_expr(Expr::FunctionCall {
                function: Box::new(loc_expr(Expr::Variable("my_add".to_string()))),
                positional_arguments: vec![
                    CallArgument { expression: Box::new(loc_expr(Expr::Int(10))), loc: 0..0 },
                    CallArgument { expression: Box::new(loc_expr(Expr::Int(15))), loc: 0..0 },
                ],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
            }),
        }), // z = 25

        // var result_str = "";
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("result_str".to_string())),
            expression: loc_expr(Expr::Str("".to_string())),
        }),

        // if (true) { ... } else { ... }
        loc_stmt(Stmt::IfElse {
            condition: loc_expr(Expr::Bool(true)), // Faking a condition like (z > 20)
            if_body: Box::new(loc_stmt(Stmt::Block {
                statements: vec![
                    loc_stmt(Stmt::Assignment {
                        target: loc_expr(Expr::Variable("result_str".to_string())),
                        expression: loc_expr(Expr::Str("big".to_string())),
                    }),
                ]
            })),
            else_body: Box::new(loc_stmt(Stmt::Block {
                statements: vec![
                    loc_stmt(Stmt::Assignment {
                        target: loc_expr(Expr::Variable("result_str".to_string())),
                        expression: loc_expr(Expr::Str("small".to_string())),
                    }),
                ]
            })),
        }), // result_str = "big"

        // var run_loop = true;
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("run_loop".to_string())),
            expression: loc_expr(Expr::Bool(true)),
        }),

        // var my_list = [100];
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("my_list".to_string())),
            expression: loc_expr(Expr::List(vec![
                loc_expr(Expr::Int(100)),
            ])),
        }),

        // while (run_loop) { ... }
        loc_stmt(Stmt::While {
            condition: loc_expr(Expr::Variable("run_loop".to_string())),
            body: Box::new(loc_stmt(Stmt::Block {
                statements: vec![
                    // my_list[0] = my_list[0] + z;
                    loc_stmt(Stmt::Assignment {
                        target: loc_expr(Expr::Indexing {
                            indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                            indexer: Box::new(loc_expr(Expr::Int(0))),
                        }),
                        expression: loc_expr(Expr::BinOp {
                            op: BinOp::Add,
                            left: Box::new(loc_expr(Expr::Indexing {
                                indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                                indexer: Box::new(loc_expr(Expr::Int(0))),
                            })),
                            right: Box::new(loc_expr(Expr::Variable("z".to_string()))),
                        }),
                    }), // my_list[0] = 100 + 25 = 125
                    // run_loop = false;
                    loc_stmt(Stmt::Assignment {
                        target: loc_expr(Expr::Variable("run_loop".to_string())),
                        expression: loc_expr(Expr::Bool(false)),
                    }),
                ]
            })),
        }), // my_list[0] is 125

        // var func_result = test_func(my_list[0]);
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("func_result".to_string())),
            expression: loc_expr(Expr::FunctionCall {
                function: Box::new(loc_expr(Expr::FunctionPtr("test_func".to_string()))),
                positional_arguments: vec![
                    CallArgument {
                        expression: Box::new(loc_expr(Expr::Indexing {
                            indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                            indexer: Box::new(loc_expr(Expr::Int(0))),
                        })), // Pass 125
                        loc: 0..0
                    },
                ],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
            }),
        }), // func_result = test_func(125) = 250

        // return (z, result_str, my_list[0], func_result);
        loc_stmt(Stmt::Return {
            expression: loc_expr(Expr::Tuple(vec![
                loc_expr(Expr::Variable("z".to_string())), // 25
                loc_expr(Expr::Variable("result_str".to_string())), // "big"
                loc_expr(Expr::Indexing {
                    indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                    indexer: Box::new(loc_expr(Expr::Int(0))),
                }), // 125
                loc_expr(Expr::Variable("func_result".to_string())), // 250
            ])),
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
        _ => todo!()
    }
}

fn main() {
    println!("--- Interpreter Test ---");

    let mut state = interpreter::State::new();

    // --- NEW: Define a function for the program ---
    // Helper to create placeholder type literals, as the interpreter doesn't use them
    fn dummy_type_literal() -> ast::LocTypeLiteral {
        ast::LocTypeLiteral {
            expr: ast::TypeLiteral::Void,
            loc: 0..0,
        }
    }

    let mut functions = HashMap::new();
    // Define `fn test_func(p1) { return p1 + p1; }`
    let test_func = ast::Function {
        name: "test_func".to_string(),
        contract: ast::FunctionPrototype {
            positional_arguments: vec![ast::Argument {
                name: "p1".to_string(),
                arg_type: dummy_type_literal(),
                loc: 0..0,
            }],
            variadic_argument: None,
            keyword_arguments: vec![],
            keyword_variadic_argument: None,
            return_type: dummy_type_literal(),
        },
        body: Box::new(loc_stmt(Stmt::Return {
            expression: loc_expr(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(loc_expr(Expr::Variable("p1".to_string()))),
                right: Box::new(loc_expr(Expr::Variable("p1".to_string()))),
            }),
        })),
        loc: 0..0,
    };
    functions.insert("test_func".to_string(), test_func);

    let program = ast::Program { functions };
    // --- END NEW ---


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