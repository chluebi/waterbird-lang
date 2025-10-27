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
    //   // --- Original Tests ---
    //   var my_add = fn(a, b) { a + b };
    //   var z = my_add(10, 15); // z = 25
    //   var result_str = "";
    //   if (true) { ... } else { ... } // result_str = "big"
    //   var run_loop = true;
    //   var my_list = [100];
    //   while (run_loop) { ... } // my_list = [125]
    //
    //   // --- NEW: 1. Different Operators ---
    //   var ops_test = (((10 * 2) - 5) / 3) + 1; // ops_test = 6
    //   var ops_bool = (true && !false) || (5 > 10); // ops_bool = true
    //
    //   // --- NEW: 2. Block returning an expression ---
    //   var block_val = {
    //     var inner = 2;
    //     inner * 5
    //   }; // block_val = 10
    //
    //   // --- NEW: 3. Block running by itself (for scope) ---
    //   {
    //     var scoped_var = "should not be visible";
    //   }
    //
    //   // --- NEW: 4. Recursive Unpacking ---
    //   var (unpack_a, [unpack_b, unpack_c]) = (1, [2, 3]);
    //   // unpack_a = 1, unpack_b = 2, unpack_c = 3
    //
    //   // --- NEW: 5. Recursion (call to 'factorial') ---
    //   var fact_result = factorial(5); // fact_result = 120
    //
    //   // --- Original Test ---
    //   var func_result = test_func(my_list[0]); // func_result = test_func(125) = 250
    //
    //   // --- Updated Return ---
    //   return (z, result_str, my_list[0], func_result, block_val, ops_test, ops_bool, fact_result, unpack_a, unpack_b, unpack_c);
    //   // Expected: (25, "big", 125, 250, 10, 6, true, 120, 1, 2, 3)
    // }

    let statements = vec![
        // --- Original Tests ---
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

        // ---
        // --- NEW FEATURES START HERE ---
        // ---

        // 1. Different Operators
        // var ops_test = (((10 * 2) - 5) / 3) + 1; // 6
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("ops_test".to_string())),
            expression: loc_expr(Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(loc_expr(Expr::BinOp {
                    op: BinOp::Div,
                    left: Box::new(loc_expr(Expr::BinOp {
                        op: BinOp::Sub,
                        left: Box::new(loc_expr(Expr::BinOp {
                            op: BinOp::Mul,
                            left: Box::new(loc_expr(Expr::Int(10))),
                            right: Box::new(loc_expr(Expr::Int(2))),
                        })), // 20
                        right: Box::new(loc_expr(Expr::Int(5))),
                    })), // 15
                    right: Box::new(loc_expr(Expr::Int(3))),
                })), // 5
                right: Box::new(loc_expr(Expr::Int(1))),
            }), // 6
        }),
        // var ops_bool = (true && !false) || (5 > 10); // true
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("ops_bool".to_string())),
            expression: loc_expr(Expr::BinOp {
                op: BinOp::Or,
                left: Box::new(loc_expr(Expr::BinOp {
                    op: BinOp::And,
                    left: Box::new(loc_expr(Expr::Bool(true))),
                    right: Box::new(loc_expr(Expr::Unop { // !false
                        op: UnOp::Not,
                        expr: Box::new(loc_expr(Expr::Bool(false))),
                    })),
                })), // true
                right: Box::new(loc_expr(Expr::BinOp { // 5 > 10
                    op: BinOp::Gt,
                    left: Box::new(loc_expr(Expr::Int(5))),
                    right: Box::new(loc_expr(Expr::Int(10))),
                })), // false
            }), // true
        }),

        // 2. Block returning an expression
        // var block_val = { var inner = 2; inner * 5 }; // 10
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("block_val".to_string())),
            expression: loc_expr(Expr::Block {
                statements: vec![
                    // var inner = 2;
                    loc_stmt(Stmt::Assignment {
                        target: loc_expr(Expr::Variable("inner".to_string())),
                        expression: loc_expr(Expr::Int(2)),
                    }),
                    // inner * 5
                    loc_stmt(Stmt::Expression {
                        expression: loc_expr(Expr::BinOp {
                            op: BinOp::Mul,
                            left: Box::new(loc_expr(Expr::Variable("inner".to_string()))),
                            right: Box::new(loc_expr(Expr::Int(5))),
                        })
                    })
                ]
            })
        }), // block_val = 10

        // 3. Block running by itself (for scope)
        // { var scoped_var = "..."; }
        loc_stmt(Stmt::Block {
            statements: vec![
                loc_stmt(Stmt::Assignment {
                    target: loc_expr(Expr::Variable("scoped_var".to_string())),
                    expression: loc_expr(Expr::Str("should not be visible".to_string())),
                })
            ]
        }),

        // 4. Recursive Unpacking
        // var (unpack_a, [unpack_b, unpack_c]) = (1, [2, 3]);
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Tuple(vec![
                loc_expr(Expr::Variable("unpack_a".to_string())),
                loc_expr(Expr::List(vec![
                    loc_expr(Expr::Variable("unpack_b".to_string())),
                    loc_expr(Expr::Variable("unpack_c".to_string())),
                ]))
            ])),
            expression: loc_expr(Expr::Tuple(vec![
                loc_expr(Expr::Int(1)),
                loc_expr(Expr::List(vec![
                    loc_expr(Expr::Int(2)),
                    loc_expr(Expr::Int(3)),
                ]))
            ])),
        }), // a=1, b=2, c=3

        // 5. Recursion (call to 'factorial')
        // var fact_result = factorial(5); // 120
        loc_stmt(Stmt::Assignment {
            target: loc_expr(Expr::Variable("fact_result".to_string())),
            expression: loc_expr(Expr::FunctionCall {
                function: Box::new(loc_expr(Expr::FunctionPtr("factorial".to_string()))),
                positional_arguments: vec![
                    CallArgument {
                        expression: Box::new(loc_expr(Expr::Int(5))),
                        loc: 0..0
                    },
                ],
                variadic_argument: None,
                keyword_arguments: vec![],
                keyword_variadic_argument: None,
            }),
        }), // fact_result = 120

        // ---
        // --- END OF NEW FEATURES ---
        // ---

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

        // return (z, result_str, my_list[0], func_result, ...new_values...);
        loc_stmt(Stmt::Return {
            expression: loc_expr(Expr::Tuple(vec![
                loc_expr(Expr::Variable("z".to_string())), // 25
                loc_expr(Expr::Variable("result_str".to_string())), // "big"
                loc_expr(Expr::Indexing {
                    indexed: Box::new(loc_expr(Expr::Variable("my_list".to_string()))),
                    indexer: Box::new(loc_expr(Expr::Int(0))),
                }), // 125
                loc_expr(Expr::Variable("func_result".to_string())), // 250
                
                // --- NEWLY ADDED FOR VERIFICATION ---
                loc_expr(Expr::Variable("block_val".to_string())), // 10
                loc_expr(Expr::Variable("ops_test".to_string())), // 6
                loc_expr(Expr::Variable("ops_bool".to_string())), // true
                loc_expr(Expr::Variable("fact_result".to_string())), // 120
                loc_expr(Expr::Variable("unpack_a".to_string())), // 1
                loc_expr(Expr::Variable("unpack_b".to_string())), // 2
                loc_expr(Expr::Variable("unpack_c".to_string())), // 3
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

    // --- NEW: Add recursive factorial function ---
    // Define `fn factorial(n) { 
    //   if (n == 0) { return 1; }
    //   else { return n * factorial(n - 1); }
    // }`
    let factorial_func = ast::Function {
        name: "factorial".to_string(),
        contract: ast::FunctionPrototype {
            positional_arguments: vec![ast::Argument {
                name: "n".to_string(),
                arg_type: dummy_type_literal(),
                loc: 0..0,
            }],
            variadic_argument: None,
            keyword_arguments: vec![],
            keyword_variadic_argument: None,
            return_type: dummy_type_literal(),
        },
        body: Box::new(loc_stmt(Stmt::IfElse {
            // condition: n == 0
            condition: loc_expr(Expr::BinOp {
                op: BinOp::Eq,
                left: Box::new(loc_expr(Expr::Variable("n".to_string()))),
                right: Box::new(loc_expr(Expr::Int(0))),
            }),
            // if_body: return 1;
            if_body: Box::new(loc_stmt(Stmt::Return {
                expression: loc_expr(Expr::Int(1)),
            })),
            // else_body: return n * factorial(n - 1);
            else_body: Box::new(loc_stmt(Stmt::Return {
                expression: loc_expr(Expr::BinOp {
                    op: BinOp::Mul,
                    // n
                    left: Box::new(loc_expr(Expr::Variable("n".to_string()))),
                    // factorial(n - 1)
                    right: Box::new(loc_expr(Expr::FunctionCall {
                        function: Box::new(loc_expr(Expr::FunctionPtr("factorial".to_string()))),
                        positional_arguments: vec![
                            CallArgument {
                                expression: Box::new(loc_expr(Expr::BinOp {
                                    op: BinOp::Sub,
                                    left: Box::new(loc_expr(Expr::Variable("n".to_string()))),
                                    right: Box::new(loc_expr(Expr::Int(1))),
                                })),
                                loc: 0..0
                            }
                        ],
                        variadic_argument: None,
                        keyword_arguments: vec![],
                        keyword_variadic_argument: None,
                    })),
                }),
            })),
        })),
        loc: 0..0,
    };
    functions.insert("factorial".to_string(), factorial_func);
    // --- END NEW ---


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